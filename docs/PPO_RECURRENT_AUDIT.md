# Аудит ActorCritic перед добавлением LSTM (POMDP)

> **Примечание (документ исторический).**  
> Актуальное и полное описание архитектуры ActorCritic, включая encoder, LSTM и головы actor/critic,
> находится в `docs/PPO_MODEL.md`. Настоящий файл сохранён как аудит исходного варианта модели
> перед добавлением LSTM и не должен использоваться как источник правды по текущей архитектуре.

## Encoder
- **Модуль:** `self.encoder = nn.Sequential(base.conv, nn.Flatten())`, где base = BCNet или BCNetDeep.
- **Выход:** вектор фиксированной длины.
  - **arch=default (BCNet):** 64×7×7 = **3136**.
  - **arch=deep (BCNetDeep):** 128×5×5 = **3200**.
- Вход: `(B, in_channels, 84, 84)`, in_channels = FRAME_STACK (4).

## Actor / Critic
- **Actor:** `Linear(enc_size, 512)` → ReLU → `Linear(512, num_actions)`. Вход = выход encoder.
- **Critic:** `Linear(enc_size, 512)` → ReLU → `Linear(512, 1)`.
- Обе головы работают с одним и тем же представлением `h` после encoder.

## Forward
- **Сигнатура:** `forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]`.
- **Возврат:** `(logits, value)`; logits `(B, num_actions)`, value `(B, 1)`.
- Документация в коде упоминает третий элемент (log_probs), фактически возвращаются только logits и value.

## get_action
- **Сигнатура:** `get_action(self, x: torch.Tensor, deterministic: bool = False) -> tuple[int, torch.Tensor, torch.Tensor]`.
- **Возврат:** `(action, log_prob, value)` для одного примера (batch size 1).
- Внутри: `forward(x)` → Categorical(logits) → sample/argmax → log_prob.

## Использование в train_ppo
- Rollout: по одному наблюдению на env вызывается `model.get_action(x, deterministic=False)`; x имеет форму `(1, C, 84, 84)`.
- GAE: траектории разбиты по env; для последнего шага каждого env при не-done вызывается `model.get_action(x_last, deterministic=True)` для last_value.
- PPO update: `model(obs_t[mb])` → logits, values для мини-батча.

## Вывод для LSTM
- LSTM вставить **между** encoder и головами: `encoder(x)` → `lstm(h)` → actor/critic.
- Размер входа LSTM: enc_size (3136 или 3200); выход: lstm_hidden_size (задаётся флагом).
- Отдельные (h, c) на каждый env; при done обнулять (h, c) этого env.
- Сохранить обратную совместимость: при `recurrent=False` поведение и сигнатуры без изменений.

## Реализация (после доработок)

- **ActorCritic:** опциональный `nn.LSTM(enc_size, lstm_hidden_size, batch_first=True)`; `forward(x, hidden=None)` возвращает при recurrent `(logits, value, h_n, c_n)`. `get_action(x, deterministic, hidden)` возвращает `(action, log_prob, value, next_hidden)`; при non-recurrent `next_hidden=None`.
- **train_ppo:** при `--recurrent` хранятся `roll_h`, `roll_c` по шагам rollout; при done скрытое состояние env обнуляется; GAE и PPO update используют сохранённые (h,c) для повторного прохода. Чекпоинт сохраняет `recurrent` и `lstm_hidden_size`.
- **Диагностика:** в лог выводится `h_norm` (средняя норма скрытого состояния по rollout) при recurrent.
- **Бенчмарк:** см. `docs/PPO_RECURRENT_BENCHMARK.md`.
