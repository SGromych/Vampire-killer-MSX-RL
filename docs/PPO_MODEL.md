# Архитектура PPO-модели (ActorCritic)

Этот документ является **основным источником правды (single source of truth)** по архитектуре ActorCritic:
encoder, опциональный LSTM и головы actor/critic.

Модуль: `msx_env/ppo_model.py`. Дополнительные материалы:
`msx_env/bc_model.py` (encoder), `docs/PPO_RECURRENT_AUDIT.md` (исторический аудит перед добавлением LSTM),
`docs/PPO_RECURRENT_BENCHMARK.md` (краткий бенчмарк recurrent‑режима).

---

## 1. Общая схема

```
obs (B, C, 84, 84)  →  Encoder  →  [LSTM]  →  Actor/Critic heads  →  action, value
```

- **Encoder:** CNN (как BCNet/BCNetDeep) — извлекает признаки из стопки кадров.
- **LSTM (опционально):** между encoder и головами, хранит скрытое состояние между шагами (память для лабиринта).
- **Actor head:** линейные слои → logits (10 действий) → Categorical.
- **Critic head:** линейные слои → value (скаляр).

---

## 2. Encoder

Вход: `(B, C, 84, 84)`, где C = FRAME_STACK (4 кадра по умолчанию).

### arch=default (BCNet)

| Слой | Описание | Вход | Выход |
|------|----------|------|-------|
| Conv2d | 32 фильтров 8x8, stride 4 | (B, 4, 84, 84) | (B, 32, 20, 20) |
| ReLU | | | |
| Conv2d | 64 фильтров 4x4, stride 2 | (B, 32, 20, 20) | (B, 64, 9, 9) |
| ReLU | | | |
| Conv2d | 64 фильтров 3x3, stride 1 | (B, 64, 9, 9) | (B, 64, 7, 7) |
| ReLU | | | |
| Flatten | | (B, 64, 7, 7) | (B, **3136**) |

### arch=deep (BCNetDeep)

| Слой | Описание | Выход |
|------|----------|-------|
| Conv2d 32@8x8 s4 | | (B, 32, 20, 20) |
| ReLU | | |
| Conv2d 64@4x4 s2 | | (B, 64, 9, 9) |
| ReLU | | |
| Conv2d 64@3x3 s1 | | (B, 64, 7, 7) |
| ReLU | | |
| Conv2d 128@3x3 s1 | | (B, 128, 5, 5) |
| ReLU | | |
| Flatten | | (B, **3200**) |

---

## 3. LSTM (память, --recurrent)

При `recurrent=True` между encoder и головами вставлен LSTM:

| Параметр | Значение | Описание |
|----------|----------|----------|
| Вход LSTM | enc_size (3136 или 3200) | вектор от encoder |
| Скрытое состояние | lstm_hidden_size (по умолчанию 256) | h, c |
| Выход LSTM | lstm_hidden_size | вектор для actor/critic |

**Поведение:**
- Скрытое состояние (h, c) передаётся от шага к шагу внутри эпизода.
- При `done=True` (смерть, stuck, timeout) скрытое состояние **обнуляется** (`zero_hidden`).
- LSTM даёт «память» — агент может учитывать, где уже был в лабиринте (POMDP).

**Как проверить, что память включена и работает:**

1. **Чекпоинт:** в `checkpoints/ppo/.../config_snapshot.json` поле `"recurrent": true` — память включена.
2. **Лог:** при `--recurrent` в summary выводится `h_norm=...` — средняя норма скрытого состояния LSTM. Если h_norm > 0 и меняется — LSTM активен.
3. **CLI:** запуск с `--recurrent` — `python train_ppo.py --recurrent --lstm-hidden-size 256`.
4. **Число параметров:** `recurrent=True` добавляет ~1.5M параметров (LSTM).

---

## 4. Головы Actor и Critic

**Actor:**
- `Linear(head_in, 512)` → ReLU
- `Linear(512, num_actions)` → logits
- Categorical(logits) → sample / argmax

**Critic:**
- `Linear(head_in, 512)` → ReLU
- `Linear(512, 1)` → value

`head_in` = lstm_hidden_size (256) при recurrent, иначе enc_size (3136 или 3200).

---

## 5. Действия (action space)

10 дискретных действий: NOOP, RIGHT, LEFT, UP, DOWN, ATTACK, RIGHT_JUMP, LEFT_JUMP, RIGHT_JUMP_ATTACK, LEFT_JUMP_ATTACK.

---

## 6. Инициализация из BC

При `--bc-checkpoint` encoder (и при non-recurrent — actor) инициализируются весами из BC. При recurrent копируется только encoder (головы питаются от LSTM, размер входа другой).
