# Чеклист улучшений reward/обучения (по советам и актуальным конфигам)

Проверено по: `configs/reward_v3.json`, `configs/night_training.json`, код в `msx_env/reward/` и `train_ppo.py`. Цель: агент стабильно проходит уровень и обходит комнаты; убрать ложный unique_rooms; лучше использовать LSTM.

---

## Какие конфиги используются в боевых прогонах

| Роль | Файл (от корня репо) | Когда читается |
|------|----------------------|----------------|
| **Главный конфиг запуска** (тренер, супервизор, пути, PPO, reward_config) | **`configs/night_training.json`** | Супервизор при старте/рестарте: оттуда берёт параметры для первого запуска и путь к reward. |
| **Конфиг наград** (reward: novelty, key, door, stuck, комнаты и т.д.) | **`configs/reward_v3.json`** | Указан в `night_training.json` как `"reward_config": "configs/reward_v3.json"`. Загружается train_ppo при старте (и при рестарте из снимка). |
| **Снимок конфига прогона** (заморозка на время run) | **`runs/<run_name>/config_snapshot.json`** | При рестарте супервизор запускает train_ppo с `--config runs/.../config_snapshot.json --resume`. Все параметры (в т.ч. reward) берутся из снимка, а не из night_training.json. |

Итого: чтобы менять настройки для **новых** прогонов — правь **`configs/night_training.json`** и **`configs/reward_v3.json`**. Уже запущенный прогон при рестарте продолжает работать по **`runs/<run_name>/config_snapshot.json`** (он создаётся при первом запуске train_ppo).

---

## 1. Детектор «комнат» и unique_rooms (главная причина глюка)

**Как сейчас:** В метриках два разных счётчика:
- **unique_rooms** (в CSV) — из **novelty**: `block_mean_hash` с `room_hash_crop_top=14`, стабильность = `novelty_stability_frames` (в v3 = 3 кадра). Это легко «дребезжит» из‑за HUD/анимации.
- **unique_rooms_ep** — из эпизодных метрик: `stable_room_hash_playfield` (playfield, blur+quantize) + `episode_room_debounce_k=7`.

**Что имеет смысл сделать:**

| Действие | Где | Смысл |
|----------|-----|--------|
| Увеличить стабильность хэша для novelty | `configs/reward_v3.json` | `novelty_stability_frames`: 3 → **8–12**. Меньше ложных «новых комнат» из‑за мерцания. |
| Усилить debounce для эпизодных метрик | `configs/reward_v3.json` | `episode_room_debounce_k`: 7 → **15–20**. Комната не переключается каждые 7 кадров. |
| (Опционально) Кап на novelty за эпизод | `configs/reward_v3.json` | `novelty_saturation_cap`: 0 → **25–40** (и оставить decay). Чтобы novelty не доминировала и не фармилась. |
| Не трогать в конфиге (уже есть в коде) | — | Playfield‑хэш для **unique_rooms_ep** уже с crop/blur/quantize в `stable_room_hash_playfield`. Для **награды** novelty по-прежнему используется `block_mean_hash` с crop_top=14. Имеет смысл со временем считать novelty по тому же playfield‑хэшу (требует правки кода). |

---

## 2. Нормализация novelty reward

**Как сейчас:** `novelty_reward=0.35`, `novelty_saturation_cap=0`, `novelty_persistence_frames=3` (в v3 для стабильности используется `novelty_stability_frames=3`).

**Что имеет смысл сделать:**

| Действие | Где | Смысл |
|----------|-----|--------|
| Включить кап | `configs/reward_v3.json` | `novelty_saturation_cap`: 0 → **30** (или 40), `novelty_saturation_decay`: оставить 0.95. |
| После стабилизации room_id снизить вес | `configs/reward_v3.json` | После пункта 1: `novelty_reward`: 0.35 → **0.15–0.25**, чтобы не фармить новизну в ущерб ключу/двери. |

---

## 3. Цели уровня: ключ / дверь / stage

**Как сейчас (reward_v3.json):** `key_reward=0.3`, `door_reward=0.5`, `enable_stage_reward=false`.

**Что имеет смысл сделать:**

| Действие | Где | Смысл |
|----------|-----|--------|
| Поднять награды за ключ и дверь | `configs/reward_v3.json` | `key_reward`: 0.3 → **1.0–1.5**; `door_reward`: 0.5 → **2.0–3.0**. Чтобы агент имел явный стимул к прогрессу. |
| Включить stage reward | `configs/reward_v3.json` | `enable_stage_reward`: false → **true**. Оставить `stage_advance_bonus=2.0` или поднять до **3.0**. |
| progress_reward_per_pixel | Оставить 0 | Нет надёжного сигнала прогресса по пикселям в коде — не включать без реализации. |

---

## 4. Death penalty vs шкала наград

**Как сейчас:** `death_penalty=-1.0`, `step_penalty=-0.001`. Один смертельный эпизод ≈ −1.0, длинный эпизод без прогресса ≈ сотни шагов × (−0.001).

**Что имеет смысл сделать:**

| Действие | Где | Смысл |
|----------|-----|--------|
| Вариант A: чуть снизить death | `configs/reward_v3.json` | `death_penalty`: -1.0 → **-0.5…-0.7**, чтобы градиент не забивался одним событием. |
| Вариант B (предпочтительно): усилить ключ/дверь/stage | П.3 выше | Оставить death=-1.0, но поднять key/door/stage так, чтобы «правильные» действия давали заметный положительный return. |

---

## 5. Stuck / ping-pong: не душить эпизоды раньше времени

**Как сейчас:** `stuck_no_room_change_steps=200`, `stuck_truncate=true`, `stuck_penalty=-0.3`. При сломанном room_id «нет смены комнаты» может срабатывать даже при движении.

**Что имеет смысл сделать:**

| Действие | Где | Смысл |
|----------|-----|--------|
| Увеличить окно до срабатывания stuck | `configs/reward_v3.json` | `stuck_no_room_change_steps`: 200 → **350–500**. Меньше ранних обрезок эпизода. |
| На время отладки отключить обрезку по stuck | `configs/reward_v3.json` | `stuck_truncate`: true → **false** (штраф оставить). После стабилизации room_id можно вернуть true. |
| Ping-pong | Оставить как есть | Уже считается по stable room_hash в policy; после стабилизации хэша будет адекватнее. |

---

## 6. PPO + LSTM: больше данных на обновление

**Как сейчас:** В снимке и по умолчанию: `rollout_steps=128`, `num_envs=1`. Супервизор не передаёт `rollout_steps` — берётся дефолт из train_ppo.

**Что имеет смысл сделать:**

| Действие | Где | Смысл |
|----------|-----|--------|
| Увеличить rollout_steps | Добавить в `configs/night_training.json` и в `train_supervisor.build_argv()` передачу `--rollout-steps` | `rollout_steps`: 128 → **256** (или 512 при достаточном CPU). Больше контекста на одно обновление для LSTM. |
| (Опционально) num_envs | `configs/night_training.json` | При возможности: **num_envs 2–4** для стабильности градиента (нужны ресурсы и настройка окон при window capture). |

Сейчас супервизор при первом старте **не передаёт** `--rollout-steps` и `--entropy-coef` (в `build_argv()` их нет), поэтому train_ppo берёт дефолты 128 и 0.01. Чтобы значения из night_training применялись: добавить в `configs/night_training.json` поля `rollout_steps` и при желании `entropy_coef`, и в `train_supervisor.build_argv()` при сборке argv по cfg добавить: `--rollout-steps`, `--entropy-coef` (если ключи есть в cfg).

---

## 7. Энтропия: один источник правды

**Как сейчас:** В снимке: `entropy_coef=0.01`. В `night_training.json`: `entropy_floor=0.3` — это **порог для предупреждения** в train_ppo («низкая энтропия»), а не коэффициент в loss. Рассинхрона нет: коэффициент = 0.01, floor = порог для guardrail.

**Что имеет смысл сделать:**

| Действие | Где | Смысл |
|----------|-----|--------|
| Для ранней фазы чуть поднять entropy_coef | Добавить в `configs/night_training.json` и передавать в тренер (как rollout_steps) | `entropy_coef`: 0.01 → **0.02–0.03** для большей разведки в начале. |
| Оставить entropy_floor=0.3 | — | Используется только для предупреждений. |

Сейчас супервизор не передаёт `--entropy-coef`; значение берётся из дефолта train_ppo (0.01). Чтобы менять из ночного конфига — добавить в `night_training.json` и в `build_argv()`.

---

## Приоритет внедрения (коротко)

1. **Сначала (только конфиг):** п.1 (novelty_stability_frames 8–12, episode_room_debounce_k 15–20), п.3 (key/door выше, enable_stage_reward true), п.5 (stuck_no_room_change_steps 350–500, stuck_truncate false).
2. **Потом:** п.2 (novelty_saturation_cap, при необходимости снизить novelty_reward), п.4 (решить: чуть снизить death или усилить key/door/stage).
3. **Отдельно (супервизор + конфиг):** п.6 (rollout_steps в night_training + build_argv), п.7 (entropy_coef в night_training + build_argv при желании).

После изменений в `configs/reward_v3.json` ночные прогоны подхватят их автоматически (reward_config уже указывает на этот файл). Изменения в `night_training.json` для rollout_steps и entropy_coef потребуют доп. передачи соответствующих флагов в `build_argv()`.

---

## 8. Новые механизмы (door shaping, block break, novelty-after-key)

### 8.1. Door approach shaping (будет включён отдельно)

**Что добавлено в `RewardConfig` и `reward_v3.json`:**

- `enable_door_distance_reward: bool` (по умолчанию `false`)
- `door_distance_reward: float` (по умолчанию `0.01`)
- `door_distance_min_delta: float` (по умолчанию `0.0`)
- `door_distance_clip: float` (по умолчанию `0.05`)
- `door_distance_requires_key: bool` (по умолчанию `true`)

**Текущее состояние реализации:**

- В `RewardPolicy` добавлены флаги и защита: если `enable_door_distance_reward=true`, но нет надёжного детектора расстояния до двери, в лог один раз выводится предупреждение и **shaping не применяется**.
- Причина: в текущем коде нет сигнала «позиция двери» (нет RAM/tilemap‑детектора); вводить эвристику по raw‑кадру рискованно (ложные сигналы, ломает обучение).

**Когда включать:**

- Пока **не включать** в боевых конфигурациях (оставить `enable_door_distance_reward=false`), пока не появится надёжный детектор двери (например по тайловой карте или координатам объекта из RAM).

### 8.2. Block-break reward (будет включён отдельно)

**Новые поля в `RewardConfig` и `reward_v3.json`:**

- `enable_block_break_reward: bool` (по умолчанию `false`)
- `block_break_reward: float` (по умолчанию `0.1`)
- `block_break_debounce_frames: int` (по умолчанию `8`)

**Текущее состояние реализации:**

- Аналогично door shaping: если `enable_block_break_reward=true`, `RewardPolicy` один раз пишет в лог предупреждение, что детектор разрушения блоков не реализован, и **дополнительная награда не выдаётся**.
- В коде сейчас нет надёжного сигнала «разрушен тайл/блок» (ни по RAM, ни по tilemap), только по кадрам; делать эвристику по diff кадров без контекста слишком шумно.

**Рекомендация:**

- Держать `enable_block_break_reward=false` до тех пор, пока не будет реализован детектор в окружении (например сравнение тайлов playfield между кадрами).

### 8.3. Снижение novelty после получения ключа (включено в v3)

**Новые поля:**

- `novelty_after_key_multiplier: float` (по умолчанию `0.25` в `reward_v3.json`)

**Как работает:**

- В `RewardPolicyState` добавлено поле `after_key` и счётчики:
  - `novelty_reward_pre_key_sum`
  - `novelty_reward_post_key_sum`
- В `RewardPolicy.compute()`:
  - novelty считается через `novelty_component` как раньше;
  - если `after_key=True` и `0 < novelty_after_key_multiplier < 1`, вклад novelty умножается на этот множитель;
  - до получения ключа сумма идёт в `novelty_reward_pre_key_sum`, после — в `novelty_reward_post_key_sum`;
  - в `extra` попадают:
    - `novelty_reward_pre_key_sum`
    - `novelty_reward_post_key_sum`
    - `novelty_after_key_multiplier_used`.
- Флаг `after_key` становится `True`, когда:
  - HUD показывает ключ (`key_chest` или `key_door`),
  - или сработал `key_reward` (детектор ключа),
  - или `extra["key_detected"]` от `KeyDetector`.

**Зачем это нужно:**

- До ключа novelty поощряет исследование лабиринта.
- После ключа вклад novelty ослабляется (×0.25), чтобы политика была менее мотивирована «гулять» и больше — искать дверь/выход.

**Как проверять:**

- Запустить с `--debug` и посмотреть последние строки отчёта по эпизоду в `train.log`:
  - поля `reward_novelty`, `reward_key`, `reward_door` в суммарной строке.
  - при необходимости добавить вывод `novelty_reward_pre_key_sum` / `novelty_reward_post_key_sum` из `info["reward_*"]`.
- В `metrics.csv` сейчас эти суммы не агрегируются, но их можно читать из `info` при done (по желанию — добавить в отдельный диагностический скрипт).
