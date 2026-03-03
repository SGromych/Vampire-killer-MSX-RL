# Модули, флаги и выходы проекта

**Назначение документа:** единая справочная точка по всем точкам входа, флагам, каталогам вывода и связям между модулями. Документ дополняется при появлении новых скриптов, опций и выходов.

Связанные документы:
- **`docs/CONTEXT.md`** — цели проекта, архитектура, ключевые решения.
- **`docs/PPO_MODEL.md`** — архитектура PPO (encoder, LSTM, actor/critic), как проверить память (LSTM).
- **`docs/TRAINING.md`** — PPO: аудит, стабильность, эксперименты (--run-name, --config, --reward-config), guardrails.
- **`docs/REWARD.md`** — система наград (компоненты, масштабы, отладка).
- **`docs/CAPTURE.md`** — бэкенды захвата (png/single/window), калибровка окна, бенчмарк.
- **`docs/CAPTURE_REFACTOR_ANALYSIS.md`** — анализ и рефакторинг захвата кадров.
- **`docs/MULTI_ENV_DEBUG_SESSION.md`** — саммари отладки multi-env (num_envs>1): изменения, ошибки, env0 vs env1, рекомендации и рефакторинг.

---

## 1. Структура проекта (ключевые пути)

```
RL_msx/
├── msx_env/           # Окружение и модели
│   ├── env.py         # VampireKillerEnv, EnvConfig
│   ├── env_diagnostics.py  # Реестр ресурсов multi-env, вывод ENV RESET/TERMINATION при debug
│   ├── capture.py     # Бэкенды захвата (png, single, window)
│   ├── make_env.py    # Фабрика env для num_envs (уникальный workdir)
│   ├── hud_parser.py  # HUD → награда за подбор
│   ├── life_bar.py    # Оценка полоски жизни (смерть)
│   ├── dataset.py     # Датасет демо: save/load/validate
│   ├── bc_model.py    # BCNet, BCNetDeep, load_bc_checkpoint
│   ├── ppo_model.py   # ActorCritic, load_ppo_checkpoint, init_from_bc
│   ├── human_controller.py  # Клавиши pygame → action id
│   ├── replay_utils.py       # Сборка preview.mp4 из obs
│   └── reward/        # Модульная система наград
│       ├── config.py  # RewardConfig, default_v1_config
│       ├── policy.py  # RewardPolicy
│       ├── components.py, hashers.py, logger.py, event_detectors.py
│       └── __init__.py
├── openmsx_bridge.py  # Управление openMSX (Tcl, screenshot, keymatrix)
├── demos/
│   ├── record_demo.py   # Запись демо человеком
│   ├── validate_demo.py # Валидация рана
│   └── replay_demo.py   # Просмотр записанного рана
├── train_bc.py         # Обучение BC
├── train_ppo.py        # Обучение PPO
├── test_policy.py      # Тест BC-политики в игре
├── test_ppo.py         # Тест PPO-политики в игре
├── benchmark_env.py    # Бенчмарк env (png/single/window, avg/p95)
├── scripts/
│   ├── calibrate_window_capture.py  # Калибровка window capture, вывод JSON
│   └── run_2env_smoke.py             # Ручной smoke 2 env
├── tests/
│   ├── test_reward_smoke.py    # Smoke-тест политики наград
│   └── test_multi_env_smoke.py # Smoke изоляции workdir (make_env)
├── checkpoints/
│   ├── bc/             # Чекпоинты BC (best.pt, last.pt)
│   └── ppo/            # Чекпоинты PPO (last.pt, epoch_*.pt)
│       └── run/        # Workdir openMSX при обучении/тесте PPO
├── demos/runs/<run_id>/ # Один записанный прогон (data.npz, manifest.json, логи openMSX)
└── docs/               # Документация
```

---

## 2. Окружение (env)

**Модуль:** `msx_env/env.py`

**Классы:** `EnvConfig`, `VampireKillerEnv`.

**Назначение:** Gym-подобное окружение: `reset()` → obs, info; `step(action)` → obs, reward, terminated, truncated, info. Управление openMSX через `openmsx_bridge`, захват кадра через `msx_env/capture.py`.

### EnvConfig — параметры

| Параметр | По умолчанию | Влияние |
|----------|--------------|---------|
| `rom_path` | — | Путь к VAMPIRE.ROM. |
| `workdir` | — | Каталог для openMSX (commands.tcl, reply.txt, step_frame.png, логи). У каждого инстанса свой workdir для параллельных env. |
| `frame_size` | (84, 84) | Размер obs (H, W) после resize. Менять нельзя без смены формата датасетов/моделей. |
| `poll_ms` | 15 | Интервал опроса Tcl в openMSX (мс). Меньше — быстрее реакция, больше нагрузка. |
| `hold_keys` | True | Клавиши удерживаются до смены действия. False — импульсный режим (press с hold_ms). |
| `hud_reward` | True | В legacy-режиме (без reward_config) учитывать награду за подбор по HUD. |
| `terminated_on_death` | False | True: при падении жизни (life_bar) эпизод завершается как terminated, в reward добавляется штраф. Для PPO обычно True. |
| `max_episode_steps` | 0 | >0: эпизод обрезается (truncated) по достижении шагов. Для PPO задаётся (например 1500). |
| `action_repeat` | 1 | N внутренних шагов эмулятора на один захват кадра. >1 уменьшает число захватов за «логический» шаг. |
| `decision_fps` | None | Фиксированная частота решений (Гц). None — без ограничения. |
| `capture_backend` | "png" | Бэкенд захвата: "png", "single", "window". |
| `window_crop` | None | Для window: (x, y, w, h) в координатах экрана. |
| `window_title` | None | Для window: подстрока в заголовке окна (по умолчанию openMSX). |
| `capture_lag_ms` | 0 | Задержка (мс) перед grab (уменьшение tearing). |
| `post_action_delay_ms` | 0 | Задержка (мс) после нажатия клавиш перед grab; при num_envs>1 train_ppo ставит 50. |
| `reward_config` | None | None — legacy (HUD + death в env; step_penalty в train_ppo). Иначе RewardConfig (v1). |
| `instance_id` | None | Для логов и многопоточности; задаётся make_env(rank). |
| `tmp_root` | "runs/tmp" | База для workdir при num_envs > 1 (workdir = tmp_root/rank). |
| `soft_reset` | True | При reset() не убивать процесс openMSX, а «продолжить» клавишами (SPACE после Game Over); при False — полный перезапуск. |
| `perf_log_interval` | 0 | Каждые N шагов печатать в консоль `[perf]` (capture/prep/step). 0 = вывод отключён (рекомендуется). |
| `death_warmup_steps` | 50 | В первых N шагах не объявлять смерть (защита от ложных срабатываний по кадру). |
| `reset_handshake_stable_frames` | 0 | При num_envs>1 train_ppo по умолчанию ставит 3; 0 = выкл. См. `--no-reset-handshake`. |

### Выход env

- **obs:** `(84, 84)` uint8, grayscale.
- **info:** при legacy — `hud` (weapon, key_chest, key_door, items). При `reward_config` — плюс `reward_components`, `episode_reward_components`, `reward_unique_rooms`, `reward_stuck_truncate` и др.
- В конце эпизода при политике наград в консоль выводится отчёт `[reward] episode total=...`.

---

## 3. Захват кадра (capture)

**Модуль:** `msx_env/capture.py`

**Назначение:** абстракция захвата кадра. `FrameCaptureBackend`: `start()`, `grab()` → RGB (H,W,3), `close()`. Env вызывает `grab()`, делает resize до `frame_size` и grayscale → obs; тот же RGB передаётся в HUD/награды.

**Бэкенды:** `FileCapturePNG`, `FileCaptureSinglePath` — оба пишут в один файл (step_frame.png) в workdir. Фабрика: `make_capture_backend("png"|"single", emu, workdir, filename)`.

Калибровка window: **`python scripts/calibrate_window_capture.py --title openMSX`** (список окон, FPS, dump кадров, JSON-конфиг). Подробнее: **`docs/CAPTURE.md`**.

---

## 4. Запись демонстраций (demos)

### 4.1. record_demo.py

**Назначение:** запись одного прогона: человек играет в pygame, действия и кадры пишутся в датасет.

**Флаги:**

| Флаг | По умолчанию | Влияние |
|------|--------------|---------|
| `--run-id` | обязательный | Имя подкаталога: `demos/runs/<run_id>/`. |
| `--max-steps` | 2000 | Лимит шагов записи. |
| `--max-minutes` | 3.0 | Лимит времени (минуты). |
| `--fps` | 15 | Частота опроса клавиш. |
| `--poll-ms` | 15 | Передаётся в EnvConfig (опрос openMSX). |
| `--preview` | False | После записи создать preview.mp4. |
| `--action-repeat` | 1 | Передаётся в EnvConfig. |
| `--decision-fps` | None | Передаётся в EnvConfig. |
| `--capture-backend` | png | Передаётся в EnvConfig. |

**Куда пишется:** `demos/runs/<run_id>/data.npz`, `manifest.json`; в том же каталоге — логи openMSX, step_frame.png, служебные Tcl-файлы.

**Команда:**  
`python demos/record_demo.py --run-id test01 --max-steps 2000 --preview`

---

### 4.2. validate_demo.py

**Назначение:** проверка одного рана (размер, распределение действий, отличие кадров; опционально preview).

**Флаги:** `run_id` (позиционный), `--preview`.

**Команда:**  
`python demos/validate_demo.py test01 --preview`

---

### 4.3. replay_demo.py

**Назначение:** просмотр записанного рана: по кадрам (pygame) или через env (воспроизведение действий в окружении).

**Флаги:** `run_id`, `--mode` (frames | env), `--fps` (для frames).

**Команда:**  
`python demos/replay_demo.py test01 --mode frames --fps 10`

---

## 5. Обучение BC (train_bc.py)

**Назначение:** обучение политики имитации по демонстрациям. Загружает все прогоны из `demos/runs/` (или указанные `--runs`), строит frame-stacked obs, взвешенный CrossEntropy loss.

**Флаги:**

| Флаг | По умолчанию | Влияние |
|------|--------------|---------|
| `--runs` | все с data.npz | Список run_id. Пример: `--runs run_01 run_02`. |
| `--epochs` | 40 | Число эпох обучения. |
| `--batch-size` | 64 | Размер батча. |
| `--lr` | 1e-3 | Learning rate. |
| `--checkpoint-dir` | checkpoints/bc | Каталог для best.pt и last.pt. |
| `--device` | cuda/cpu | Устройство. |
| `--noop-weight` | 0.5 | Вес класса NOOP в loss. |
| `--oversample` | 2.0 | Во сколько раз чаще сэмплировать шаги с action≠NOOP. |
| `--frame-stack` | 4 | Число кадров в стопке. |
| `--deep` | False | True — BCNetDeep. |
| `--rare-weight` | 1.5 | Вес редких действий (ATTACK, прыжки). |
| `--move-weight` | 1.6 | Вес движения RIGHT/LEFT/UP/DOWN. |

**Куда пишется:**  
`<checkpoint-dir>/best.pt`, `<checkpoint-dir>/last.pt`. В каждом чекпоинте: `state_dict`, `frame_stack`, `arch` ("default" | "deep").

**Команды:**  
`python train_bc.py --epochs 40`  
`python train_bc.py --runs run_full_01 run_full_02 --deep --move-weight 1.6 --checkpoint-dir checkpoints/bc`

**Как понять, сколько обучений:** по дате/содержимому каталога `checkpoints/bc/` и по тому, какой чекпоинт подставляется в test_policy (best.pt или last.pt). Отдельного счётчика запусков нет.

---

## 6. Обучение PPO (train_ppo.py)

**Назначение:** обучение ActorCritic по reward. Rollout в env (rollout_steps шагов), GAE, несколько эпох PPO по мини-батчам. Может инициализироваться из BC.

**Флаги:**

| Флаг | По умолчанию | Влияние |
|------|--------------|---------|
| `--checkpoint-dir` | checkpoints/ppo | Каталог для last.pt и epoch_*.pt. |
| `--bc-checkpoint` | None | Путь к BC-чекпоинту для инициализации весов. |
| `--epochs` | 100 | Число PPO-обновлений (каждое = один rollout + обучение). |
| `--rollout-steps` | 128 | Шагов на один rollout. |
| `--ppo-epochs` | 3 | Эпох обучения на одном rollout. |
| `--batch-size` | 64 | Размер мини-батча. |
| `--lr`, `--gamma`, `--gae-lambda`, `--clip-eps` | стандартные | Гиперпараметры PPO. |
| `--max-episode-steps` | 1500 | Лимит шагов эпизода (truncated). |
| `--step-penalty` | -0.001 | В legacy-режиме добавляется к reward в скрипте; при reward_config в env не дублируется. |
| `--device` | cuda/cpu | Устройство. |
| `--arch` | deep | Архитектура (default | deep); может переопределяться из BC-чекпоинта. |
| `--num-envs` | 1 | Число параллельных env; при >1 каждый инстанс получает свой workdir (tmp_root/rank). |
| `--tmp-root` | runs/tmp | База для workdir при num_envs > 1 (workdir = tmp_root/0, tmp_root/1, …). |
| `--post-action-delay-ms` | 50 при num-envs>1, иначе 0 | Задержка (мс) после нажатия клавиш перед захватом кадра; при нескольких env даёт эмулятору время отрисовать кадр. |
| `--action-repeat` | 1 | Передаётся в EnvConfig. |
| `--decision-fps` | None | Передаётся в EnvConfig. |
| `--capture-backend` | png | Передаётся в EnvConfig. |
| `--run-name` | None | Имя эксперимента; логи в `<log-dir>/<run_name>/` (metrics.csv, config_snapshot.json). |
| `--log-dir` | = checkpoint-dir | Каталог логов. |
| `--config` | None | Путь к JSON конфигу эксперимента (lr, entropy_coef, value_loss_coef, …); CLI переопределяет. |
| `--reward-config` | None | Путь к JSON конфигу наград (RewardConfig); иначе default_v1_config(). |
| `--entropy-coef` | 0.01 | Коэффициент энтропии в loss PPO. |
| `--value-loss-coef` | 0.5 | Коэффициент value loss в loss PPO. |
| `--entropy-floor` | 0.3 | Ниже — предупреждение о коллапсе политики (guardrail). |
| `--stuck-updates` | 20 | После скольких обновлений без роста unique_rooms — предупреждение (guardrail). |
| `--recurrent` | False | LSTM после encoder (память для лабиринта). См. docs/PPO_MODEL.md §3. |
| `--lstm-hidden-size` | 256 | Размер скрытого состояния LSTM при recurrent. |
| `--stuck-nudge-steps` | 20 | При stuck N шагов RIGHT/LEFT/UP/DOWN по очереди для выхода. |
| `--sequence-length` | 0 | Зарезервировано для диагностики (пока не используется). |
| `--dry-run-seconds` | 0 | Прогнать N секунд и вывести steps/sec, updates/hour; 0 = выкл. См. docs/TRAINING.md §7. |
| `--no-reset-handshake` | False | При num_envs>1 отключить reset handshake (для отладки, если кнопки в env 1 не работают). |

Подробнее: **`docs/TRAINING.md`** (эксперименты, метрики, отладка); **`docs/PPO_RECURRENT_AUDIT.md`**, **`docs/PPO_RECURRENT_BENCHMARK.md`** (LSTM).

**Куда пишется:**

- **Чекпоинты:** `<checkpoint-dir>/last.pt` (после каждого обновления), `<checkpoint-dir>/epoch_10.pt`, `epoch_20.pt`, … (каждые 10 обновлений).
- **Логи эксперимента** (при `--run-name`): `<log-dir>/<run_name>/metrics.csv`, `config_snapshot.json`.
- **Workdir openMSX:** `checkpoints/ppo/run/` (step_frame.png, commands.tcl, логи эмулятора) — не «результаты обучения», а артефакты текущего запуска env.

**Как понять, сколько было обучений (обновлений):**  
В рамках одного запуска — по `--epochs` и по наличию файлов `epoch_10.pt`, `epoch_20.pt`, … в `checkpoint-dir`. Отдельного счётчика запусков или лог-файла с историей нет; при повторном запуске с тем же `--checkpoint-dir` файлы перезаписываются/дополняются.

**Команды:**  
`python train_ppo.py`  
`python train_ppo.py --bc-checkpoint checkpoints/bc/best.pt --epochs 100`  
`python train_ppo.py --checkpoint-dir checkpoints/ppo_exp2 --epochs 200`  
`python train_ppo.py --num-envs 2 --epochs 2 --rollout-steps 32` (короткий прогон с двумя env)

---

## 7. Тест BC-политики (test_policy.py)

**Назначение:** запуск обученной BC-модели в окружении (openMSX поднимается автоматически).

**Флаги:** `--checkpoint` (по умолчанию checkpoints/bc/best.pt), `--max-steps`, `--device`, `--workdir`, `--stop-on-death`, `--smooth`, `--sticky`, `--max-idle-steps`, `--transition-assist`, `--stair-assist-steps`, `--capture-backend`.

**Куда пишется:** логи/скриншоты openMSX — в `workdir` (по умолчанию `checkpoints/bc/run/`).

**Команда:**  
`python test_policy.py --checkpoint checkpoints/bc/best.pt --max-steps 2000 --smooth 3 --sticky`

---

## 8. Тест PPO-политики (test_ppo.py)

**Назначение:** запуск обученной PPO-модели в окружении.

**Флаги:** `--checkpoint` (по умолчанию checkpoints/ppo/last.pt), `--max-steps`, `--device`, `--workdir`, `--stop-on-death`, `--deterministic`, `--smooth`, `--sticky`, `--max-idle-steps`, `--transition-assist`, `--stair-assist-steps`, `--capture-backend`.

**Куда пишется:** логи/скриншоты openMSX — в `workdir` (по умолчанию `checkpoints/ppo/run/`).

**Команда:**  
`python test_ppo.py --checkpoint checkpoints/ppo/last.pt --deterministic`

---

## 8.1. Ночной супервизор (train_supervisor.py)

**Назначение:** непрерывный ночной прогон PPO с автоперезапуском при падении, откатом при NaN и watchdog по таймауту. Читает **`configs/night_training.json`**.

**Поведение:** запускает `train_ppo.py` с `--num-envs 2`, `--run-name auto_night`, `--resume`; при exit code != 0 перезапускает после задержки; при обнаружении NaN в последней строке metrics.csv откатывает `last.pt` из rolling backup и перезапускает; при отсутствии обновления метрик дольше `watchdog_timeout_minutes` минут завершает процесс и перезапускает. При каждом перезапуске пишет лог в `<run_dir>/supervisor.log`, сохраняет копию чекпоинта как `crash_restart_N.pt`.

**Конфиг (configs/night_training.json):** `num_envs`, `max_updates`, `checkpoint_every`, `restart_limit`, `watchdog_timeout_minutes`, `entropy_floor`, `auto_entropy_boost`, `run_name`, `checkpoint_dir`, `log_dir`, `restart_delay_seconds`.

**Команда:**  
`python train_supervisor.py`

**Куда пишется:** логи супервизора — `<checkpoint_dir>/<run_name>/supervisor.log`; метрики и чекпоинты — как у train_ppo; при падении — `checkpoints/ppo/crash_restart_N.pt`.

---

## 9. Бенчмарк env (benchmark_env.py)

**Назначение:** замер производительности env (шагов в секунду/минуту) и времени захвата/препроцессинга (avg, p95) для сравнения бэкендов.

**Флаги:** `--backend` (png | single | window), `--steps` (1000), `--action-repeat`, `--decision-fps`, `--workdir`, `--window-crop` (x y w h), `--window-title`, `--capture-lag-ms`.

**Команды:**  
`python benchmark_env.py --backend png`  
`python benchmark_env.py --backend single --steps 500`  
`python benchmark_env.py --backend window --steps 500` (при необходимости `--window-crop` после калибровки)

---

## 10. Система наград (reward)

**Модуль:** `msx_env/reward/`

**Назначение:** модульная политика наград (step, death, pickup, novelty, ping-pong, stuck). Включается при `EnvConfig.reward_config = default_v1_config()`; иначе используется legacy (HUD + death в env, step_penalty в train_ppo).

**Конфиг:** `RewardConfig` в `msx_env/reward/config.py`; дефолт v1 — `default_v1_config()`.

Подробно: **`docs/REWARD.md`** (компоненты, формулы, пороги, отладка, таблица дефолтов).

---

## 11. Датасет демо (dataset)

**Модуль:** `msx_env/dataset.py`

**Назначение:** сохранение и загрузка одного прогона; метаданные в manifest; валидация (мин. шагов, распределение действий, отличие кадров).

**Формат:** в каталоге рана — `data.npz` (obs, actions, rewards, next_obs, dones, timestamps), `manifest.json` (schema_version, metadata, frame_hashes). Формат датасета не меняется при смене reward/capture, если не меняется форма obs.

---

## 12. Мульти-инстанс (make_env, num_envs)

**Модуль:** `msx_env/make_env.py` — `make_env(rank, base_cfg)` возвращает фабрику (callable), создающую `VampireKillerEnv` с `workdir=tmp_root/rank` и `instance_id=rank`. У каждого инстанса свой каталог и свой процесс openMSX (файловый протокол), коллизий нет.

**train_ppo:** при `--num-envs 2` (или больше) создаётся несколько env через `make_env(i, base_cfg)()` с `tmp_root=--tmp-root` (по умолчанию `runs/tmp`). Rollout собирается со всех env по очереди; GAE считается по траекториям каждого env отдельно.

**Ограничение:** при `capture_backend=window` несколько env означают несколько окон openMSX и отдельный crop на окно (продвинутый режим); надёжно многопоточность с бэкендом **png/single**.

**Smoke:** `python tests/test_multi_env_smoke.py` (проверка уникальных workdir без ROM); `python scripts/run_2env_smoke.py` (2 env по 200 шагов с ROM).

---

## 13. Тесты

**tests/test_reward_smoke.py:** 500 шагов с фейковыми obs, проверка отсутствия NaN в награде и разбивке. Запуск: `python tests/test_reward_smoke.py`.

**tests/test_multi_env_smoke.py:** проверка, что make_env выдаёт конфиги с разными workdir и instance_id.

---

## 14. Сводка: куда смотреть результаты и как запускать

| Действие | Куда смотреть результат | Команда запуска |
|----------|-------------------------|-----------------|
| Запись демо | `demos/runs/<run_id>/data.npz`, manifest, логи | `python demos/record_demo.py --run-id <id>` |
| Валидация рана | Консоль PASS/FAIL, опционально preview.mp4 | `python demos/validate_demo.py <run_id> --preview` |
| Обучение BC | `checkpoints/bc/best.pt`, `last.pt` | `python train_bc.py [--runs ...] [--deep]` |
| Обучение PPO | `checkpoints/ppo/last.pt`, `epoch_10.pt`, … | `python train_ppo.py [--bc-checkpoint ...] [--epochs 100]` |
| Тест BC | Консоль (шаги, action, life); openMSX в workdir | `python test_policy.py --checkpoint checkpoints/bc/best.pt` |
| Тест PPO | То же | `python test_ppo.py --checkpoint checkpoints/ppo/last.pt` |
| Бенчмарк env | Консоль (steps/sec, avg/p95 capture, prep, step) | `python benchmark_env.py --backend png \| single \| window` |
| Мульти-env PPO | Без коллизий (уникальные workdir) | `python train_ppo.py --num-envs 2 --tmp-root runs/tmp` |

---

*Документ обновляется при добавлении новых модулей, флагов и выходов.*
