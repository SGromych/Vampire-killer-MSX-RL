# Configuration system: how to run and how to override

Single source of truth for all config and flags; every run uses the same resolved config. Paths are absolute under the run directory; missing user-specified files cause immediate failure.

## How to run

1. **Primary (recommended): config file**
   - Create or use a run directory; put a full config there or pass `--config <path>` to the trainer.
   - Supervisor, when restarting, passes `--config run_dir/config_snapshot.json --resume` so the same config is used for all restarts.

2. **Legacy: CLI only**
   - Run `train_ppo.py` with flags as before. The central module still resolves paths, validates, and writes `config_snapshot.json` and `resolved_paths.json` into the run directory.

3. **Supervisor**
   - Reads `configs/night_training.json`, creates a run dir (e.g. `runs/<timestamp>_<git>_<run_name>/` when `use_runs_dir: true`).
   - First start: runs trainer with full flags + `--run-dir`.
   - Restarts: runs trainer with `--config run_dir/config_snapshot.json --resume` so no flag drift.

## How to override

- **Order of precedence:** defaults → config file (if `--config` given) → CLI.
- **Reward config:** `--reward-config <path>`. If the path is given and the file is missing, the process exits with an error (no silent default).
- **Config file:** `--config <path>`. If the path is given and missing, exit with an error.
- All paths written into the run directory are absolute. Artifacts: `train.log`, `supervisor.log`, `metrics.csv`, `config_snapshot.json`, `resolved_paths.json`, `metrics_schema.json`, `checkpoints/ppo/`.

## Key modules

- **project_config.py:** `load_config(argv)` (single parser), `build_resolved_config_from_args(args, root)`, `RunLayout`, `validate_config()`, schemas: `RunConfig`, `PPOConfig`, `EnvConfigSchema`, `CaptureConfig`, `LoggingConfig`, `ResolvedConfig`.
- **train_ppo.py:** If `--config` in argv → `load_config()`; else `parse_args()` + `build_resolved_config_from_args()`. Uses `config.layout` for paths when central config is used.
- **train_supervisor.py:** `build_argv(..., run_dir_override)`: when `run_dir/config_snapshot.json` exists, returns `[..., "--config", snapshot, "--resume"]`.

## Inventory and graph

- `python tools/config_inventory.py` — writes `docs/CONFIG_INVENTORY.md` (grouped flags/fields/env vars) and `docs/CONFIG_GRAPH.md` (which module consumes which).

## Checkpoints and resume

- Checkpoints include an `arch_signature` (arch, frame_stack, recurrent, lstm_hidden_size). On resume, the current expected signature is compared; on mismatch a clear `ValueError` is raised with a diff.

---

## Как убедиться, что изменения кода попали в ночной прогон

- **Код (логика, баги, фичи)** загружается при каждом **запуске процесса** Python. Первый запуск ночи и каждый рестарт супервизора — это новый процесс, поэтому всегда выполняется актуальный код из репозитория (из той директории, из которой запущен супервизор).
- **Конфиг (lr, epochs, reward_config, num_envs и т.д.)** при рестартах берётся из `config_snapshot.json` (записанного при первом запуске этой папки run). Чтобы новые значения из `night_training.json` попали в прогон, нужна **новая папка run** (например новый запуск супервизора при `use_runs_dir: true` создаёт новую `runs/<дата>_<время>_<git>_<name>/`).

**Проверка по логам:**

1. В **train.log** в начале каждого запуска (в т.ч. после рестарта) пишется строка вида  
   `code_version=<git short hash> (check this in train.log to confirm which commit ran)`.  
   Сравните этот хеш с `git rev-parse --short HEAD` в репозитории — они должны совпадать для того коммита, с которого вы запускали обучение.
2. В **config_snapshot.json** и **resolved_paths.json** в run dir поле **code_version** — это коммит, при котором был записан снимок (первый запуск этой папки).
3. Имя папки run при `use_runs_dir: true` уже содержит короткий git-хеш, например `runs/20250305_230000_a1b2c3d4_auto_night/` — по нему видно, с какого коммита создана папка.

Итого: если вы обновили код, закоммитили и заново запустили супервизор (или произошёл рестарт), в train.log при следующем старте процесса будет новый `code_version`. Конфиг (гиперпараметры) при рестартах в той же папке run не меняется — он зафиксирован в config_snapshot.json.

---

## Чеклист: что уже применяется в прогонах

| Наработка | Где включено / как применяется |
|-----------|---------------------------------|
| **Reward v3** | `configs/night_training.json` → `reward_config: "configs/reward_v3.json"`; супервизор передаёт `--reward-config` при первом запуске. |
| **Единый конфиг (project_config)** | Тренер при каждом запуске вызывает `load_config()` или `build_resolved_config_from_args()`; пути и снимки пишутся в run dir. |
| **Снимок при рестарте** | Супервизор при наличии `config_snapshot.json` передаёт `--config run_dir/config_snapshot.json --resume` — один и тот же конфиг на все рестарты. |
| **Первый запуск без --resume** | Супервизор не передаёт `--resume`, если снимка ещё нет (новая папка run), чтобы не падать по «checkpoint not found». |
| **Строгая загрузка reward** | Если указан `--reward-config` и файла нет — процесс завершается с ошибкой (без тихого дефолта). |
| **Архитектурная подпись при resume** | В чекпоинтах сохраняется `arch_signature`; при несовпадении с текущей конфигурацией — явная ошибка. |
| **code_version в логах** | В начале каждого запуска в `train.log` и в `config_snapshot.json` / `resolved_paths.json` записывается git short hash. |
