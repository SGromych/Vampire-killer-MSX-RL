# Саммари проекта RL_msx (Vampire Killer PPO) — для копирования в другой чат

Скопируйте блок ниже в новый чат как контекст.

---

## Проект RL_msx: PPO для Vampire Killer (MSX2)

**Цель:** обучить PPO-агента играть в Vampire Killer через эмулятор openMSX (file-based управление + захват экрана).

**Стек:** Python, PyTorch, Gym-подобное окружение `msx_env.env.VampireKillerEnv`, модульная система наград `msx_env/reward/`, BC → PPO, опционально recurrent (LSTM), multi-env.

**Ключевые точки входа:**
- Обучение PPO: `train_ppo.py` (конфиг `configs/night_training.json`, флаги: `--resume`, `--num-envs`, `--run-name`, `--config`, `--reward-config`).
- Ночной запуск: `train_supervisor.py` читает `configs/night_training.json`, запускает `train_ppo.py` с `--resume`, рестартует при падении, при `use_runs_dir: true` создаёт `runs/<YYYYMMDD>_<HHMMSS>_<git>_<run_name>/`.
- Тест политики: `test_ppo.py --checkpoint checkpoints/ppo/last.pt`; диагностика: `test_ppo.py --diagnose-policy`.
- Документация: `docs/PROJECT_OVERVIEW.md` (обзор), `docs/TRAINING.md` (обучение, метрики, guardrails), `docs/SESSION.md` (пути, как запускать, где искать run), `docs/REWARD.md`, `docs/MODULES_AND_FLAGS.md`.

**Где лежат результаты ночного обучения:**
- При `use_runs_dir: true`: run dir = `runs/<YYYYMMDD>_<HHMMSS>_<git>_<run_name>/` (например `runs/20260304_181828_58548f2_auto_night/`).
- «Сегодняшний» прогон — папка с самой свежей датой изменения файлов (запись может идти до утра следующего дня).
- В run dir: метрики в `metrics.csv` (или `metrics1.csv`), лог обучения `train.log`, конфиг на старт `config_snapshot.json`, лог супервизора `supervisor.log` (или `supervisor1.log`).
- Чекпоинты: при use_runs_dir — `runs/.../checkpoints/ppo/last.pt` и backup_*.pt; иначе `checkpoints/ppo/last.pt`.

**Ключевые метрики в CSV:** `update`, `steps_per_sec`, `reward_mean`, `ep_return_mean`, `unique_rooms_mean`/`unique_rooms_max`, `entropy`, `policy_loss`, `value_loss`, `deaths`, `stuck_events`. Полный список — `docs/TRAINING.md` и `docs/EPISODE_METRICS_FIX.md`.

**Конфиг ночного запуска:** `configs/night_training.json` (`num_envs`, `max_updates`, `run_name`, `checkpoint_dir`, `use_runs_dir`, `novelty_reward`, `recurrent` и т.д.). Префлайт перед ночью: `python train_supervisor.py --preflight`.

**Состояние кода:** документация актуализирована (пути run dir, метрики, варианты имён metrics.csv/metrics1.csv). Linter по основным файлам (train_ppo.py, train_supervisor.py, msx_env) — без ошибок.

---

*Конец саммари для копирования.*
