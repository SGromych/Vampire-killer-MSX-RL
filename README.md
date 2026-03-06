# RL_msx — PPO для Vampire Killer (MSX2)

Обучение PPO-агента для игры Vampire Killer через эмулятор openMSX: управление по файлам (commands.tcl / reply.txt) и захват экрана. Поддержка multi-env (несколько инстансов openMSX), модульная система наград, BC → PPO, опционально recurrent (LSTM).

**Документация:** обзор и архитектура — `docs/PROJECT_OVERVIEW.md`, контекст и решения — `docs/CONTEXT.md`, сессия и пути — `docs/SESSION.md`. Модули и флаги — `docs/MODULES_AND_FLAGS.md`, обучение и метрики — `docs/TRAINING.md`, награды — `docs/REWARD.md`, захват кадра — `docs/CAPTURE.md`.

## Быстрый старт

- **Обучение PPO:** `python train_ppo.py` (по умолчанию: захват **dxcam**, без скриншотов на диск). С двумя env: `python train_ppo.py --num-envs 2`. Сухой прогон: `python train_ppo.py --dry-run-seconds 30`.
- **Ночной запуск:** `python train_supervisor.py` (читает `configs/night_training.json`, рестарт при падении).
- **Тест политики:** `python test_ppo.py --checkpoint checkpoints/ppo/last.pt`.
- **Конфиг:** один источник правды — `project_config.load_config(argv)` (defaults → `--config` → CLI). Артефакты рана: `config_snapshot.json`, `resolved_paths.json`, `train.log`, `metrics.csv` в каталоге рана. Подробнее: `docs/CONFIG_SYSTEM.md`, инвентарь опций: `tools/config_inventory.py`.

## Захват кадра

| Бэкенд   | Описание |
|----------|----------|
| **dxcam** | По умолчанию в train_ppo. Захват окна openMSX в память (Windows, `pip install dxcam`). Без записи PNG на диск. |
| **png** / **single** | openMSX пишет step_frame.png, Python читает с диска. Надёжно при отладке или без dxcam. |
| **window** | Захват по области экрана (dxcam/mss), с fallback на png. |

При **num_envs > 1** у каждого env свой workdir (`runs/tmp/0`, `runs/tmp/1`); при dxcam окно определяется по PID процесса openMSX. Подробнее: `docs/CAPTURE.md`, `docs/MODULES_AND_FLAGS.md`.

## Что можно удалить

См. **`docs/CLEANUP.md`**: артефакты в корне (reply.txt, bootstrap_status.txt), каталог diagnostics/, устаревшие аудиты в docs/, шаблоны.
