## Контекст и решения

- **`docs/SESSION.md`** — **точка входа для нового чата**: саммари проекта, недавние правки, ключевые пути и команды. При необходимости смотреть `docs/MODULES_AND_FLAGS.md` и `docs/CONTEXT.md`.
- **`docs/CONTEXT.md`** — контекст проекта, архитектура, ключевые решения (frame stacking, полоска жизни, лаги).
- **`docs/MODULES_AND_FLAGS.md`** — справочник по модулям, флагам и выходам: скрипты, чекпоинты, логи, запуск обучения и тестов.
- **`docs/TRAINING.md`** — PPO: аудит обучения, стабильность, система экспериментов (--run-name, --config, --reward-config), guardrails, отладка.
- **`docs/REWARD.md`** — система наград (компоненты, масштабы, отладка, v1).
- **`docs/CAPTURE.md`** — бэкенды захвата кадра (png, single, window), калибровка окна, бенчмарк.

---

## Демозапись для Vampire Killer (openMSX + RL)

### Быстрый старт: короткий прогон (Phase Demo‑0)

1. Установить зависимости:

```bash
pip install -r requirements.txt
```

2. Положить ROM игры рядом с кодом под именем `VAMPIRE.ROM`.

3. Записать короткую демонстрацию (до пол‑уровня):

```bash
python demos/record_demo.py --run-id test01 --max-minutes 3 --max-steps 2000 --preview
```

Скрипт:
- поднимает openMSX через `VampireKillerEnv`,
- открывает маленькое окно pygame и читает нажатия:
  - стрелки — движение,
  - Z / Space — удар,
  - X или стрелка вверх — прыжок,
  - Esc — завершить запись,
- пишет датасет в `demos/runs/test01/`,
- сразу гоняет валидацию и (при `--preview`) создаёт `preview.mp4`.

Всё, что касается конкретного рана (`test01`), складывается в отдельную папку:

- `demos/runs/test01/data.npz` — массивы obs/actions/…;
- `demos/runs/test01/manifest.json` — метаданные и хэши;
- `demos/runs/test01/openmsx_stdout.log` и `openmsx_stderr.log` — логи эмулятора;
- `demos/runs/test01/commands.tcl`, `reply.txt`, `bootstrap.tcl`, `bootstrap_status.txt` —
  служебные файлы протокола с openMSX;
- `demos/runs/test01/step_frame.png` — последний захваченный кадр (для быстрой ручной проверки).

4. Отдельно можно проверить рун:

```bash
python demos/validate_demo.py test01 --preview
python demos/replay_demo.py test01 --mode frames --fps 10
```

### Half‑level test checklist

- Есть каталог `demos/runs/<run_id>/`.
- Файл `data.npz`:
  - размер > 1 КБ,
  - число шагов > 500.
- `manifest.json` присутствует и содержит:
  - `schema_version`,
  - `metadata` (rom, action_set_version, obs_shape, timestamps).
- Валидация (`validate_demo_run`) прошла:
  - распределение действий не вырождено (NOOP не > 95%),
  - хэши кадров отличаются (кадр реально меняется),
  - при `--preview` создан `preview.mp4`.
- `replay_demo.py` показывает разумное воспроизведение.

Только после этого имеет смысл писать более длинные демонстрации (например, 30‑минутные).

### Отключение записи при обучении

- Вся логика записи изолирована в `demos/` и `msx_env/dataset.py`.
- Для тренировочного кода:
  - используем только `VampireKillerEnv` из `msx_env.env`,
  - **не** вызываем `save_demo_run` / `validate_demo_run`,
  - режимы `--mode=train/--mode=eval` можно реализовать отдельно, просто не создавая датасет.

### Файлы / модули

- `msx_env/env.py` — `VampireKillerEnv` и дискретный action‑space.
- `msx_env/human_controller.py` — слой, читающий клавиши из окна pygame и возвращающий action id.
- `msx_env/dataset.py` — схема датасета (`DemoMetadata`), сохранение/загрузка, базовая валидация.
- `msx_env/replay_utils.py` — утилита для сборки `preview.mp4` из массива obs.
- `demos/record_demo.py` — основной скрипт записи демонстраций.
- `demos/validate_demo.py` — автономная валидация существующего рана.
- `demos/replay_demo.py` — проигрывание записанного рана (по кадрам или через env).
- `msx_env/life_bar.py` — оценка полоски жизни (PLAYER) по кадру для детекции смерти.

### Отзывчивость (лаги)

- При записи: `--fps 15` (или 20) и `--poll-ms 15` — чаще опрос клавиш и быстрее реакция openMSX.
- В env уменьшены `hold_ms` для действий (100 ms для движения/удара).

---

## Обучение (Behavior Cloning)

После записи одного или нескольких прогонов можно обучить политику имитации (BC).

### 1. Обучение

Используются **все** прогоны в `demos/runs/` (у которых есть `data.npz`), либо только указанные. По умолчанию включён **frame stacking (4 кадра)** — модель видит последние 4 кадра, что уменьшает зацикливание на переходах между экранами.

```bash
# Все прогоны из demos/runs/, 4 кадра в стопке
python train_bc.py --epochs 40 --batch-size 64

# Усиленная сеть + приоритет движения (идти по лабиринту, подниматься/спускаться по лестницам, подбирать призы) и редкие действия (прыжки, удар)
python train_bc.py --epochs 60 --deep --move-weight 1.6 --rare-weight 1.5 --oversample 3

# Конкретные прогоны
python train_bc.py --runs run_full_01 run_full_02 --epochs 40 --frame-stack 4 --checkpoint-dir checkpoints/bc
```

Чекпоинты сохраняются в `checkpoints/bc/`:
- `best.pt`, `last.pt` — dict с `state_dict`, `frame_stack` (4) и `arch` (`"default"` или `"deep"`). При загрузке `test_policy.py` и `load_bc_checkpoint` автоматически выбирают BCNet или BCNetDeep по полю `arch`.

### 2. Тест политики в игре

Запуск обученной модели в окружении (openMSX поднимается автоматически). Модель с frame_stack=4 получает стопку из 4 последних кадров. Опции: завершать при смерти (`--stop-on-death`), сглаживание действий (`--smooth N` — majority vote по последним N шагам), «липкое» действие (`--sticky` — при NOOP один раз повторить последнее ненулевое действие, меньше замираний).

```bash
python test_policy.py --checkpoint checkpoints/bc/best.pt --max-steps 2000
python test_policy.py --checkpoint checkpoints/bc/best.pt --stop-on-death
python test_policy.py --checkpoint checkpoints/bc/best.pt --smooth 5 --sticky
# Рекомендуется: анти-залипание, переходы, лестницы (при right-left цикле)
python test_policy.py --checkpoint checkpoints/bc/best.pt --smooth 3 --sticky --max-idle-steps 40 --transition-assist --stair-assist-steps 50
```

Логи и скриншоты openMSX при тесте пишутся в `checkpoints/bc/run/` (или в `--workdir`, если указан). Оценка полоски жизни (PLAYER) — в `msx_env.life_bar`; при `--stop-on-death` при резком падении жизни прогон завершается. **Reward за подбор** (оружие, ключи, предметы) считается по HUD (`msx_env.hud_parser`) и отдаётся в `step()` — пригоден для будущего RL; в BC не используется.

### Откуда подтягивать данные при обучении

- **Демонстрации**: каталоги `demos/runs/<run_id>/`, в каждом — `data.npz` и `manifest.json`.
- **Загрузка**: `from msx_env.dataset import load_demo_run`.
- **Загрузка в коде**: `from msx_env.dataset import load_demo_run`; затем `obs, actions, ... = load_demo_run(Path("demos/runs/run_full_01"))`.
- **Чекпоинты BC**: `checkpoints/bc/best.pt`; **PPO**: `checkpoints/ppo/last.pt`.

---

## PPO (Reinforcement Learning)

PPO обучает политику по reward (подбор предметов, смерть). BC остаётся доступен через `test_policy.py`.

### Обучение PPO (инициализация из BC)

```bash
python train_ppo.py --bc-checkpoint checkpoints/bc/best.pt --epochs 100 --arch deep
```

### Тест PPO

```bash
python test_ppo.py --checkpoint checkpoints/ppo/last.pt --deterministic --smooth 3 --sticky
```

#   V a m p i r e - k i l l e r - M S X - R L 
 
 