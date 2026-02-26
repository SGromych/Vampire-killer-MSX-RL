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

---

## Обучение (Behavior Cloning)

После записи одного или нескольких прогонов можно обучить политику имитации (BC).

### 1. Обучение

Используются **все** прогоны в `demos/runs/` (у которых есть `data.npz`), либо только указанные:

```bash
# Все прогоны из demos/runs/
python train_bc.py --epochs 20 --batch-size 64

# Конкретные прогоны
python train_bc.py --runs run_full_01 run_full_02 --epochs 30 --checkpoint-dir checkpoints/bc
```

Чекпоинты сохраняются в `checkpoints/bc/`:
- `best.pt` — по минимальному loss на обучении;
- `last.pt` — после последней эпохи.

### 2. Тест политики в игре

Запуск обученной модели в окружении (openMSX поднимается автоматически):

```bash
python test_policy.py --checkpoint checkpoints/bc/best.pt --max-steps 2000
```

Логи и скриншоты openMSX при тесте пишутся в `checkpoints/bc/run/` (или в `--workdir`, если указан).

### Откуда подтягивать данные при обучении

- **Демонстрации**: каталоги `demos/runs/<run_id>/`, в каждом — `data.npz` и `manifest.json`.
- **Загрузка в коде**: `from msx_env.dataset import load_demo_run`; затем `obs, actions, ... = load_demo_run(Path("demos/runs/run_full_01"))`.
- **Чекпоинты BC**: `checkpoints/bc/best.pt` (или `last.pt`) — `state_dict` модели `BCNet` из `msx_env.bc_model`.

