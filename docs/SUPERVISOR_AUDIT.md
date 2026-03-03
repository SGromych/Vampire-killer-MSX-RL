# Аудит: чекпоинты и resume в train_ppo.py

Проведён перед внедрением ночного супервизора (train_supervisor.py).

## Итог после доработок

- **last.pt** сохраняется после каждого update с полями: `state_dict`, `frame_stack`, `arch`, `update`, `optimizer_state`, `rng_state` (torch, numpy, python).
- **--resume**: загрузка last.pt, восстановление модели, оптимизатора (если есть), номера update, RNG; цикл с `range(start_update, args.epochs)`.
- **--checkpoint-every N**: каждые N обновлений создаётся rolling backup (backup_0.pt … backup_4.pt, хранятся последние 5).
- **Супервизор** (`train_supervisor.py`) читает `configs/night_training.json`, запускает train_ppo с `--num-envs 2 --run-name auto_night --resume`, перезапускает при падении, при NaN откатывает last.pt из backup, watchdog по таймауту. В CSV дописываются колонки restart_count, uptime_seconds, crash_flag (через env и расчёт uptime в train_ppo).

---

## Текущее поведение (до доработок)

### Сохранение чекпоинтов

- **last.pt** — сохраняется после **каждого** обновления (каждая итерация цикла `for update in range(args.epochs)`).
- **best.pt** — **не сохраняется** в train_ppo.py (есть только в BC: train_bc.py).
- **epoch_N.pt** — сохраняется каждые 10 обновлений (`(update + 1) % 10 == 0`), только модель (`state_dict`, `frame_stack`, `arch`).

### Содержимое чекпоинта (last.pt)

- `state_dict` — веса модели (ActorCritic).
- `frame_stack` — размер стопки кадров.
- `arch` — строка архитектуры ("default" | "deep").
- **Не сохраняются:** номер обновления (update), состояние оптимизатора, состояние RNG (torch/numpy/random). Нет rolling backups.

### Resume

- Логики **--resume** нет: при повторном запуске обучение всегда начинается с update=0. Загрузить last.pt для продолжения нельзя без ручного кода.

### Итог для супервизора

- Добавить в train_ppo: **--resume** (загрузка last.pt, восстановление update, optimizer, RNG).
- Расширить чекпоинт: **update**, **optimizer state**, **rng state** (torch, numpy, random).
- Добавить **периодическое сохранение** (например каждые X обновлений) и **rolling backups** (хранить последние 5 копий) для отката при NaN.
- best.pt при необходимости можно ввести отдельно (например по лучшему ep_return_mean); для ночного режима достаточно last.pt + backups.
