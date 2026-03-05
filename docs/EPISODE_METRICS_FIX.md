# Episode Metrics Fix (PR episode-metrics-fix)

## Цель

Корректные эпизодные метрики (per-env) для принятия решений по обучению. Старые метрики **не изменены** — добавлены новые с суффиксом `_ep`.

## Изменённые файлы

| Файл | Изменения |
|------|-----------|
| `msx_env/reward/episode_metrics.py` | **Новый.** EpisodeRoomTracker, debounce room_hash, update_episode_room_metrics() |
| `msx_env/reward/policy.py` | Интеграция EpisodeRoomTracker, вызов update_episode_room_metrics() |
| `train_ppo.py` | Сбор новых метрик при done, новые колонки в metrics.csv |
| `scripts/test_episode_metrics.py` | **Новый.** Acceptance test |
| `docs/EPISODE_METRICS_FIX.md` | **Новый.** Документация |

## Новые колонки в metrics.csv

```
unique_rooms_ep_env0, unique_rooms_ep_env1, unique_rooms_ep_mean, unique_rooms_ep_min, unique_rooms_ep_max,
room_transitions_ep_env0, room_transitions_ep_env1, room_transitions_ep_mean,
room_dwell_steps_ep_mean,
stage00_exit_rate_ep, stage00_time_to_exit_steps_ep_env0, stage00_time_to_exit_steps_ep_env1,
stage00_time_to_exit_steps_ep_mean, stage00_time_to_exit_steps_ep_min, stage00_time_to_exit_steps_ep_max,
backtrack_rate_ep_env0, backtrack_rate_ep_env1, backtrack_rate_ep_mean, backtrack_rate_ep_min, backtrack_rate_ep_max
```

## Семантика

- **unique_rooms_ep** — число уникальных комнат за эпизод (debounced room_hash).
- **room_transitions_ep** — число переходов между комнатами за эпизод.
- **room_dwell_steps_ep_mean** — среднее время (шагов) в комнате до перехода.
- **stage00_exit_rate_ep** — доля эпизодов с переходом Stage00→01 (событие 0→1, не привязано к done).
- **stage00_time_to_exit_steps_ep** — шаги до перехода 0→1.
- **backtrack_rate_ep** — доля backtrack-переходов (A→B→A) среди всех переходов.

> **Источник правды по метрикам.**  
> Сводный список всех логируемых метрик и их использование в тренировочном цикле описан в
> `docs/TRAINING.md` (раздел «Логируемые метрики»). Настоящий документ фиксирует только
> **новые эпизодные колонки** с суффиксом `_ep` и их формальные определения.

## Debounce (D)

- `room_hash`: K=4 кадров (конфиг `episode_room_debounce_k`).
- `stage`: N=4 кадра (конфиг `episode_stage_stable_frames`).

## Как убедиться, что stage00_exit_rate_ep > 0

1. Запустить обучение с политикой, которая может дойти до Stage01.
2. Или acceptance test с долгим эпизодом и scripted RIGHT:
   ```bash
   python scripts/test_episode_metrics.py
   ```
3. При успешном выходе из Stage00 в Stage01: `stage00_exit_recorded_ep=1` и `stage00_exit_steps_ep` > 0.

## Acceptance test

```bash
python scripts/test_episode_metrics.py
```

Выводит: `unique_rooms_ep`, `room_transitions_ep`, `room_dwell_steps_ep_mean`, `stage00_exit_recorded_ep`, `stage00_exit_steps_ep`, `backtrack_rate_ep` для 1–2 эпизодов со scripted RIGHT.
