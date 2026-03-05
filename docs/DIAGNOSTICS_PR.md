# Диагностический PR: unique_rooms, Stage00, backtrack, after-key

> **NOTE:** структура `metrics.csv`, описанная в этом документе, **устарела**.
> Актуальный список колонок и эпизодных метрик см. в `docs/TRAINING.md`
> (раздел про логируемые метрики) и `docs/EPISODE_METRICS_FIX.md`.

## 1. Изменённые файлы

| Файл | Изменения |
|------|-----------|
| `msx_env/reward/diagnostics.py` | **Новый.** EpisodeDiagnostics, update_diagnostics (room_transition, backtrack, stage00, after-key) |
| `msx_env/reward/policy.py` | Интеграция diagnostics, parse_stage для Stage00, ключ из HUD |
| `msx_env/env.py` | info["env_id"], debug_room_change, вывод `[ENV i] room change prev->new` |
| `train_ppo.py` | episode_stats расширены, новые метрики, --debug-room-change, metrics.csv колонки |

---

## 2. Новый header metrics.csv

```
update,steps_per_sec,sample_throughput,policy_loss,value_loss,entropy,approx_kl,explained_var,
reward_mean,ep_return_mean,ep_steps_mean,unique_rooms_mean,unique_rooms_max,deaths,stuck_events,
unique_rooms_env0,unique_rooms_env1,room_transition_env0,room_transition_env1,
stage00_exit_rate,stage00_exit_steps_mean,stage00_room_trans_mean,backtrack_rate_mean,
steps_after_key_total_mean,steps_after_key_until_exit_mean,steps_after_key_until_death_mean,
reward_step,reward_pickup,reward_death,reward_novelty,reward_pingpong,reward_stuck,reward_key,reward_door,
reward_stage_step,reward_stage_advance[,restart_count,uptime_seconds,crash_flag]
```

---

## 3. Как проверить

### A) unique_rooms > 0

1. Запустить короткий тест:  
   `python train_ppo.py --num-envs 2 --epochs 20 --no-quiet`
2. В summary смотреть:  
   `unique_rooms=X.XX(max=Y)` — если агент заходит в новые комнаты, X и Y должны расти.
3. В `metrics.csv`:  
   `unique_rooms_env0`, `unique_rooms_env1` — отдельно по каждому env.
4. Если unique_rooms всё ещё 0 при визуальной смене комнат:  
   `python train_ppo.py --num-envs 2 --epochs 5 --debug-room-change`  
   — вывод `[ENV i] room change prev->new` при смене room_hash. Если лог пустой — room_hash не меняется (настройка хэша/HUD).

### B) Stage00 exit метрика

1. Stage detector использует HUD crop (`parse_stage`), не RAM.
2. При выходе STAGE 00 → STAGE 01:  
   `stage00_exit_rate` > 0, `stage00_exit_steps_mean` > 0.
3. В `metrics.csv`:  
   `stage00_exit_rate` — доля эпизодов с успешным выходом,  
   `stage00_exit_steps_mean` — среднее число шагов до выхода.

### C) backtrack_rate при хождении A↔B

1. При движении по циклу A→B→A→B:  
   `backtrack_rate_mean` > 0.
2. В `metrics.csv`:  
   `backtrack_rate_mean` = backtracks / room_transitions.
3. Паттерн backtrack: `h[i-2]==h[i]` и `h[i-1]!=h[i]` (возврат в комнату через одну).

### D) After-key (если ключ взят)

- `steps_after_key_total_mean` — среднее число шагов после получения ключа до конца эпизода.
- `steps_after_key_until_exit_mean` — среднее число шагов от ключа до выхода (если был выход).
- `steps_after_key_until_death_mean` — среднее число шагов от ключа до смерти (если умер).
- Ключ берётся по HUD: `key_door` (слот KEY DOOR).

---

## 4. Ограничения

- Используются только obs, HUD и room_hash — без RAM/координат эмулятора.
- reward не меняется.
- Для новых колонок нужен новый прогон (старые `metrics.csv` с другим header при append будут иметь лишние колонки в новых строках).
