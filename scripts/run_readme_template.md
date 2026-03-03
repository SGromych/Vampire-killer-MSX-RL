# Run Directory

## Содержимое

- **metrics.csv** — метрики за каждое PPO-обновление
- **train.log** — полный лог обучения (stdout/stderr)
- **config_snapshot.json** — снимок гиперпараметров и reward_config на старт
- **checkpoints/ppo/** — last.pt, backup_*.pt, epoch_*.pt

## Метрики (ключевые)

| Метрика | Описание |
|---------|----------|
| steps_per_sec | Шаги env в секунду (суммарно) |
| steps_per_sec_per_env | steps_per_sec / num_envs |
| rollout_fps | Кадры/obs в секунду при rollout |
| unique_rooms_mean/max | Среднее/макс. число уникальных комнат за эпизод |
| ep_return_mean | Средний return эпизода |
| ep_return_env0/env1 | Return по env (multi-env) |
| stage00_exit_rate | Доля эпизодов с выходом из STAGE 00 |
| stage00_time_to_exit_steps | Шаги до перехода на STAGE 01 |
| backtrack_rate_mean | Доля backtrack-переходов (A→B→A) |
| room_dwell_steps_mean | Ср. время (steps) в комнате до перехода |
| door_encounter_count_mean | Детекции двери (best effort) |
| steps_after_key_total_mean | Шаги после получения ключа |
| policy_loss, value_loss, entropy | PPO loss и энтропия |
| recurrent_hidden_norm_mean/std | LSTM: норма hidden state (если recurrent) |
| resume_count, last_resume_update | Счётчик ресюмов и последний update |

## Запуск

```bash
python train_ppo.py --num-envs 2 --use-runs-dir --run-name exp01
python train_supervisor.py  # ночной прогон с configs/night_training.json
```

В конце обучения: `LAST_RUN_PATH=runs/<timestamp>_<git>_<name>/`
