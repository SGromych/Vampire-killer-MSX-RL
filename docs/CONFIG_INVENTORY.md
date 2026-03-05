# Configuration inventory

Grouped list of configuration sources (argparse flags, dataclass fields, env vars).

## Argparse flags (by file)

### benchmark_env.py

- `--backend`: (no help)
- `--steps`: (no help)
- `--action-repeat`: (no help)
- `--decision-fps`: (no help)
- `--workdir`: (no help)
- `--window-crop`: (no help)
- `--window-title`: (no help)
- `--capture-lag-ms`: (no help)

### debug_env.py

- `--capture-backend`: (no help)
- `--workdir`: (no help)
- `--debug`: включить [debug] вывод (по умолчанию вкл)
- `--debug-force-action`: действие: RIGHT, LEFT, ...
- `--debug-episode-max-steps`: (no help)
- `--debug-every`: (no help)
- `--debug-dump-frames`: (no help)

### demos\record_demo.py

- `--mode`: (no help)
- `--run-id`: (no help)
- `--max-steps`: (no help)
- `--max-minutes`: (no help)
- `--fps`: частота опроса клавиш (15–20 уменьшает лаги)
- `--poll-ms`: интервал опроса openMSX (мс), меньше = быстрее реакция
- `--save-raw-every`: зарезервировано, пока не используется
- `--preview`: (no help)
- `--action-repeat`: N внутр. шагов на 1 захват (1 = backward compat)
- `--decision-fps`: частота решений (10–15 Hz), None=макс.
- `--capture-backend`: (no help)

### demos\replay_demo.py

- `run_id`: имя подкаталога в demos/runs/
- `--mode`: (no help)
- `--fps`: (no help)

### demos\validate_demo.py

- `run_id`: имя подкаталога в demos/runs/
- `--preview`: (no help)

### project_config.py

- `--config`: Primary config file (JSON); CLI overrides
- `--reward-config`: Reward config JSON; if missing when specified -> fail
- `--run-dir`: Explicit run directory (absolute or relative to CWD)
- `--run-name`: (no help)
- `--log-dir`: (no help)
- `--use-runs-dir`: (no help)
- `--checkpoint-dir`: (no help)
- `--bc-checkpoint`: (no help)
- `--epochs`: (no help)
- `--rollout-steps`: (no help)
- `--ppo-epochs`: (no help)
- `--batch-size`: (no help)
- `--lr`: (no help)
- `--gamma`: (no help)
- `--gae-lambda`: (no help)
- `--clip-eps`: (no help)
- `--max-episode-steps`: (no help)
- `--step-penalty`: (no help)
- `--device`: (no help)
- `--arch`: (no help)
- `--action-repeat`: (no help)
- `--decision-fps`: (no help)
- `--capture-backend`: (no help)
- `--capture-fallback`: (no help)
- `--num-envs`: (no help)
- `--tmp-root`: (no help)
- `--post-action-delay-ms`: (no help)
- `--novelty-reward`: (no help)
- `--entropy-coef`: (no help)
- `--value-loss-coef`: (no help)
- `--entropy-floor`: (no help)
- `--stuck-updates`: (no help)
- `--resume`: (no help)
- `--checkpoint-every`: (no help)
- `--recurrent`: (no help)
- `--lstm-hidden-size`: (no help)
- `--sequence-length`: (no help)
- `--dry-run-seconds`: (no help)
- `--no-quiet`: (no help)
- `--summary-every`: (no help)
- `--summary-interval-sec`: (no help)
- `--debug`: (no help)
- `--debug-room-change`: (no help)
- `--debug-every`: (no help)
- `--debug-episode-max-steps`: (no help)
- `--debug-dump-frames`: (no help)
- `--debug-force-action`: (no help)
- `--ignore-death`: (no help)
- `--window-title-pattern`: (no help)
- `--window-rects-json`: (no help)
- `--no-reset-handshake`: (no help)
- `--nudge-right-steps`: (no help)
- `--stuck-nudge-steps`: (no help)
- `--dump-hud-every-n-steps`: (no help)
- `--perf`: (no help)
- `--export-metrics`: (no help)

### scripts\calibrate_window_capture.py

- `--title`: Substring in window title
- `--dump-frames`: Save N sample frames to disk
- `--out-dir`: Directory for dumped frames and config snippet
- `--fps-frames`: Frames to measure FPS

### scripts\debug_stage_detector.py

- `--image`: путь к скриншоту (PNG) для разбора
- `--dump`: запустить env, сделать N шагов, вывести stage/conf на каждом
- `--capture-backend`: (no help)

### test_policy.py

- `--checkpoint`: (no help)
- `--max-steps`: (no help)
- `--device`: (no help)
- `--workdir`: workdir для openMSX; по умолчанию checkpoints/bc/run
- `--stop-on-death`: завершить прогон при падении полоски жизни (смерть)
- `--smooth`: сглаживание: majority vote по последним N действиям (0=выкл)
- `--sticky`: при NOOP один раз повторить последнее ненулевое действие (меньше замираний)
- `--max-idle-steps`: анти-залипание: если N шагов подряд нет движения (RIGHT/LEFT/UP/DOWN), принудите
- `--transition-assist`: при резкой смене кадра (переход между экранами) при NOOP повторять последнее дви
- `--stair-assist-steps`: если N шагов подряд только RIGHT/LEFT (без UP/DOWN) — попробовать UP (лестница);
- `--capture-backend`: (no help)

### test_ppo.py

- `--checkpoint`: (no help)
- `--max-steps`: (no help)
- `--device`: (no help)
- `--workdir`: (no help)
- `--stop-on-death`: (no help)
- `--deterministic`: argmax вместо sample
- `--smooth`: (no help)
- `--sticky`: (no help)
- `--max-idle-steps`: (no help)
- `--transition-assist`: (no help)
- `--stair-assist-steps`: при right-left цикле пробовать UP
- `--capture-backend`: (no help)
- `--diagnose-policy`: диагностика детерминированной/стохастической политики, без изменения обучения

### tools\diagnose_throughput.py

- `--minutes`: Minutes per test
- `--rollout-steps`: Steps per rollout
- `--envs`: Comma-separated: 1,2
- `--modes`: capture_on,capture_off
- `--policy`: random,policy
- `--train`: off,on
- `--rom`: (no help)
- `--bc-checkpoint`: (no help)
- `--device`: (no help)

### tools\test_room_metrics.py

- `--episodes`: Number of episodes to run
- `--num-envs`: Number of parallel envs (only 1 supported)
- `--policy`: Policy: random actions
- `--rom`: Path to VAMPIRE.ROM (default: ROOT/VAMPIRE.ROM)
- `--max-steps`: Max steps per episode (truncation)

### train_bc.py

- `--runs`: run_id через пробел (например run_full_01 run_full_02). По умолчанию — все прого
- `--epochs`: больше эпох — лучше выучивает редкие действия
- `--batch-size`: (no help)
- `--lr`: (no help)
- `--checkpoint-dir`: (no help)
- `--device`: (no help)
- `--noop-weight`: вес класса NOOP в loss (меньше 1 = реже предсказывать NOOP)
- `--oversample`: во сколько раз чаще сэмплировать шаги с action!=NOOP
- `--frame-stack`: число кадров в стопке (4 — рекомендуется)
- `--deep`: усиленная сеть (BCNetDeep): лучше прыжки/удары/свечки
- `--rare-weight`: доп. вес редких действий (ATTACK, прыжки) в loss
- `--move-weight`: вес движения RIGHT/LEFT/UP/DOWN в loss; приоритет «идти по лабиринту (в т.ч. по 

### train_ppo.py

- `--checkpoint-dir`: (no help)
- `--bc-checkpoint`: инициализация из BC (best.pt)
- `--epochs`: число PPO-обновлений (каждое = rollout + train)
- `--rollout-steps`: шагов на один rollout
- `--ppo-epochs`: эпох обучения на rollout
- `--batch-size`: (no help)
- `--lr`: (no help)
- `--gamma`: (no help)
- `--gae-lambda`: (no help)
- `--clip-eps`: (no help)
- `--max-episode-steps`: (no help)
- `--step-penalty`: (no help)
- `--device`: (no help)
- `--arch`: (no help)
- `--action-repeat`: N внутренних шагов на 1 захват (backward compat: 1)
- `--decision-fps`: фикс. частота решений (10–15 Hz), None=макс. скорость
- `--capture-backend`: (no help)
- `--num-envs`: число параллельных env (уникальный workdir на инстанс)
- `--tmp-root`: база для workdir при num_envs>1
- `--post-action-delay-ms`: задержка после действия перед grab (мс). При num-envs>1 по умолчанию 50
- `--run-name`: имя эксперимента (суффикс в runs/<ts>_<git>_<name>/)
- `--log-dir`: каталог логов (по умолчанию checkpoint-dir)
- `--use-runs-dir`: использовать runs/<timestamp>_<git>_<runname>/ как run directory
- `--run-dir`: явно задать run directory (для supervisor)
- `--export-metrics`: скопировать metrics.csv в указанную папку после обучения
- `--config`: путь к JSON конфигу эксперимента (переопределяется CLI)
- `--reward-config`: путь к JSON конфигу наград (RewardConfig)
- `--novelty-reward`: переопределить novelty_reward (награда за новую комнату)
- `--entropy-coef`: коэффициент энтропии в loss PPO
- `--value-loss-coef`: коэффициент value loss в loss PPO
- `--entropy-floor`: ниже — предупреждение о коллапсе политики
- `--stuck-updates`: после скольких обновлений без роста unique_rooms — предупреждение
- `--resume`: продолжить с last.pt (update, optimizer, RNG)
- `--checkpoint-every`: период сохранения rolling backup (0=выкл); хранятся последние 5
- `--recurrent`: LSTM после encoder (память для лабиринта)
- `--lstm-hidden-size`: размер скрытого состояния LSTM
- `--sequence-length`: длина последовательности для лога/диагностики (0=не используется)
- `--dry-run-seconds`: прогнать N секунд и вывести steps/sec, updates/hour (для оценки скорости перед н
- `--no-quiet`: включить подробный вывод (Update/env return, backup saved); по умолчанию quiet (
- `--summary-every`: печатать компактный summary каждые N обновлений
- `--summary-interval-sec`: печатать summary не реже чем раз в N секунд
- `--debug`: диагностика: room_hash, stage, stuck, действие каждые N шагов
- `--debug-room-change`: печатать [ENV i] room change prev->new при смене комнаты
- `--debug-every`: печатать [debug] строку каждые N шагов
- `--debug-episode-max-steps`: ограничить эпизод при debug (0=не менять)
- `--debug-dump-frames`: сохранить первые N кадров в debug_frames/ (0=выкл)
- `--debug-force-action`: подменить действие: RIGHT, LEFT, ... для проверки
- `--ignore-death`: не ставить terminated=True при смерти (только логировать, для проверки ложных ср
- `--window-title-pattern`: подстрока заголовка окна для window capture (по умолчанию openMSX)
- `--window-rects-json`: путь к JSON с per-env окнами: {"0": {"title": "...", "crop": [x,y,w,h]}, ...}
- `--debug-single-episode`: запустить ровно 1 эпизод на каждый env, вывести reset/termination и выйти (для о
- `--no-reset-handshake`: при num_envs>1 не включать reset handshake (для отладки, если кнопки в env 1 не 
- `--nudge-right-steps`: в начале каждого эпизода N шагов RIGHT (подталкивание вправо, чтобы не застреват
- `--stuck-nudge-steps`: при застревании (stuck) N шагов по очереди RIGHT/LEFT/UP/DOWN для попытки выхода
- `--dump-hud-every-n-steps`: fix-room-metrics: сохранять HUD crop каждые N шагов в run_dir/debug/ (0=выкл, на
- `--perf`: throughput diagnostics: собирать t_action/t_capture/t_reward (p50/p95) в info

### train_supervisor.py

- `--preflight`: run 60s dry-run preflight (no supervision loop)
- `--resume-smoke-test`: run short two-stage resume smoke test

## Dataclass config fields (by file)

### msx_env\dataset.py

- `version`: `int`
- `game_id`: `str`
- `machine_config`: `str`
- `rom_path`: `str`
- `action_set_version`: `str`
- `obs_shape`: `Tuple[int, ...]`
- `obs_dtype`: `str`
- `step_rate_hz`: `float`
- `created_at`: `float`
- `total_steps`: `int`
- `max_minutes`: `float`
- `run_id`: `str`

### msx_env\env.py

- `rom_path`: `str`
- `workdir`: `str`
- `frame_size`: `Tuple[int, int]`
- `poll_ms`: `int`
- `hold_keys`: `bool`
- `hud_reward`: `bool`
- `terminated_on_death`: `bool`
- `max_episode_steps`: `int`
- `action_repeat`: `int`
- `decision_fps`: `float | None`
- `capture_backend`: `str`
- `window_crop`: `Tuple[int, int, int, int] | None`
- `window_title`: `str | None`
- `capture_lag_ms`: `float`
- `post_action_delay_ms`: `float`
- `reward_config`: `RewardConfig | None`
- `instance_id`: `int | str | None`
- `tmp_root`: `str`
- `soft_reset`: `bool`
- `perf_log_interval`: `int`
- `quiet`: `bool`
- `openmsx_log_dir`: `str | None`
- `debug`: `bool`
- `debug_every`: `int`
- `debug_episode_max_steps`: `int`
- `debug_dump_frames`: `int`
- `debug_force_action`: `str`
- `debug_dump_dir`: `str | None`
- `debug_room_change`: `bool`
- `ignore_death`: `bool`
- `death_warmup_steps`: `int`
- `reset_handshake_stable_frames`: `int`
- `reset_handshake_conf_min`: `float`
- `reset_handshake_timeout_s`: `float`
- `dump_hud_every_n_steps`: `int`
- `dump_hud_dir`: `str | None`
- `perf_profile`: `bool`
- `capture_off`: `bool`

### msx_env\hud_parser.py

- `weapon`: `bool`
- `key_chest`: `bool`
- `key_door`: `bool`
- `items`: `int`

### msx_env\human_controller.py

- `fps`: `int`

### msx_env\perf_timers.py

- `max_samples`: `int`
- `samples`: `dict[str, list[float]]`
- `per_env_samples`: `dict[str, list[float]]`
- `update_wall_sec`: `deque`

### msx_env\reward\components.py

- `reward`: `float`
- `extra`: `dict[str, Any]`
- `last_weapon_step`: `int`
- `last_key_chest_step`: `int`
- `last_key_door_step`: `int`
- `last_items_step`: `int`
- `seen_rooms`: `set[str]`
- `candidate_deque`: `deque`
- `last_novelty_step`: `int`
- `last_room_change_step`: `int`
- `last_room_hash`: `str | None`
- `recent_x_proxy`: `deque`

### msx_env\reward\config.py

- `version`: `str`
- `step_penalty`: `float`
- `death_penalty`: `float`
- `pickup_weapon`: `float`
- `pickup_key_chest`: `float`
- `pickup_key_door`: `float`
- `pickup_item`: `float`
- `pickup_cooldown_steps`: `int`
- `novelty_reward`: `float`
- `novelty_persistence_frames`: `int`
- `novelty_saturation_cap`: `int`
- `novelty_saturation_decay`: `float`
- `pingpong_window`: `int`
- `pingpong_min_alternations`: `int`
- `pingpong_penalty`: `float`
- `stuck_no_room_change_steps`: `int`
- `stuck_frame_diff_threshold`: `float`
- `stuck_penalty`: `float`
- `stuck_truncate`: `bool`
- `stuck_penalty_severity_levels`: `list[float] | None`
- `progress_reward_per_pixel`: `float`
- `progress_backtrack_penalty`: `float`
- `backtrack_penalty_enabled`: `bool`
- `backtrack_penalty_value`: `float`
- `room_hash_crop_top`: `int`
- `room_hash_crop_bottom`: `int`
- `novelty_stability_frames`: `int`
- `pingpong_threshold`: `int`
- `key_reward`: `float`
- `door_reward`: `float`
- `stuck_progressive`: `bool`
- `stuck_position_variance_steps`: `int`
- `stuck_position_variance_threshold`: `float`
- `enable_stage_reward`: `bool`
- `stage_step_penalty`: `float`
- `stage_advance_bonus`: `float`
- `stage_only_for`: `list[int] | None`
- `stage_conf_threshold`: `float`
- `stage_stability_frames`: `int`
- `episode_room_debounce_k`: `int`
- `episode_stage_stable_frames`: `int`
- `episode_playfield_crop_top`: `int`
- `episode_playfield_crop_bottom`: `int`
- `episode_playfield_crop_right`: `int`
- `episode_unique_rooms_sanity_warn`: `int`

### msx_env\reward\diagnostics.py

- `room_transition_count`: `int`
- `room_hash_buffer`: `deque`
- `backtrack_count`: `int`
- `room_dwell_steps_sum`: `float`
- `room_dwell_visits`: `int`
- `last_room_transition_step`: `int`
- `door_encounter_count`: `int`
- `loop_len_buffer`: `deque`
- `stage00_entered_step`: `int`
- `stage00_exit_step`: `int | None`
- `stage00_exit_success`: `bool`
- `stage00_room_transitions`: `int`
- `in_stage0`: `bool`
- `key_obtained_at_step`: `int | None`
- `steps_after_key_total`: `int | None`
- `steps_after_key_until_exit`: `int | None`
- `steps_after_key_until_death`: `int | None`

### msx_env\reward\episode_metrics.py

- `candidate_room_hash`: `str | None`
- `candidate_count`: `int`
- `stable_room_hash`: `str | None`
- `stable_stage`: `int`
- `stable_room_id`: `str | None`
- `prev_stable_room_id`: `str | None`
- `episode_room_set`: `set`
- `episode_transition_count`: `int`
- `episode_room_history`: `deque`
- `episode_room_dwell_steps`: `int`
- `episode_dwell_accum`: `list`
- `stage00_exit_recorded`: `bool`
- `stage00_exit_steps`: `int`
- `stage_prev`: `int`
- `stage_candidate`: `int`
- `stage_candidate_frames`: `int`
- `backtrack_count_ep`: `int`
- `_last_stable_room_ids`: `deque`
- `_last_raw_hashes`: `deque`

### msx_env\reward\logger.py

- `episode_components`: `dict[str, float]`
- `episode_total`: `float`
- `step_count`: `int`
- `last_extra`: `dict | None`

### msx_env\reward\policy.py

- `step_index`: `int`
- `prev_obs`: `np.ndarray | None`
- `prev_hud`: `HudState | None`
- `prev_room_hash`: `str | None`
- `pickup_state`: `PickupState`
- `novelty_state`: `NoveltyState`
- `stuck_state`: `StuckState`
- `hash_history`: `deque`
- `pingpong_count_episode`: `int`
- `key_rewarded_this_episode`: `bool`
- `door_rewarded_this_episode`: `bool`
- `accepted_stage`: `int`
- `steps_in_current_stage`: `int`
- `stage_candidate`: `int`
- `stage_candidate_frames`: `int`
- `diagnostics`: `EpisodeDiagnostics`
- `episode_room_tracker`: `EpisodeRoomTracker`

### project_config.py

- `run_dir`: `Path`
- `experiment_name`: `str`
- `seed`: `int`
- `device`: `str`
- `use_runs_dir`: `bool`
- `checkpoint_dir`: `Path`
- `bc_checkpoint`: `Path | None`
- `epochs`: `int`
- `rollout_steps`: `int`
- `ppo_epochs`: `int`
- `batch_size`: `int`
- `lr`: `float`
- `gamma`: `float`
- `gae_lambda`: `float`
- `clip_eps`: `float`
- `entropy_coef`: `float`
- `value_loss_coef`: `float`
- `max_episode_steps`: `int`
- `step_penalty`: `float`
- `arch`: `str`
- `entropy_floor`: `float`
- `stuck_updates`: `int`
- `resume`: `bool`
- `checkpoint_every`: `int`
- `recurrent`: `bool`
- `lstm_hidden_size`: `int`
- `sequence_length`: `int`
- `nudge_right_steps`: `int`
- `stuck_nudge_steps`: `int`
- `no_reset_handshake`: `bool`
- `dry_run_seconds`: `float`
- `summary_every`: `int`
- `summary_interval_sec`: `float`
- `export_metrics`: `Path | None`
- `rom_path`: `Path`
- `frame_size`: `tuple[int, int]`
- `action_repeat`: `int`
- `decision_fps`: `float | None`
- `capture_backend`: `str`
- `post_action_delay_ms`: `float`
- `max_episode_steps`: `int`
- `tmp_root`: `str`
- `soft_reset`: `bool`
- `num_envs`: `int`
- `window_title`: `str | None`
- `window_rects_path`: `Path | None`
- `reset_handshake_stable_frames`: `int`
- `reset_handshake_conf_min`: `float`
- `reset_handshake_timeout_s`: `float`
- `debug`: `bool`
- `debug_every`: `int`
- `debug_episode_max_steps`: `int`
- `debug_dump_frames`: `int`
- `debug_force_action`: `str`
- `debug_room_change`: `bool`
- `ignore_death`: `bool`
- `dump_hud_every_n_steps`: `int`
- `perf_profile`: `bool`
- `quiet`: `bool`
- `openmsx_log_dir`: `Path | None`
- `backend`: `str`
- `fallback_policy`: `str`
- `window_crop`: `tuple[int, int, int, int] | None`
- `window_title`: `str | None`
- `capture_lag_ms`: `float`
- `log_diagnostics`: `bool`
- `train_log`: `str`
- `supervisor_log`: `str`
- `metrics_csv`: `str`
- `config_snapshot`: `str`
- `resolved_paths`: `str`
- `metrics_schema`: `str`
- `checkpoint_subdir`: `str`
- `run`: `RunConfig`
- `ppo`: `PPOConfig`
- `env_schema`: `EnvConfigSchema`
- `reward_config`: `Any`
- `reward_config_path`: `Path | None`
- `capture`: `CaptureConfig`
- `logging`: `LoggingConfig`
- `layout`: `"RunLayout"`

## Environment variables

- `RUN_DIR`
- `SUPERVISOR_CRASH_FLAG`
- `SUPERVISOR_RESTART_COUNT`
- `X`
