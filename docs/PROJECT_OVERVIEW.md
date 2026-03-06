## Project overview

**Goal.** Train a PPO agent to play the MSX2 game *Vampire Killer* via the openMSX emulator, using file‑based control and screen capture, with a modular reward system and support for multi‑env and recurrent PPO.

Related entry points:
- High‑level context: `docs/CONTEXT.md`
- Modules & CLI flags: `docs/MODULES_AND_FLAGS.md`
- Training details & metrics: `docs/TRAINING.md`
- Reward system: `docs/REWARD.md`, `docs/REWARD_V3_AUDIT.md`
- Capture backends: `docs/CAPTURE.md`
- Multi‑env debug history: `docs/MULTI_ENV_DEBUG_SESSION.md`

---

## Architecture

- **Environment**: `msx_env.env.VampireKillerEnv` wraps openMSX:
  - Control via `openmsx_bridge.OpenMSXFileControl` (file protocol: `commands.tcl` / `reply.txt`).
  - Capture via `msx_env.capture` (backends `png`, `single`, `window`, `dxcam`; default in training: `dxcam`, no screenshots to disk).
  - Observation: \(84\times84\) grayscale `uint8`, frame stack of 4 frames for policies.
- **Action space**: 10 discrete actions (NOOP, RIGHT, LEFT, UP, DOWN, ATTACK, RIGHT_JUMP, LEFT_JUMP, RIGHT_JUMP_ATTACK, LEFT_JUMP_ATTACK), defined in `msx_env.env`.
- **Models**:
  - **BC**: `msx_env.bc_model.BCNet/BCNetDeep`, trained from human demos (`train_bc.py`).
  - **PPO**: `msx_env.ppo_model.ActorCritic` (CNN encoder + optional LSTM + actor/critic heads).
- **Reward**: `msx_env.reward.*` implements `RewardPolicy` with configurable `RewardConfig` (v1/v3).

---

## Environment and capture

- **Config**: `EnvConfig` (see `docs/MODULES_AND_FLAGS.md`), key fields:
  - `rom_path`, `workdir`, `frame_size=(84,84)`, `capture_backend`, `action_repeat`, `decision_fps`,
  - `terminated_on_death`, `max_episode_steps`, `reward_config`, `soft_reset`, `post_action_delay_ms`,
  - multi‑env fields: `instance_id`, `tmp_root`, reset‑handshake / perf / debug options.
- **Capture backends** (`docs/CAPTURE.md`, `msx_env/capture.py`):
  - `dxcam`: (default in train_ppo) grab window by PID into memory; no PNG files; Windows only.
  - `png` / `single`: openMSX writes a PNG file, Python reads it and converts to \(84\times84\) grayscale.
  - `window`: grabs pixels from the openMSX window via `dxcam`/`mss` (by title/crop), then the same preprocessing.
  - All backends are required to produce identical obs format for compatibility with existing datasets.

---

## Reward system

Main reference: `docs/REWARD.md`, implementation in `msx_env/reward/policy.py`, `config.py`, `hashers.py`,
`episode_metrics.py`, `event_detectors.py`, and diagnostics in `docs/REWARD_V3_AUDIT.md`, `docs/DIAGNOSTICS_PR.md`,
`docs/EPISODE_METRICS_FIX.md`.

Core components (v1/v3):
- **Step penalty**: constant per step, discourages idle episodes.
- **Death penalty**: large negative reward when life bar crosses death thresholds; death detection is implemented in
  `msx_env.env` + `life_bar.py` with hysteresis and Stage‑00 safeguards.
- **Pickup (HUD)**: rewards for weapon, chest key, door key, items based on HUD changes with cooldown.
- **Novelty**: reward for entering new rooms; uses stable room hashing over playfield crops and a hysteresis window.
- **Ping‑pong penalty**: penalizes repeated A↔B↔A room alternations.
- **Stuck penalty**: detects no‑movement situations via room hash + frame difference and applies penalties and optional truncation.
- **Stage rewards (optional)**: per‑stage step penalty and bonus for stage progression, gated by HUD stage confidence.
- **Key/door rewards (v3)**: once‑per‑episode rewards based on `KeyDetector`/`DoorDetector`.
- **Backtrack penalty (optional)**: extra penalty when backtracking events are detected.

Episode‑level metrics (unique rooms per episode, backtrack rate, Stage‑00 exit statistics, etc.) are produced by
`EpisodeRoomTracker` and documented in `docs/EPISODE_METRICS_FIX.md`.

---

## Training pipeline (PPO)

Detailed spec: `docs/TRAINING.md`, CLI flags and outputs: `docs/MODULES_AND_FLAGS.md`.

High‑level loop (`train_ppo.py`):
1. Initialize envs (`VampireKillerEnv`) and model (`ActorCritic`), optionally from a BC checkpoint.
2. For each PPO update:
   - **Rollout collection**:
     - Loop over `rollout_steps`, iterate envs round‑robin,
     - Build stacked obs from frame buffers, call `model.get_action(...)` (stochastic), apply env step,
     - Log rewards, value estimates, done flags, reward components, and recurrent hidden states if enabled.
   - **Advantage computation (GAE)**:
     - For each env separately, compute advantages/returns over its slice of the trajectory.
   - **PPO update**:
     - Shuffle rollout, run multiple epochs over mini‑batches,
     - Compute clipped policy loss, value loss, entropy, approx KL, apply gradient step with clipping.
   - **Metrics & guardrails**:
     - Aggregate reward statistics, episode stats, entropy, KL, explained variance; log to console and `metrics.csv`;
     - Emit warnings on NaNs, low entropy, critic divergence, and “stuck” unique_rooms.
   - **Checkpointing**:
     - Save `last.pt` and periodic `epoch_<N>.pt`, rotate rolling backups if enabled.

Single‑env and multi‑env rollouts share the same logic; multi‑env uses a flattened rollout with per‑env indexing.

---

## Metrics

All core training metrics and their exact meanings are documented in `docs/TRAINING.md`
and, for episode‑level additions, `docs/EPISODE_METRICS_FIX.md` and `docs/DIAGNOSTICS_PR.md`.

Key metrics produced per update:
- **Optimization**: `policy_loss`, `value_loss`, `entropy`, `approx_kl`, `explained_var`.
- **Reward / performance**: `reward_mean`, `ep_return_mean/min/max`, `ep_steps_mean/min/max`,
  per‑env `ep_return_env{i}`, `ep_len_env{i}`.
- **Exploration / environment**: `unique_rooms_mean/min/max`, per‑env `unique_rooms_env{i}`,
  `room_transition_env{i}`, `stage` statistics, Stage‑00 exit metrics, backtrack rate.
- **Stability / events**: `deaths`, `stuck_events`, recurrent hidden norms/deltas, episode‑level backtrack and
  Stage‑00 exit metrics (`*_ep`).
- **Reward decomposition**: `reward_components` on step‑level and aggregated per update (step, pickup, death,
  novelty, pingpong, stuck, key, door, stage_step, stage_advance, backtrack).

The source of truth for the CSV layout and aggregation logic is `train_ppo.py`.

---

## Checkpoints and resume

Primary reference: `docs/SUPERVISOR_AUDIT.md`, implementation in `train_ppo.py` and `train_supervisor.py`.

- **`last.pt`**:
  - Saved after each PPO update.
  - Contains: `state_dict`, `frame_stack`, `arch`, `update`, `optimizer_state`, `rng_state`, and recurrent flags.
- **Rolling backups**:
  - When `--checkpoint-every N > 0`, a ring of `backup_0.pt`…`backup_4.pt` is maintained using `_rotate_backups`.
  - Used by the night supervisor for rollback on NaNs.
- **Periodic snapshots**:
  - Every 10 updates an `epoch_<N>.pt` with lightweight `{state_dict, frame_stack, arch}` is written.
- **Resume**:
  - `--resume` loads `last.pt`, restores model, optimizer (if present), RNG states and resumes from `update+1`.
  - Night supervisor always uses `--resume` so long runs carry on from the latest checkpoint.

---

## Multi‑env support

Multi‑env behaviour is documented in `docs/MODULES_AND_FLAGS.md` and `docs/MULTI_ENV_DEBUG_SESSION.md`,
and implemented via `msx_env.make_env.make_env`.

- **Isolation**:
  - Each env gets its own `workdir = tmp_root/<rank>` and `instance_id = rank`.
  - Each instance has its own openMSX process, TCL command/reply files and capture backend.
- **Rollout**:
  - Rollout interleaves envs (`env0, env1, env0, env1, …`), but GAE operates per‑env using strided indices.
- **Capture & window mode**:
  - File backends (`png`/`single`) are safe by design for multi‑env.
  - Window capture in multi‑env requires per‑env window rects; otherwise training falls back to file capture.
- **Reset & stability**:
  - `soft_reset=True` and reset‑handshake options help ensure stable initial frames before each episode.

---

## Recurrent PPO (LSTM)

Design and behaviour are described in `docs/PPO_MODEL.md` and `docs/PPO_RECURRENT_BENCHMARK.md`.

- **Flags**: `--recurrent`, `--lstm-hidden-size`, `--sequence-length` (reserved for future truncated BPTT).
- **Architecture**:
  - LSTM is inserted between encoder and heads; hidden state \((h, c)\) has shape \((1, B, H)\).
- **State handling**:
  - Hidden state is carried across steps within an episode and reset on `done` per env.
  - During training, per‑step LSTM states are stored to re‑run mini‑batches with correct hidden context.
- **Metrics**:
  - Training logs recurrent diagnostics (e.g. `recurrent_hidden_norm_mean`, `recurrent_hidden_delta_mean`),
    used to detect non‑updating hidden states.

Empirically, recurrent PPO should not degrade performance vs non‑recurrent CNN and may improve exploration on
longer‑horizon levels; `docs/PPO_RECURRENT_BENCHMARK.md` sketches small experiments to validate this.

---

## Night‑run readiness

Night‑run aspects are covered in `docs/TRAINING.md` (dry‑run, summaries, guardrails) and
`train_supervisor.py` / `docs/SUPERVISOR_AUDIT.md` (supervisor).

- **Dry‑run mode**:
  - `--dry-run-seconds N` runs a short benchmark and prints steps/sec, updates/hour and an estimate of max steps.
- **Summaries & guardrails**:
  - Periodic compact summaries (uptime, steps/s, health indicators).
  - Warnings for NaNs, low entropy, bad explained variance and stalled `unique_rooms`.
- **Supervisor**:
  - `train_supervisor.py` reads `configs/night_training.json`, launches `train_ppo.py` with `--resume`,
    restarts on crashes, rolls back `last.pt` from backups if `metrics.csv` contains NaNs, and runs a watchdog
    thread that kills runs with no metric updates for too long.

For night runs, monitor: latest summaries in `train.log`, `metrics.csv` (entropy, unique_rooms, deaths, stuck),
supervisor log, and disk usage of checkpoints and backups.

### Run directory and where to find metrics

- **With `use_runs_dir: true`** (e.g. in `configs/night_training.json`): each supervisor start creates
  `runs/<YYYYMMDD>_<HHMMSS>_<gitshort>_<run_name>/` (e.g. `runs/20260304_181828_58548f2_auto_night/`).
  The **latest** run by wall-clock time is the folder with the most recent **LastWriteTime** (e.g. files
  written into the small hours of the next day belong to “tonight’s” run).
- **With `use_runs_dir: false`**: run dir is `<checkpoint_dir>/<run_name>/` (e.g. `checkpoints/ppo/auto_night/`).
- **Metrics file:** normally `metrics.csv` inside the run dir. In some runs you may see `metrics1.csv` or
  `supervisor1.log` (e.g. from manual copies or alternate setups); check both the standard names and any
  `metrics*.csv` / `supervisor*.log` when looking for “today’s” data.
- **Key files in run dir:** `metrics.csv` (or `metrics1.csv`), `train.log`, `config_snapshot.json`,
  `metrics_schema.json`, `supervisor.log` (or `supervisor1.log`).

