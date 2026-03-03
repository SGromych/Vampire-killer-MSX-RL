# STAGE 00 early-training audit & diagnosis

## Pipeline audit (no code changes)

### 1. room_hash

- **Where:** `msx_env/reward/hashers.py`: `block_mean_hash(obs)` → `room_hash_with_hysteresis(obs, candidate_deque, persistence)`.
- **Input:** `obs` is the **full frame** (84×84 grayscale) passed from env after `_grab_frame_and_obs()` (resize of full capture to `frame_size`). So **HUD is included** (top rows: life, STAGE, weapons, keys).
- **Computation:** obs → downscale to 32×32 → 4×4 grid of block means → SHA1. No crop; whole 84×84 is used.
- **Implication:** Top blocks cover HUD. When the character moves right, only the gameplay area changes; HUD is largely static. So **room_hash can stay constant until a full screen transition**, and may be dominated by HUD.

### 2. unique_rooms

- **Where:** `msx_env/reward/components.py` `novelty_component()` returns `len(state.seen_rooms)`.
- **Level:** **Episode-level**. `NoveltyState.seen_rooms` is per episode; reset on policy reset (each episode).
- **Wiring:** In policy, `extra["unique_rooms"] = unique_count`; in train_ppo, episode_stats take `info.get("reward_unique_rooms", 0)` at episode end. So **unique_rooms is correctly wired** to the novelty detector.
- **Implication:** If `room_hash` (and thus `stable_hash`) never changes or never becomes stable (hysteresis), `unique_rooms` stays 0.

### 3. novelty reward and logging

- **Logic:** When a **new** `stable_hash` appears (after persistence K frames), +`novelty_reward` (0.2) and hash is added to `seen_rooms`. Logged as `components["novelty"]`.
- **Stability:** `stable_hash` is only set when the same hash is seen K times in a row (`novelty_persistence_frames` or `novelty_stability_frames`). So fast flicker does not count as a new room.

### 4. stuck detection and penalty

- **Where:** `msx_env/reward/components.py` `stuck_component()`.
- **Conditions:** No room_hash change for `stuck_no_room_change_steps` (200) **and** `frame_diff < stuck_frame_diff_threshold` (0.02) **and** (v3) low X-position variance over last `stuck_position_variance_steps` (60).
- **Penalty:** `stuck_penalty` (-0.2 by default), with `stuck_progressive`: `penalty * (1 + 0.5 * severity)`, severity 0..2. So about -0.2, -0.3, -0.45.
- **Termination:** `stuck_truncate=True` → episode is truncated when stuck fires.
- **Observed:** "stuck ~ -0.001" in top3 is the **per-step average** (e.g. one -0.2 over 200 steps ≈ -0.001/step). So stuck **is** firing, but the per-step signal is small.

### 5. Actions (RIGHT and motion)

- **Mode:** Default `hold_keys=True`: keydown/keyup; keys held until action changes. So RIGHT = keydown("RIGHT") and kept.
- **Timing:** One `_apply_action()` per step (or `action_repeat` times if > 1). Then `post_action_delay_ms` (e.g. 50 for multi-env), then `_grab_frame_and_obs()`. So one frame per step after 50 ms delay.
- **Implication:** If keydown is applied correctly, the character should move. Possible issues: key mapping wrong, capture shows wrong window/crop, or delay too short so frame is unchanged.

### 6. STAGE detection

- **Where:** `msx_env/hud_parser.py` `parse_stage(rgb)` → (stage_int, confidence). ROI `STAGE_DIGITS_ROI` (default 0.08–0.18 x, 0.02–0.12 y in normalized coords).
- **Output:** `info["stage"]`, `info["stage_conf"]`. If template mismatch or low confidence, stage can be 0 and confidence low; then stage rewards are disabled (anti-exploit).
- **Implication:** If HUD layout differs from assumed, or ROI is wrong, we get stage=0 and/or low confidence.

---

## Diagnosis hypotheses

1. **Agent not moving (action timing/mapping)**  
   RIGHT may not be applied long enough per step, or keydown/keyup may be wrong for this build; or capture/crop is a different window or stale.

2. **room_hash too stable (HUD included)**  
   Full 84×84 includes HUD; 4×4 grid on 32×32 can be dominated by top (HUD) blocks. Moving right within the same screen changes mainly center/bottom, so hash may not change until a full screen transition.

3. **unique_rooms metric not wired**  
   **Wired correctly** to novelty_component’s `seen_rooms`. So 0.00 means no new stable room hashes in the episode (see above).

4. **stuck penalty too small / not terminating**  
   Stuck **does** truncate. Per-step magnitude is small (-0.001 average). Increasing base penalty (e.g. -0.3) and progressive steps (-0.3, -0.5, -0.8) would make the signal stronger.

5. **stage detector always 0**  
   ROI or digit templates may not match Vampire Killer HUD; or confidence stays below threshold so we never apply stage rewards and report 0.

---

## Implemented fixes

- **room_hash:** `RewardConfig.room_hash_crop_top=14` (и `room_hash_crop_bottom=0`): хэш считается по кадру без верхних 14 строк (HUD). Для 84×84 это зона геймплея. В v3 по умолчанию включено.
- **stuck:** `stuck_penalty` по умолчанию -0.3. Добавлен `stuck_penalty_severity_levels` (например `[-0.3, -0.5, -0.8]` в v3): при срабатывании stuck штраф берётся из уровня severity.
- **debug:** флаги `--debug`, `--debug-every`, `--debug-episode-max-steps`, `--debug-dump-frames`, `--debug-force-action`. В env при `debug=True` каждые `debug_every` шагов печатается строка `[debug]` (step, action, stage, stage_conf, room_hash, room_changes, unique_rooms, frame_diff, stuck_events, reward, top components). В конце эпизода — `[debug EPISODE]` (total_steps, unique_rooms, room_changes, stage_changes, stuck_events, steps_no_room_change). При `debug_force_action=RIGHT` действие подменяется на RIGHT для проверки движения.
- **debug_env.py:** скрипт без обучения: один env, принудительное действие, вывод [debug] и [debug EPISODE].
