# Аудит RewardPolicy (перед v3)

## Novelty
- **Реализация:** `hashers.block_mean_hash(obs)` → блоки 4×4, downscale 32×32; `room_hash_with_hysteresis(obs, candidate_deque, persistence)` — стабильный хэш только если один и тот же хэш повторялся `persistence` (K) кадров подряд. `NoveltyState.candidate_deque` maxlen=10.
- **Награда:** за первый визит комнаты в эпизоде, с saturation decay при cap.

## Pingpong
- **Реализация:** `pingpong_component(cfg, room_hash, step, hash_history)`. `hash_history` в `RewardPolicyState` — `deque(maxlen=50)`. Штраф при чередовании A-B-A-B в окне `pingpong_window` (50), минимум `pingpong_min_alternations` (4) переключений.

## Stuck
- **Реализация:** `stuck_component(cfg, obs, prev_obs, room_hash, step, stuck_state)`. Условие: нет смены комнаты ≥ `stuck_no_room_change_steps` (200) и `frame_diff(prev, obs) < stuck_frame_diff_threshold` (0.02). Возвращает penalty и `stuck_truncate`.

## Логирование компонентов
- **compute()** возвращает `(total, components_dict, extra)`. В `extra`: `unique_rooms`, `room_hash`, `pingpong_event`, `stuck_event`, `stuck_truncate`. В `info` попадают `reward_components`, `episode_reward_components`, `reward_*` из extra. Episode report в `EpisodeRewardLogger.format_episode_report()`.

## EventDetector
- **Интерфейс:** `EventDetector.detect(obs, info) -> dict`. Есть заглушки `KeyDoorDetector`, `HPDetector`, `ProgressProxyDetector`.
