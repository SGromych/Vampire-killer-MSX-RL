"""
Компоненты наград: каждый возвращает вклад в reward и опционально флаги (stuck_truncate и т.д.).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from msx_env.hud_parser import HudState, parse_hud
from msx_env.reward.config import RewardConfig
from msx_env.reward.hashers import (
    block_mean_hash,
    frame_diff_metric,
    position_proxy_x,
    room_hash_with_hysteresis,
)


@dataclass
class ComponentOutput:
    reward: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


# --------- StepPenalty ---------


def step_penalty_component(cfg: RewardConfig) -> float:
    return cfg.step_penalty


# --------- DeathPenalty ---------
# Death и terminated определяются в env по life; компонент только даёт величину штрафа для логирования.
def death_penalty_component(cfg: RewardConfig, terminated_by_death: bool) -> ComponentOutput:
    if not terminated_by_death:
        return ComponentOutput(0.0)
    return ComponentOutput(cfg.death_penalty, {"death": True})


# --------- PickupReward ---------
@dataclass
class PickupState:
    last_weapon_step: int = -9999
    last_key_chest_step: int = -9999
    last_key_door_step: int = -9999
    last_items_step: int = -9999


def pickup_component(
    cfg: RewardConfig,
    prev_hud: HudState | None,
    curr_hud: HudState | None,
    step_index: int,
    state: PickupState,
) -> tuple[float, PickupState]:
    """Награда за подбор с cooldown и нормализованными весами."""
    if curr_hud is None:
        return 0.0, state
    cooldown = cfg.pickup_cooldown_steps
    r = 0.0
    if prev_hud is None:
        return 0.0, state

    if not prev_hud.weapon and curr_hud.weapon and (step_index - state.last_weapon_step) >= cooldown:
        r += cfg.pickup_weapon
        state = PickupState(
            last_weapon_step=step_index,
            last_key_chest_step=state.last_key_chest_step,
            last_key_door_step=state.last_key_door_step,
            last_items_step=state.last_items_step,
        )
    if not prev_hud.key_chest and curr_hud.key_chest and (step_index - state.last_key_chest_step) >= cooldown:
        r += cfg.pickup_key_chest
        state = PickupState(
            last_weapon_step=state.last_weapon_step,
            last_key_chest_step=step_index,
            last_key_door_step=state.last_key_door_step,
            last_items_step=state.last_items_step,
        )
    if not prev_hud.key_door and curr_hud.key_door and (step_index - state.last_key_door_step) >= cooldown:
        r += cfg.pickup_key_door
        state = PickupState(
            last_weapon_step=state.last_weapon_step,
            last_key_chest_step=state.last_key_chest_step,
            last_key_door_step=step_index,
            last_items_step=state.last_items_step,
        )
    if prev_hud.items < curr_hud.items and (step_index - state.last_items_step) >= cooldown:
        r += cfg.pickup_item * (curr_hud.items - prev_hud.items)
        state = PickupState(
            last_weapon_step=state.last_weapon_step,
            last_key_chest_step=state.last_key_chest_step,
            last_key_door_step=state.last_key_door_step,
            last_items_step=step_index,
        )
    return r, state


# --------- RoomNoveltyReward ---------
@dataclass
class NoveltyState:
    seen_rooms: set[str] = field(default_factory=set)
    candidate_deque: deque = field(default_factory=lambda: deque(maxlen=10))
    last_novelty_step: int = -1


def novelty_component(
    cfg: RewardConfig,
    obs: np.ndarray,
    step_index: int,
    state: NoveltyState,
) -> tuple[float, NoveltyState, int, str | None]:
    """
    +novelty_reward за первую встречу комнаты (хэш стабилен K кадров).
    Возвращает (reward, new_state, unique_rooms_count, stable_room_hash).
    v3: K = novelty_stability_frames если >0, иначе novelty_persistence_frames.
    """
    persistence = max(
        1,
        getattr(cfg, "novelty_stability_frames", 0) or cfg.novelty_persistence_frames,
    )
    crop_top = max(0, getattr(cfg, "room_hash_crop_top", 0))
    crop_bottom = max(0, getattr(cfg, "room_hash_crop_bottom", 0))
    stable_hash = room_hash_with_hysteresis(
        obs, state.candidate_deque, persistence,
        crop_rows_top=crop_top, crop_rows_bottom=crop_bottom,
    )

    if stable_hash is None:
        return 0.0, state, len(state.seen_rooms), None

    if stable_hash in state.seen_rooms:
        return 0.0, state, len(state.seen_rooms), stable_hash

    state.seen_rooms.add(stable_hash)
    r = cfg.novelty_reward
    if cfg.novelty_saturation_cap > 0 and len(state.seen_rooms) > cfg.novelty_saturation_cap:
        r *= (cfg.novelty_saturation_decay ** (len(state.seen_rooms) - cfg.novelty_saturation_cap))
    return r, state, len(state.seen_rooms), stable_hash


# --------- PingPongPenalty ---------
def pingpong_component(
    cfg: RewardConfig,
    room_hash: str | None,
    step_index: int,
    hash_history: deque[str],
) -> tuple[float, deque[str], bool]:
    """
    Если в последних W шагах паттерн A-B-A-B (минимум min_alternations переключений) — штраф.
    Возвращает (reward, history, applied_penalty). v3: pingpong_threshold переопределяет min_alternations.
    """
    min_alt = getattr(cfg, "pingpong_threshold", 0) or cfg.pingpong_min_alternations
    if room_hash is None or cfg.pingpong_penalty >= 0:
        return 0.0, hash_history, False

    history = hash_history
    history.append(room_hash)
    while len(history) > cfg.pingpong_window:
        history.popleft()

    if len(history) < 4:
        return 0.0, history, False

    unique_list: list[str] = []
    for h in history:
        if not unique_list or h != unique_list[-1]:
            unique_list.append(h)
    count_switches = sum(1 for i in range(1, len(unique_list)) if unique_list[i] != unique_list[i - 1])
    if count_switches >= min_alt and len(unique_list) >= 2:
        a, b = unique_list[-2], unique_list[-1]
        if a != b:
            return cfg.pingpong_penalty, history, True
    return 0.0, history, False


# --------- StuckPenalty (v3: progressive + position variance) ---------
@dataclass
class StuckState:
    last_room_change_step: int = 0
    last_room_hash: str | None = None
    recent_x_proxy: deque = field(default_factory=lambda: deque(maxlen=200))


def stuck_component(
    cfg: RewardConfig,
    obs: np.ndarray,
    prev_obs: np.ndarray | None,
    room_hash: str | None,
    step_index: int,
    state: StuckState,
) -> tuple[float, bool, StuckState, int]:
    """
    Если нет смены комнаты N шагов, frame_diff ниже порога и (v3) дисперсия X-позиции низкая ->
    penalty, опционально truncate, stuck_severity 0..2.
    Возвращает (penalty, should_truncate, new_state, severity).
    """
    frame_diff = frame_diff_metric(prev_obs, obs) if prev_obs is not None else 0.0
    if room_hash is not None and room_hash != state.last_room_hash:
        state = StuckState(
            last_room_change_step=step_index,
            last_room_hash=room_hash,
            recent_x_proxy=state.recent_x_proxy,
        )
    steps_since_room = step_index - state.last_room_change_step

    # v3: накапливаем proxy X для дисперсии
    x_proxy = position_proxy_x(obs)
    state.recent_x_proxy.append(x_proxy)
    variance_steps = getattr(cfg, "stuck_position_variance_steps", 60)
    var_thr = getattr(cfg, "stuck_position_variance_threshold", 0.001)
    progressive = getattr(cfg, "stuck_progressive", True)

    if steps_since_room < cfg.stuck_no_room_change_steps or frame_diff >= cfg.stuck_frame_diff_threshold:
        return 0.0, False, state, 0

    # Дисперсия X в последних variance_steps шагах
    if len(state.recent_x_proxy) >= max(2, variance_steps // 2):
        arr = np.array(list(state.recent_x_proxy)[-variance_steps:], dtype=np.float64)
        x_var = float(np.var(arr))
    else:
        x_var = 1.0

    if x_var > var_thr:
        return 0.0, False, state, 0

    severity = min(2, steps_since_room // max(1, cfg.stuck_no_room_change_steps))
    levels = getattr(cfg, "stuck_penalty_severity_levels", None)
    if levels and len(levels) > 0:
        penalty = levels[min(severity, len(levels) - 1)]
    else:
        penalty = cfg.stuck_penalty
        if progressive:
            penalty = cfg.stuck_penalty * (1.0 + 0.5 * severity)
    return penalty, cfg.stuck_truncate, state, severity
