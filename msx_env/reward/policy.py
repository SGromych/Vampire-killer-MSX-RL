"""
Политика наград: один вызов compute(prev_obs, obs, info, ...) -> (reward, components).
Детерминирована при фиксированных входах.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

import logging
import numpy as np

from msx_env.hud_parser import HudState, parse_hud, parse_stage
from msx_env.reward.components import (
    NoveltyState,
    PickupState,
    StuckState,
    death_penalty_component,
    pickup_component,
    novelty_component,
    pingpong_component,
    step_penalty_component,
    stuck_component,
)
from msx_env.reward.config import RewardConfig
from msx_env.reward.diagnostics import EpisodeDiagnostics, update_diagnostics
from msx_env.reward.episode_metrics import EpisodeRoomTracker, update_episode_room_metrics
from msx_env.reward.event_detectors import KeyDetector, DoorDetector
from msx_env.reward.hashers import position_proxy_x, position_proxy_y


logger = logging.getLogger(__name__)


@dataclass
class RewardPolicyState:
    """Состояние политики на протяжении эпизода."""
    step_index: int = 0
    prev_obs: np.ndarray | None = None
    prev_hud: HudState | None = None
    prev_room_hash: str | None = None
    pickup_state: PickupState = field(default_factory=PickupState)
    novelty_state: NoveltyState = field(default_factory=NoveltyState)
    stuck_state: StuckState = field(default_factory=StuckState)
    hash_history: deque = field(default_factory=lambda: deque(maxlen=50))
    pingpong_count_episode: int = 0
    key_rewarded_this_episode: bool = False
    door_rewarded_this_episode: bool = False
    # Stage: принятый stage (после гистерезиса), счётчик шагов в нём, стабильность для смены
    accepted_stage: int = 0
    steps_in_current_stage: int = 0
    stage_candidate: int = 0
    stage_candidate_frames: int = 0
    diagnostics: EpisodeDiagnostics = field(default_factory=EpisodeDiagnostics)
    episode_room_tracker: EpisodeRoomTracker = field(default_factory=EpisodeRoomTracker)
    # Novelty after key: track cumulative sums for diagnostics
    novelty_reward_pre_key_sum: float = 0.0
    novelty_reward_post_key_sum: float = 0.0
    after_key: bool = False
    # Position novelty: set of (cell_x, cell_y) visited this episode
    visited_position_cells: set = field(default_factory=set)


class RewardPolicy:
    """
    Модульная политика наград. compute() возвращает суммарную награду и разбивку по компонентам.
    v3: novelty_rate, pingpong_count, key/door (раз за эпизод), progressive stuck, диагностика доминирования.
    """

    def __init__(self, cfg: RewardConfig | None = None):
        self.cfg = cfg or RewardConfig()
        self._state = RewardPolicyState()
        self._key_detector = KeyDetector()
        self._door_detector = DoorDetector()
        self._door_distance_warned: bool = False
        self._block_break_warned: bool = False

    def reset(self) -> None:
        self._state = RewardPolicyState()

    def compute(
        self,
        prev_obs: np.ndarray | None,
        obs: np.ndarray,
        info: dict[str, Any],
        terminated: bool,
        truncated: bool,
        terminated_by_death: bool,
        rgb_for_hud: np.ndarray | None = None,
    ) -> tuple[float, dict[str, float], dict[str, Any]]:
        """
        Возвращает (total_reward, components_dict, extra).
        extra может содержать stuck_truncate=True, unique_rooms и т.д.
        """
        cfg = self.cfg
        state = self._state
        step = state.step_index

        components: dict[str, float] = {
            "step": 0.0,
            "pickup": 0.0,
            "death": 0.0,
            "novelty": 0.0,
            "position_novelty": 0.0,
            "pingpong": 0.0,
            "stuck": 0.0,
            "progress": 0.0,
            "key": 0.0,
            "door": 0.0,
            "stage_step": 0.0,
            "stage_advance": 0.0,
            "backtrack": 0.0,
        }
        extra: dict[str, Any] = {}
        info_for_detectors = dict(info)
        info_for_detectors["reward_room_hash"] = None
        info_for_detectors["reward_prev_room_hash"] = state.prev_room_hash

        # 1) Step penalty
        components["step"] = step_penalty_component(cfg)

        # 2) Death (уже учтён в env как terminated; здесь только величина для лога)
        out = death_penalty_component(cfg, terminated_by_death)
        components["death"] = out.reward

        # 3) Pickup (HUD)
        curr_hud = None
        if rgb_for_hud is not None:
            try:
                curr_hud = parse_hud(rgb_for_hud)
            except Exception:
                pass
        pickup_r, new_pickup_state = pickup_component(
            cfg, state.prev_hud, curr_hud, step, state.pickup_state
        )
        components["pickup"] = pickup_r
        state.pickup_state = new_pickup_state
        state.prev_hud = curr_hud

        # 4) Novelty (с возможным уменьшением после получения ключа)
        novelty_r, state.novelty_state, unique_count, stable_hash = novelty_component(
            cfg, obs, step, state.novelty_state
        )
        novelty_mult = 1.0
        after_key_mult = getattr(cfg, "novelty_after_key_multiplier", 1.0)
        if state.after_key and after_key_mult > 0.0 and after_key_mult < 1.0:
            novelty_mult = after_key_mult
        scaled_novelty = novelty_r * novelty_mult
        components["novelty"] = scaled_novelty
        extra["unique_rooms"] = unique_count
        extra["room_hash"] = stable_hash
        extra["novelty_rate"] = unique_count / max(1, step) if step else 0.0
        # Track pre/post key novelty sums for diagnostics
        if state.after_key:
            state.novelty_reward_post_key_sum += scaled_novelty
        else:
            state.novelty_reward_pre_key_sum += scaled_novelty
        extra["novelty_reward_pre_key_sum"] = state.novelty_reward_pre_key_sum
        extra["novelty_reward_post_key_sum"] = state.novelty_reward_post_key_sum
        extra["novelty_after_key_multiplier_used"] = novelty_mult

        info_for_detectors["reward_room_hash"] = stable_hash

        # 4b) Position novelty: reward for visiting new (quantized) grid cells
        if getattr(cfg, "enable_position_novelty", False) and obs is not None:
            quantize = max(1, getattr(cfg, "position_novelty_quantize", 8))
            reward_val = getattr(cfg, "position_novelty_reward", 0.005) or 0.0
            if reward_val > 0:
                h, w = obs.shape[0], obs.shape[1]
                x_norm = position_proxy_x(obs)
                y_norm = position_proxy_y(obs)
                x_px = x_norm * max(0, w - 1)
                y_px = y_norm * max(0, h - 1)
                cell = (int(x_px) // quantize, int(y_px) // quantize)
                if cell not in state.visited_position_cells:
                    components["position_novelty"] = reward_val
                    state.visited_position_cells.add(cell)
                extra["position_novelty_cells_ep"] = len(state.visited_position_cells)

        # 5) Ping-pong
        pingpong_r, state.hash_history, pingpong_applied = pingpong_component(
            cfg, stable_hash, step, state.hash_history
        )
        components["pingpong"] = pingpong_r
        if pingpong_applied:
            state.pingpong_count_episode += 1
            extra["pingpong_event"] = True
        extra["pingpong_count"] = state.pingpong_count_episode

        # 6) Stuck (v3: progressive + position variance, severity)
        stuck_r, stuck_truncate, state.stuck_state, stuck_severity = stuck_component(
            cfg, obs, state.prev_obs, stable_hash, step, state.stuck_state
        )
        components["stuck"] = stuck_r
        if stuck_r < 0:
            extra["stuck_event"] = True
            extra["stuck_truncate"] = stuck_truncate
            extra["stuck_severity"] = stuck_severity

        # 7) Stage (time pressure в STAGE 00 + бонус за переход)
        raw_stage = int(info.get("stage", 0))
        stage_conf = float(info.get("stage_conf", 0.0))
        enable_stage = getattr(cfg, "enable_stage_reward", False)
        stage_only_list = getattr(cfg, "stage_only_for", None)
        stage_only_list = [0] if stage_only_list is None else stage_only_list  # None = только STAGE 00
        conf_thr = getattr(cfg, "stage_conf_threshold", 0.6)
        stability_k = max(0, getattr(cfg, "stage_stability_frames", 3))
        step_pen = getattr(cfg, "stage_step_penalty", -0.002)
        advance_bonus = getattr(cfg, "stage_advance_bonus", 2.0)

        if enable_stage and stage_conf >= conf_thr:
            if raw_stage != state.stage_candidate:
                state.stage_candidate = raw_stage
                state.stage_candidate_frames = 0
            state.stage_candidate_frames += 1
            if state.stage_candidate_frames >= stability_k:
                new_stage = state.stage_candidate
                if new_stage > state.accepted_stage:
                    components["stage_advance"] = advance_bonus
                    extra["stage_advance_event"] = True
                    state.steps_in_current_stage = 0
                state.accepted_stage = new_stage
            state.steps_in_current_stage += 1
            apply_step_penalty = (
                (not stage_only_list and state.accepted_stage == 0)
                or (stage_only_list and state.accepted_stage in stage_only_list)
            )
            if apply_step_penalty:
                components["stage_step"] = step_pen
            extra["stage"] = state.accepted_stage
            extra["stage_conf"] = stage_conf
        else:
            if enable_stage and stage_conf < conf_thr and raw_stage != 0:
                extra["stage_low_conf_count"] = getattr(state, "_stage_low_conf_count", 0) + 1
                state._stage_low_conf_count = extra["stage_low_conf_count"]

        # 8) Key/Door (v3: раз за эпизод, по детекторам; door_detected — для диагностики)
        key_reward_val = getattr(cfg, "key_reward", 0.0) or 0.0
        door_reward_val = getattr(cfg, "door_reward", 0.0) or 0.0
        door_detected = False
        if rgb_for_hud is not None:
            key_ev = self._key_detector.detect(obs, info_for_detectors, rgb_for_hud)
            door_ev = self._door_detector.detect(obs, info_for_detectors, rgb_for_hud)
            door_detected = door_ev.get("door_detected", False)
            extra["key_detected"] = key_ev.get("key_detected", False)
            extra["door_detected"] = door_detected
            extra["detection_confidence"] = max(
                key_ev.get("detection_confidence", 0),
                door_ev.get("detection_confidence", 0),
            )
            if key_reward_val > 0 and key_ev.get("key_detected") and not state.key_rewarded_this_episode:
                components["key"] = key_reward_val
                state.key_rewarded_this_episode = True
            if door_reward_val > 0 and door_ev.get("door_detected") and not state.door_rewarded_this_episode:
                components["door"] = door_reward_val
                state.door_rewarded_this_episode = True

        # After-key flag: как только ключ получен (по HUD или детектору), уменьшаем novelty на последующих шагах
        if not state.after_key:
            has_key_hud = False
            if state.prev_hud is not None:
                has_key_hud = bool(state.prev_hud.key_door or state.prev_hud.key_chest)
            if has_key_hud or state.key_rewarded_this_episode or extra.get("key_detected", False):
                state.after_key = True

        # Диагностика: room transitions, backtrack, Stage00 exit, after-key
        stage_int, stage_conf_hud = 0, 0.0
        if rgb_for_hud is not None:
            try:
                stage_int, stage_conf_hud = parse_stage(rgb_for_hud)
            except Exception:
                pass
        prev_key = state.prev_hud.key_door if state.prev_hud else False
        curr_key = curr_hud.key_door if curr_hud else False
        diag_extra = update_diagnostics(
            state.diagnostics,
            step=step,
            stable_hash=stable_hash,
            prev_room_hash=state.prev_room_hash,
            stage=stage_int,
            stage_conf=stage_conf_hud,
            key_door=curr_key,
            prev_key_door=prev_key,
            terminated_by_death=terminated_by_death,
            door_detected=door_detected,
        )
        for k, v in diag_extra.items():
            extra[k] = v

        # Episode metrics (episode-metrics-fix, fix-room-metrics-stability): stable_room_id, unique_rooms_ep, stage00_exit_ep, backtrack_rate_ep
        room_debounce_k = getattr(cfg, "episode_room_debounce_k", 7)
        stage_stable_k = getattr(cfg, "episode_stage_stable_frames", 5)
        crop_top_ep = getattr(cfg, "episode_playfield_crop_top", 20)
        crop_bottom_ep = getattr(cfg, "episode_playfield_crop_bottom", 4)
        crop_right_ep = getattr(cfg, "episode_playfield_crop_right", 36)
        ep_extra = update_episode_room_metrics(
            state.episode_room_tracker,
            obs=obs,
            stage=stage_int,
            stage_conf=stage_conf_hud,
            step=step,
            room_debounce_k=room_debounce_k,
            stage_stable_frames=stage_stable_k,
            crop_top=crop_top_ep,
            crop_bottom=crop_bottom_ep,
            crop_right=crop_right_ep,
            sanity_unique_rooms_warn_threshold=getattr(cfg, "episode_unique_rooms_sanity_warn", 20),
        )
        for k, v in ep_extra.items():
            extra[k] = v

        # 9) Backtrack penalty (optional, no cheating)
        backtrack_penalty_enabled = getattr(cfg, "backtrack_penalty_enabled", False)
        backtrack_penalty_value = getattr(cfg, "backtrack_penalty_value", -0.01)
        if backtrack_penalty_enabled and diag_extra.get("backtrack_this_step", False):
            components["backtrack"] = backtrack_penalty_value
        elif "backtrack" not in components:
            components["backtrack"] = 0.0

        state.prev_obs = obs.copy() if obs is not None else None
        state.prev_room_hash = stable_hash
        state.step_index = step + 1

        # Door distance and block break rewards are not implemented yet (нет надёжного сигнала).
        # Если соответствующие флаги включены в конфиг — вывести предупреждение один раз и не применять shaping.
        if getattr(cfg, "enable_door_distance_reward", False) and not self._door_distance_warned:
            logger.warning(
                "enable_door_distance_reward=True, но детектор расстояния до двери пока не реализован; "
                "door distance shaping не применяется."
            )
            self._door_distance_warned = True
        if getattr(cfg, "enable_block_break_reward", False) and not self._block_break_warned:
            logger.warning(
                "enable_block_break_reward=True, но детектор разрушения блоков пока не реализован; "
                "block-break reward не применяется."
            )
            self._block_break_warned = True

        total = sum(components.values())
        # v3: диагностика доминирования
        if total != 0:
            by_val = [(k, v) for k, v in components.items() if v != 0]
            if by_val:
                dominant_name = max(by_val, key=lambda x: abs(x[1]))[0]
                dominant_val = components[dominant_name]
                dominance_ratio = abs(dominant_val) / abs(total)
                extra["reward_dominance_ratio"] = dominance_ratio
                extra["reward_dominant_component"] = dominant_name
                if dominant_name == "death" and dominance_ratio > 0.5:
                    extra["death_dominance_warning"] = True
        return total, components, extra
