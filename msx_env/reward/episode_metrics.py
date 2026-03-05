"""
Эпизодные метрики (episode-metrics-fix, fix-room-metrics-stability): корректный подсчёт комнат, stage00 exit, backtrack.
Использует stable_room_id = (stable_stage_id, stable_room_hash) — устойчив к спрайтам/анимации.
Старые метрики не меняем — только новые эпизодные (unique_rooms_ep_*, backtrack_rate_ep_*, и т.д.).
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field

from msx_env.reward.hashers import stable_room_hash_playfield

logger = logging.getLogger(__name__)


def _make_stable_room_id(stage: int, room_hash: str) -> str:
    """stable_room_id = (stage, hash) для уникальности по этапу и комнате."""
    return f"S{stage}_{room_hash}"


@dataclass
class EpisodeRoomTracker:
    """
    Per-env трекер эпизодных метрик.
    Сброс при reset/end эпизода.
    Использует stable_room_id = (stage, hash) для эпизодных метрик.
    """
    # Debounce (D)
    candidate_room_hash: str | None = None
    candidate_count: int = 0
    stable_room_hash: str | None = None

    # stable_room_id = (stage, hash) для episode metrics
    stable_stage: int = 0
    stable_room_id: str | None = None
    prev_stable_room_id: str | None = None

    # A) Room episode metrics (по stable_room_id)
    episode_room_set: set = field(default_factory=set)
    episode_transition_count: int = 0
    episode_room_history: deque = field(default_factory=lambda: deque(maxlen=20))
    episode_room_dwell_steps: int = 0
    episode_dwell_accum: list = field(default_factory=list)

    # B) Stage00 exit (00→01 как событие, не привязка к done)
    stage00_exit_recorded: bool = False
    stage00_exit_steps: int = -1
    stage_prev: int = 0
    stage_candidate: int = 0
    stage_candidate_frames: int = 0

    # C) Backtrack episode (по stable_room_id)
    backtrack_count_ep: int = 0

    # Debug: последние stable_room_id для sanity warning
    _last_stable_room_ids: deque = field(default_factory=lambda: deque(maxlen=10))
    _last_raw_hashes: deque = field(default_factory=lambda: deque(maxlen=10))

    def reset(self) -> None:
        self.candidate_room_hash = None
        self.candidate_count = 0
        self.stable_room_hash = None
        self.stable_stage = 0
        self.stable_room_id = None
        self.prev_stable_room_id = None
        self.episode_room_set.clear()
        self.episode_transition_count = 0
        self.episode_room_history.clear()
        self.episode_room_dwell_steps = 0
        self.episode_dwell_accum.clear()
        self.stage00_exit_recorded = False
        self.stage00_exit_steps = -1
        self.stage_prev = 0
        self.stage_candidate = 0
        self.stage_candidate_frames = 0
        self.backtrack_count_ep = 0
        self._last_stable_room_ids.clear()
        self._last_raw_hashes.clear()


def update_episode_room_metrics(
    tracker: EpisodeRoomTracker,
    *,
    obs,
    stage: int,
    stage_conf: float,
    step: int,
    room_debounce_k: int = 7,
    stage_stable_frames: int = 5,
    crop_top: int = 20,
    crop_bottom: int = 4,
    crop_right: int = 36,
    sanity_unique_rooms_warn_threshold: int = 20,
) -> dict:
    """
    Обновить эпизодные метрики. raw_hash вычисляется через stable_room_hash_playfield.
    stable_room_id = (stable_stage, stable_room_hash) — для unique_rooms_ep и т.д.
    """
    extra: dict = {}

    # Raw hash: playfield-only, blur+quantize (устойчивость к спрайтам)
    raw_hash: str | None = None
    try:
        raw_hash = stable_room_hash_playfield(
            obs,
            crop_top=crop_top,
            crop_bottom=crop_bottom,
            crop_right=crop_right,
        )
    except Exception:
        raw_hash = None

    if raw_hash is not None:
        tracker._last_raw_hashes.append(raw_hash)

    # ---- Room hash debounce ----
    if raw_hash is None:
        pass
    elif raw_hash == tracker.stable_room_hash:
        tracker.candidate_room_hash = None
        tracker.candidate_count = 0
    else:
        if raw_hash == tracker.candidate_room_hash:
            tracker.candidate_count += 1
            if tracker.candidate_count >= room_debounce_k:
                tracker.stable_room_hash = raw_hash
                tracker.candidate_room_hash = None
                tracker.candidate_count = 0
        else:
            tracker.candidate_room_hash = raw_hash
            tracker.candidate_count = 1

    stable_hash = tracker.stable_room_hash

    # ---- Stage stability and 00→01 exit ----
    if stage_conf >= 0.6:
        if stage != tracker.stage_candidate:
            tracker.stage_candidate = stage
            tracker.stage_candidate_frames = 0
        tracker.stage_candidate_frames += 1
        if tracker.stage_candidate_frames >= stage_stable_frames:
            accepted_stage = tracker.stage_candidate
            old_stage = tracker.stage_prev
            if old_stage == 0 and accepted_stage == 1 and not tracker.stage00_exit_recorded:
                tracker.stage00_exit_recorded = True
                tracker.stage00_exit_steps = step
            tracker.stage_prev = accepted_stage
            tracker.stable_stage = accepted_stage

    # ---- stable_room_id = (stage, hash) для episode metrics ----
    if stable_hash is not None:
        srid = _make_stable_room_id(tracker.stable_stage, stable_hash)
        tracker.stable_room_id = srid
        tracker._last_stable_room_ids.append(srid)
    else:
        srid = None

    # ---- A) Room transition, dwell, unique set (по stable_room_id) ----
    if srid is not None:
        tracker.episode_room_set.add(srid)

        is_first_room = tracker.prev_stable_room_id is None
        if is_first_room:
            tracker.episode_room_history.append(srid)
        elif srid != tracker.prev_stable_room_id:
            tracker.episode_transition_count += 1
            tracker.episode_room_history.append(srid)
            tracker.episode_dwell_accum.append(tracker.episode_room_dwell_steps)
            tracker.episode_room_dwell_steps = 0

            # ---- C) Backtrack A-B-A (по stable_room_id) ----
            if len(tracker.episode_room_history) >= 3:
                a, b, c = (
                    tracker.episode_room_history[-3],
                    tracker.episode_room_history[-2],
                    tracker.episode_room_history[-1],
                )
                if a == c and b != c:
                    tracker.backtrack_count_ep += 1
        else:
            tracker.episode_room_dwell_steps += 1

        tracker.prev_stable_room_id = srid
    else:
        if tracker.stable_room_id is not None:
            tracker.episode_room_dwell_steps += 1

    # ---- Extra output ----
    unique_count = len(tracker.episode_room_set)
    extra["unique_rooms_ep"] = unique_count
    extra["room_transitions_ep"] = tracker.episode_transition_count
    extra["room_dwell_steps_ep_mean"] = (
        sum(tracker.episode_dwell_accum) / len(tracker.episode_dwell_accum)
        if tracker.episode_dwell_accum else -1.0
    )
    extra["stage00_exit_recorded_ep"] = 1 if tracker.stage00_exit_recorded else 0
    extra["stage00_exit_steps_ep"] = tracker.stage00_exit_steps
    extra["backtrack_count_ep"] = tracker.backtrack_count_ep
    extra["backtrack_rate_ep"] = (
        tracker.backtrack_count_ep / max(1, tracker.episode_transition_count)
        if tracker.episode_transition_count > 0 else 0.0
    )
    extra["stable_room_hash_ep"] = stable_hash
    extra["stable_stage_ep"] = tracker.stable_stage
    extra["stable_room_id_ep"] = srid
    extra["stable_room_ids_ep"] = list(tracker.episode_room_history)[:10]  # first 10 for acceptance test

    # ---- Sanity warning ----
    if unique_count > sanity_unique_rooms_warn_threshold:
        last_srids = list(tracker._last_stable_room_ids)[-10:]
        last_raw = list(tracker._last_raw_hashes)[-10:]
        logger.warning(
            "[fix-room-metrics-stability] unique_rooms_ep=%d > %d (possible room hash jitter). "
            "Last 10 stable_room_id: %s | raw_hashes: %s",
            unique_count, sanity_unique_rooms_warn_threshold,
            [s[:24] + "..." for s in last_srids] if last_srids else [],
            [r[:12] + "..." for r in last_raw] if last_raw else [],
        )
        extra["room_metrics_sanity_warning"] = True

    return extra


def raw_room_hash(obs, crop_top: int = 14, crop_bottom: int = 0) -> str | None:
    """
    Raw hash для обратной совместимости (если где-то вызывается напрямую).
    Для эпизодных метрик используется stable_room_hash_playfield внутри update_episode_room_metrics.
    """
    from msx_env.reward.hashers import block_mean_hash
    if obs is None or obs.size == 0:
        return None
    try:
        return block_mean_hash(
            obs,
            crop_rows_top=crop_top,
            crop_rows_bottom=crop_bottom,
        )
    except Exception:
        return None
