"""
Диагностика для multi-env: room transitions, Stage00 exit, backtrack, after-key.
Использует только obs, HUD, room_hash — без RAM/координат эмулятора.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field


@dataclass
class EpisodeDiagnostics:
    """Диагностика за эпизод: per-env, сбрасывается в reset."""
    room_transition_count: int = 0
    room_hash_buffer: deque = field(default_factory=lambda: deque(maxlen=20))
    backtrack_count: int = 0
    # room_dwell: среднее время (steps) в комнате до перехода
    room_dwell_steps_sum: float = 0.0  # сумма steps с момента последнего перехода
    room_dwell_visits: int = 0
    last_room_transition_step: int = 0
    # door_encounter (best effort: по DoorDetector/событию)
    door_encounter_count: int = 0
    # loop_len: длины последних циклов (A->B->...->A), maxlen=20
    loop_len_buffer: deque = field(default_factory=lambda: deque(maxlen=20))
    # Stage00
    stage00_entered_step: int = 0
    stage00_exit_step: int | None = None
    stage00_exit_success: bool = False
    stage00_room_transitions: int = 0
    in_stage0: bool = True
    # After-key (key_door из HUD)
    key_obtained_at_step: int | None = None
    steps_after_key_total: int | None = None
    steps_after_key_until_exit: int | None = None
    steps_after_key_until_death: int | None = None

    def reset(self) -> None:
        self.room_transition_count = 0
        self.room_hash_buffer.clear()
        self.backtrack_count = 0
        self.stage00_entered_step = 0
        self.stage00_exit_step = None
        self.stage00_exit_success = False
        self.stage00_room_transitions = 0
        self.in_stage0 = True
        self.key_obtained_at_step = None
        self.steps_after_key_total = None
        self.steps_after_key_until_exit = None
        self.steps_after_key_until_death = None
        self.room_dwell_steps_sum = 0.0
        self.room_dwell_visits = 0
        self.last_room_transition_step = 0
        self.door_encounter_count = 0
        self.loop_len_buffer.clear()


def _loop_len_from_buffer(buf: deque) -> int:
    """Длина цикла A->B->...->A по последним room_hash. 0 если нет цикла."""
    if len(buf) < 3:
        return 0
    last = buf[-1]
    for i in range(len(buf) - 2, -1, -1):
        if buf[i] == last:
            return len(buf) - 1 - i
    return 0


def update_diagnostics(
    diag: EpisodeDiagnostics,
    *,
    step: int,
    stable_hash: str | None,
    prev_room_hash: str | None,
    stage: int,
    stage_conf: float,
    key_door: bool,
    prev_key_door: bool,
    terminated_by_death: bool,
    door_detected: bool = False,
) -> dict:
    """
    Обновить диагностику и вернуть extra dict для info.
    stage из parse_stage (HUD), key_door из parse_hud.
    """
    extra: dict = {}
    backtrack_this_step = False

    # Room transition
    if stable_hash is not None and prev_room_hash is not None and stable_hash != prev_room_hash:
        dwell = step - diag.last_room_transition_step
        diag.room_dwell_steps_sum += dwell
        diag.room_dwell_visits += 1
        diag.last_room_transition_step = step
        diag.room_transition_count += 1
        if diag.in_stage0:
            diag.stage00_room_transitions += 1

    # Door encounter (best effort)
    if door_detected:
        diag.door_encounter_count += 1

    # Backtrack: h[i-2]==h[i] and h[i-1]!=h[i]
    if stable_hash is not None:
        buf = diag.room_hash_buffer
        buf.append(stable_hash)
        backtrack_this_step = False
        if len(buf) >= 3:
            a, b, c = buf[-3], buf[-2], buf[-1]
            if a == c and b != c:
                diag.backtrack_count += 1
                backtrack_this_step = True
        loop_len = _loop_len_from_buffer(buf)
        if loop_len > 0:
            diag.loop_len_buffer.append(loop_len)

    # Stage00 exit (stage из HUD, conf >= 0.6)
    if stage_conf >= 0.6:
        if stage == 0:
            if diag.stage00_entered_step == 0 and step == 0:
                diag.stage00_entered_step = 0
            diag.in_stage0 = True
        elif stage >= 1 and diag.in_stage0:
            diag.in_stage0 = False
            if diag.stage00_exit_step is None:
                diag.stage00_exit_step = step
                diag.stage00_exit_success = True

    # After-key
    if key_door and not prev_key_door:
        diag.key_obtained_at_step = step
    if diag.key_obtained_at_step is not None:
        diag.steps_after_key_total = step - diag.key_obtained_at_step
        if diag.stage00_exit_step is not None and diag.key_obtained_at_step <= diag.stage00_exit_step:
            diag.steps_after_key_until_exit = diag.stage00_exit_step - diag.key_obtained_at_step
        if terminated_by_death:
            diag.steps_after_key_until_death = step - diag.key_obtained_at_step

    extra["room_transition_count"] = diag.room_transition_count
    extra["backtrack_count"] = diag.backtrack_count
    extra["backtrack_this_step"] = backtrack_this_step
    extra["stage00_exit_success"] = 1 if diag.stage00_exit_success else 0
    extra["stage00_exit_steps"] = (
        (diag.stage00_exit_step - diag.stage00_entered_step)
        if diag.stage00_exit_step is not None else -1
    )
    extra["stage00_room_transitions"] = diag.stage00_room_transitions
    extra["steps_after_key_total"] = diag.steps_after_key_total if diag.steps_after_key_total is not None else -1
    extra["steps_after_key_until_exit"] = diag.steps_after_key_until_exit if diag.steps_after_key_until_exit is not None else -1
    extra["steps_after_key_until_death"] = diag.steps_after_key_until_death if diag.steps_after_key_until_death is not None else -1
    extra["room_transition_event"] = (
        stable_hash is not None and prev_room_hash is not None and stable_hash != prev_room_hash
    )
    extra["prev_room_hash"] = prev_room_hash  # для debug room change
    extra["room_dwell_steps"] = (
        diag.room_dwell_steps_sum / diag.room_dwell_visits if diag.room_dwell_visits > 0 else -1.0
    )
    extra["door_encounter_count"] = diag.door_encounter_count
    extra["loop_len_max"] = max(diag.loop_len_buffer, default=0)

    return extra
