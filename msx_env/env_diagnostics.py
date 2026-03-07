"""
Диагностика окружения: реестр ресурсов multi-env и вывод блоков ENV RESET / TERMINATION только при debug.

Использование: env.py и train_ppo.py вызывают функции здесь только когда debug=True (или при старте — clear).
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List

from msx_env.hud_parser import parse_stage
from msx_env.reward.hashers import block_mean_hash


# Реестр ресурсов для проверки коллизий при num_envs > 1 (debug).
# Каждая запись: instance_id, workdir, screenshot_path, capture_backend, window_rect, crop_rect, pid, capture_identity.
ENV_RESOURCE_REGISTRY: List[Dict[str, Any]] = []


def clear_env_resource_registry() -> None:
    """Очистить реестр (вызывать в начале запуска обучения)."""
    global ENV_RESOURCE_REGISTRY
    ENV_RESOURCE_REGISTRY = []


def register_and_assert_resources(env: Any) -> None:
    """
    Зарегистрировать ресурсы env и проверить отсутствие коллизий с уже зарегистрированными.
    Вызывать только при debug. При коллизии — RuntimeError.
    """
    cfg = getattr(env, "cfg", None)
    if cfg is None:
        return
    workdir = getattr(cfg, "workdir", "")
    instance_id = getattr(cfg, "instance_id", None)
    backend = getattr(cfg, "capture_backend", "png")
    screenshot_path = str(Path(workdir) / "step_frame.png")
    capture = getattr(env, "_capture", None)
    if backend == "window" and capture is not None:
        region = getattr(capture, "_region", None)
        crop = getattr(capture, "_crop_rect", None)
        capture_identity = ("window", tuple(region) if region else None, tuple(crop) if crop else None)
    elif backend == "dxcam" and capture is not None:
        region = getattr(capture, "_region", None)
        capture_identity = ("dxcam", tuple(region) if region else None, None)
    else:
        capture_identity = ("file", screenshot_path, None)
    pid = None
    emu = getattr(env, "_emu", None)
    if emu is not None and getattr(emu, "proc", None) is not None:
        pid = getattr(emu.proc, "pid", None)
    entry = {
        "instance_id": instance_id,
        "workdir": workdir,
        "screenshot_path": screenshot_path,
        "capture_backend": backend,
        "window_rect": getattr(capture, "_region", None) if capture else None,
        "crop_rect": getattr(capture, "_crop_rect", None) if capture else None,
        "pid": pid,
        "capture_identity": capture_identity,
    }
    for existing in ENV_RESOURCE_REGISTRY:
        if existing.get("instance_id") == entry["instance_id"]:
            continue
        if existing["workdir"] == entry["workdir"]:
            raise RuntimeError(
                f"Multi-env collision: workdir not unique. "
                f"env instance_id={entry['instance_id']} and instance_id={existing['instance_id']} "
                f"share workdir={entry['workdir']}"
            )
        if existing["screenshot_path"] == entry["screenshot_path"]:
            raise RuntimeError(
                f"Multi-env collision: screenshot_path not unique. "
                f"env instance_id={entry['instance_id']} and instance_id={existing['instance_id']} "
                f"share path={entry['screenshot_path']}"
            )
        if backend == "window" and existing.get("capture_backend") == "window":
            if existing.get("window_rect") and entry["window_rect"] and existing["window_rect"] == entry["window_rect"]:
                raise RuntimeError(
                    f"Multi-env collision: window capture region not unique. "
                    f"env instance_id={entry['instance_id']} and instance_id={existing['instance_id']} "
                    f"share window_rect={entry['window_rect']}. "
                    "Use per-env window_rects_json or switch to file capture for num_envs>1."
                )
    if not any(e.get("instance_id") == entry["instance_id"] for e in ENV_RESOURCE_REGISTRY):
        ENV_RESOURCE_REGISTRY.append(entry)


def print_reset_block(env: Any, obs: Any, rgb: Any) -> None:
    """Вывести блок ENV RESET (workdir, pid, stage, room_hash, frame_hash). Вызывать только при debug."""
    cfg = getattr(env, "cfg", None)
    if cfg is None:
        return
    try:
        initial_stage, initial_stage_conf = parse_stage(rgb)
    except Exception:
        initial_stage, initial_stage_conf = -1, 0.0
    initial_room_hash = block_mean_hash(obs)
    room_short = (initial_room_hash or "")[:12] if initial_room_hash else "—"
    initial_frame_hash = hashlib.md5(obs.tobytes()).hexdigest()[:16]
    capture = getattr(env, "_capture", None)
    window_rect = getattr(capture, "_region", None) if capture else None
    control_channel = getattr(cfg, "workdir", "?")
    capture_backend = getattr(cfg, "capture_backend", "png")
    screenshot_path = str(Path(getattr(cfg, "workdir", "")) / "step_frame.png")
    openmsx_pid = None
    emu = getattr(env, "_emu", None)
    if emu is not None and getattr(emu, "proc", None):
        openmsx_pid = getattr(emu.proc, "pid", None)
    rank = getattr(cfg, "instance_id", "?")
    print("[ENV RESET] env_id=%s pid=%s channel=%s capture=%s" % (rank, openmsx_pid, control_channel, capture_backend))
    print("  workdir=%s" % getattr(cfg, "workdir", "?"))
    if capture_backend in ("png", "single"):
        print("  screenshot_path=%s" % screenshot_path)
    if capture_backend == "window" and window_rect is not None:
        print("  window_rect=%s" % (window_rect,))
    if capture_backend == "dxcam" and window_rect is not None:
        print("  dxcam_region=%s" % (window_rect,))
    print("  stage=%s stage_conf=%s" % (initial_stage, initial_stage_conf))
    print("  room_hash=%s frame_hash=%s" % (room_short, initial_frame_hash))


def print_step_control(env: Any, emu: Any) -> None:
    """Вывести строку о канале управления (workdir, pid, commands_tcl). Вызывать при debug на первых шагах."""
    cfg = getattr(env, "cfg", None)
    if cfg is None:
        return
    pid = getattr(emu.proc, "pid", None) if getattr(emu, "proc", None) else None
    wd = getattr(emu, "workdir", None)
    ct = getattr(emu, "commands_tcl", None)
    print("[debug] env_id=%s sending action to workdir=%s pid=%s commands_tcl=%s" % (
        getattr(cfg, "instance_id", "?"),
        str(wd) if wd is not None else "?",
        pid,
        str(ct) if ct is not None else "?",
    ))


def print_termination_block(
    env: Any,
    step_count: int,
    action_name: str,
    terminated: bool,
    truncated: bool,
    info: Dict[str, Any],
    obs: Any,
) -> None:
    """Вывести блок TERMINATION (reason, death_detector, raw_death_signals, stage, room_hash, last_action). При debug."""
    cfg = getattr(env, "cfg", None)
    if cfg is None:
        return
    reason = info.get("termination_reason", "unknown")
    room_hash = info.get("reward_room_hash") or ""
    room_short = room_hash[:12] if room_hash else "—"
    frame_hash = hashlib.md5(obs.tobytes()).hexdigest()[:16]
    rank = getattr(cfg, "instance_id", "?")
    print("[TERMINATION] env_id=%s step_in_episode=%s terminated=%s truncated=%s" % (rank, step_count, terminated, truncated))
    print("  reason=%s" % reason)
    print("  death_detector=%s" % info.get("death_detector", "?"))
    print("  raw_death_signals=%s" % info.get("raw_death_signals", {}))
    print("  hp=%s lives=%s gameover=%s" % (info.get("hp_value"), info.get("lives_value"), info.get("gameover_flag")))
    print("  stage=%s stage_conf=%s" % (info.get("stage"), info.get("stage_conf")))
    print("  room_hash=%s frame_hash=%s" % (room_short, frame_hash))
    print("  last_action=%s hold_ms=%s repeat=%s" % (action_name, "keydown_hold", getattr(cfg, "action_repeat", 2)))
