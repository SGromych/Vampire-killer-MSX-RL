"""
Single source of truth for RL_msx configuration.

- load_config(argv): parse CLI, load optional config file(s), merge (defaults -> file -> CLI),
  resolve all paths to absolute, validate, return ResolvedConfig.
- RunLayout: canonical paths under run_dir (train.log, metrics.csv, config_snapshot.json,
  resolved_paths.json, checkpoints/).
- All entrypoints (train_ppo, train_supervisor, make_env, env, reward, capture) consume
  the same resolved config object.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Schemas (dataclass-based; no pydantic)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RunConfig:
    """Run directory, experiment name, timestamps, seed, device."""
    run_dir: Path
    experiment_name: str
    seed: int
    device: str
    use_runs_dir: bool = False  # whether run_dir was created via runs/<ts>_<git>_<name>/


@dataclass(frozen=True)
class PPOConfig:
    """PPO hyperparameters and training control."""
    checkpoint_dir: Path  # resolved absolute; may be under run_dir
    bc_checkpoint: Path | None
    epochs: int
    rollout_steps: int
    ppo_epochs: int
    batch_size: int
    lr: float
    gamma: float
    gae_lambda: float
    clip_eps: float
    entropy_coef: float
    value_loss_coef: float
    max_episode_steps: int
    step_penalty: float
    arch: str
    entropy_floor: float
    stuck_updates: int
    resume: bool
    checkpoint_every: int
    recurrent: bool
    lstm_hidden_size: int
    sequence_length: int
    nudge_right_steps: int
    stuck_nudge_steps: int
    no_reset_handshake: bool
    dry_run_seconds: float
    summary_every: int
    summary_interval_sec: float
    export_metrics: Path | None


@dataclass(frozen=True)
class EnvConfigSchema:
    """Env-related options (used to build msx_env.env.EnvConfig)."""
    rom_path: Path
    frame_size: tuple[int, int]
    action_repeat: int
    decision_fps: float | None
    capture_backend: str
    post_action_delay_ms: float
    max_episode_steps: int
    tmp_root: str
    soft_reset: bool
    num_envs: int
    window_title: str | None
    window_rects_path: Path | None
    reset_handshake_stable_frames: int
    reset_handshake_conf_min: float
    reset_handshake_timeout_s: float
    debug: bool
    debug_every: int
    debug_episode_max_steps: int
    debug_dump_frames: int
    debug_force_action: str
    debug_room_change: bool
    ignore_death: bool
    dump_hud_every_n_steps: int
    perf_profile: bool
    quiet: bool
    openmsx_log_dir: Path | None


@dataclass(frozen=True)
class CaptureConfig:
    """Capture backend selection and fallback policy."""
    backend: str  # png | single | window
    fallback_policy: str  # "strict" -> fail if backend fails; "fallback" -> try file
    window_crop: tuple[int, int, int, int] | None
    window_title: str | None
    capture_lag_ms: float
    log_diagnostics: bool


@dataclass(frozen=True)
class LoggingConfig:
    """Filenames and paths for logs and metrics (all under run_dir)."""
    train_log: str = "train.log"
    supervisor_log: str = "supervisor.log"
    metrics_csv: str = "metrics.csv"
    config_snapshot: str = "config_snapshot.json"
    resolved_paths: str = "resolved_paths.json"
    metrics_schema: str = "metrics_schema.json"
    checkpoint_subdir: str = "checkpoints/ppo"


@dataclass
class ResolvedConfig:
    """Fully resolved configuration; single object passed to all components."""
    run: RunConfig
    ppo: PPOConfig
    env_schema: EnvConfigSchema
    reward_config: Any  # RewardConfig from msx_env.reward.config
    reward_config_path: Path | None  # path used to load (or None if default)
    capture: CaptureConfig
    logging: LoggingConfig
    # RunLayout: derived from run.run_dir + logging
    layout: "RunLayout"


# ---------------------------------------------------------------------------
# RunLayout
# ---------------------------------------------------------------------------


class RunLayout:
    """Canonical paths under run_dir. All writers use this; no writes outside run_dir unless explicit."""

    def __init__(self, run_dir: Path, log_cfg: LoggingConfig):
        self.run_dir = Path(run_dir).resolve()
        self._log = log_cfg

    def train_log(self) -> Path:
        return self.run_dir / self._log.train_log

    def supervisor_log(self) -> Path:
        return self.run_dir / self._log.supervisor_log

    def metrics_csv(self) -> Path:
        return self.run_dir / self._log.metrics_csv

    def config_snapshot(self) -> Path:
        return self.run_dir / self._log.config_snapshot

    def resolved_paths_file(self) -> Path:
        return self.run_dir / self._log.resolved_paths

    def metrics_schema(self) -> Path:
        return self.run_dir / self._log.metrics_schema

    def checkpoint_dir(self) -> Path:
        return self.run_dir / self._log.checkpoint_subdir

    def ensure_run_dir(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

def _default_ppo() -> dict:
    return {
        "checkpoint_dir": "checkpoints/ppo",
        "bc_checkpoint": None,
        "epochs": 100,
        "rollout_steps": 128,
        "ppo_epochs": 3,
        "batch_size": 64,
        "lr": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_eps": 0.2,
        "entropy_coef": 0.01,
        "value_loss_coef": 0.5,
        "max_episode_steps": 1500,
        "step_penalty": -0.001,
        "arch": "deep",
        "entropy_floor": 0.3,
        "stuck_updates": 20,
        "resume": False,
        "checkpoint_every": 0,
        "recurrent": False,
        "lstm_hidden_size": 256,
        "sequence_length": 0,
        "nudge_right_steps": 0,
        "stuck_nudge_steps": 0,
        "no_reset_handshake": False,
        "dry_run_seconds": 0.0,
        "summary_every": 10,
        "summary_interval_sec": 60.0,
        "export_metrics": None,
    }


def _default_env_schema() -> dict:
    return {
        "rom_path": "VAMPIRE.ROM",
        "frame_size": (84, 84),
        "action_repeat": 2,
        "decision_fps": None,
        "capture_backend": "png",
        "post_action_delay_ms": 0.0,
        "max_episode_steps": 1500,
        "tmp_root": "runs/tmp",
        "soft_reset": True,
        "num_envs": 1,
        "window_title": None,
        "window_rects_path": None,
        "reset_handshake_stable_frames": 0,
        "reset_handshake_conf_min": 0.5,
        "reset_handshake_timeout_s": 15.0,
        "debug": False,
        "debug_every": 10,
        "debug_episode_max_steps": 0,
        "debug_dump_frames": 0,
        "debug_force_action": "",
        "debug_room_change": False,
        "ignore_death": False,
        "dump_hud_every_n_steps": 0,
        "perf_profile": False,
        "quiet": True,
        "openmsx_log_dir": None,
    }


def _default_capture() -> dict:
    return {
        "backend": "png",
        "fallback_policy": "fallback",
        "window_crop": None,
        "window_title": None,
        "capture_lag_ms": 0.0,
        "log_diagnostics": False,
    }


# ---------------------------------------------------------------------------
# Reward config loading (strict)
# ---------------------------------------------------------------------------

def _load_reward_config_strict(
    path_or_dict: str | dict | None,
    root: Path,
    novelty_override: float | None,
) -> tuple[Any, Path | None]:
    """Load RewardConfig. path_or_dict: path (str), already-loaded dict (from snapshot), or None."""
    from msx_env.reward.config import RewardConfig
    from msx_env.reward import default_v1_config

    if path_or_dict is not None:
        if isinstance(path_or_dict, dict):
            cfg = RewardConfig.from_dict(path_or_dict)
            if novelty_override is not None:
                cfg = _reward_config_with_novelty(cfg, novelty_override)
            return cfg, None
        path = path_or_dict
        p = (root / path).resolve() if not Path(path).is_absolute() else Path(path)
        if not p.exists():
            raise FileNotFoundError(
                f"Reward config explicitly specified but file not found: {p}. "
                "Fix --reward-config path or omit to use default."
            )
        with open(p, encoding="utf-8") as f:
            cfg = RewardConfig.from_dict(json.load(f))
        if novelty_override is not None:
            cfg = _reward_config_with_novelty(cfg, novelty_override)
        return cfg, p
    cfg = default_v1_config()
    if novelty_override is not None:
        cfg = _reward_config_with_novelty(cfg, novelty_override)
    return cfg, None


def _reward_config_with_novelty(cfg: Any, value: float) -> Any:
    from dataclasses import replace
    return replace(cfg, novelty_reward=float(value))


# ---------------------------------------------------------------------------
# Validation (fail-fast)
# ---------------------------------------------------------------------------

def validate_config(
    run_dir: Path,
    rom_path: Path,
    reward_config_path: Path | None,
    checkpoint_dir: Path,
    resume: bool,
    bc_checkpoint: Path | None,
    layout: RunLayout,
) -> None:
    """Fail fast if run_dir not writable, ROM missing, reward path missing when specified, etc."""
    errors = []

    if not run_dir.parent.exists() and run_dir != run_dir.resolve():
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            errors.append(f"run_dir not writable/createable: {run_dir} ({e})")
    else:
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / ".write_test").write_text("")
            (run_dir / ".write_test").unlink()
        except OSError as e:
            errors.append(f"run_dir not writable: {run_dir} ({e})")

    if reward_config_path is not None and not reward_config_path.exists():
        errors.append(f"reward_config path does not exist: {reward_config_path}")

    if not rom_path.exists():
        errors.append(f"ROM not found: {rom_path}")

    try:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        errors.append(f"checkpoint_dir not createable: {checkpoint_dir} ({e})")

    if layout.metrics_csv().exists():
        csv_dir = layout.metrics_csv().parent
        if csv_dir.resolve() != run_dir.resolve():
            errors.append(f"metrics path must be under run_dir: {layout.metrics_csv()}")

    if bc_checkpoint and not bc_checkpoint.exists():
        errors.append(f"bc_checkpoint not found: {bc_checkpoint}")

    if resume:
        last_pt = checkpoint_dir / "last.pt"
        if not last_pt.exists():
            errors.append(f"resume=True but checkpoint not found: {last_pt}")

    if errors:
        raise ValueError("Config validation failed:\n  " + "\n  ".join(errors))


# ---------------------------------------------------------------------------
# Build EnvConfig from schema (for msx_env)
# ---------------------------------------------------------------------------

def env_config_from_schema(
    schema: EnvConfigSchema,
    workdir: str,
    instance_id: int | str | None,
    reward_config: Any,
    run_dir: Path,
    window_crop: tuple[int, int, int, int] | None = None,
) -> "EnvConfig":
    """Build msx_env.env.EnvConfig from EnvConfigSchema and instance-specific overrides."""
    from msx_env.env import EnvConfig

    return EnvConfig(
        rom_path=str(schema.rom_path),
        workdir=workdir,
        frame_size=schema.frame_size,
        terminated_on_death=True,
        max_episode_steps=schema.max_episode_steps,
        action_repeat=schema.action_repeat,
        decision_fps=schema.decision_fps,
        capture_backend=schema.capture_backend,
        window_crop=window_crop,
        reward_config=reward_config,
        tmp_root=schema.tmp_root,
        post_action_delay_ms=schema.post_action_delay_ms,
        soft_reset=schema.soft_reset,
        instance_id=instance_id,
        window_title=schema.window_title,
        openmsx_log_dir=str(run_dir) if schema.openmsx_log_dir else None,
        quiet=schema.quiet,
        debug=schema.debug,
        debug_every=schema.debug_every,
        debug_episode_max_steps=schema.debug_episode_max_steps,
        debug_dump_frames=schema.debug_dump_frames,
        debug_force_action=schema.debug_force_action or "",
        debug_room_change=schema.debug_room_change,
        ignore_death=schema.ignore_death,
        dump_hud_every_n_steps=schema.dump_hud_every_n_steps,
        dump_hud_dir=str(run_dir / "debug") if schema.dump_hud_every_n_steps else None,
        perf_profile=schema.perf_profile,
        reset_handshake_stable_frames=schema.reset_handshake_stable_frames,
        reset_handshake_conf_min=schema.reset_handshake_conf_min,
        reset_handshake_timeout_s=schema.reset_handshake_timeout_s,
    )


# Add window_crop to EnvConfigSchema if we pass it from capture config
# ---------------------------------------------------------------------------
# CLI parser (single place)
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="PPO Vampire Killer — config via --config or CLI")
    p.add_argument("--config", type=str, default=None, help="Primary config file (JSON); CLI overrides")
    p.add_argument("--reward-config", type=str, default=None, help="Reward config JSON; if missing when specified -> fail")
    p.add_argument("--run-dir", type=str, default=None, help="Explicit run directory (absolute or relative to CWD)")
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--log-dir", type=str, default=None)
    p.add_argument("--use-runs-dir", action="store_true")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints/ppo")
    p.add_argument("--bc-checkpoint", type=str, default=None)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--rollout-steps", type=int, default=128)
    p.add_argument("--ppo-epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-eps", type=float, default=0.2)
    p.add_argument("--max-episode-steps", type=int, default=1500)
    p.add_argument("--step-penalty", type=float, default=-0.001)
    p.add_argument("--device", type=str, default="")
    p.add_argument("--arch", choices=["default", "deep"], default="deep")
    p.add_argument("--action-repeat", type=int, default=1)
    p.add_argument("--decision-fps", type=float, default=None)
    p.add_argument("--capture-backend", choices=["png", "single", "window"], default="png")
    p.add_argument("--capture-fallback", choices=["strict", "fallback"], default="fallback")
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--tmp-root", type=str, default="runs/tmp")
    p.add_argument("--post-action-delay-ms", type=float, default=None)
    p.add_argument("--novelty-reward", type=float, default=None)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--value-loss-coef", type=float, default=0.5)
    p.add_argument("--entropy-floor", type=float, default=0.3)
    p.add_argument("--stuck-updates", type=int, default=20)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--checkpoint-every", type=int, default=0)
    p.add_argument("--recurrent", action="store_true")
    p.add_argument("--lstm-hidden-size", type=int, default=256)
    p.add_argument("--sequence-length", type=int, default=0)
    p.add_argument("--dry-run-seconds", type=float, default=0.0)
    p.add_argument("--no-quiet", action="store_true")
    p.add_argument("--summary-every", type=int, default=10)
    p.add_argument("--summary-interval-sec", type=float, default=60.0)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--debug-room-change", action="store_true")
    p.add_argument("--debug-every", type=int, default=10)
    p.add_argument("--debug-episode-max-steps", type=int, default=0)
    p.add_argument("--debug-dump-frames", type=int, default=0)
    p.add_argument("--debug-force-action", type=str, default="")
    p.add_argument("--ignore-death", action="store_true")
    p.add_argument("--window-title-pattern", type=str, default=None)
    p.add_argument("--window-rects-json", type=str, default=None)
    p.add_argument("--no-reset-handshake", action="store_true")
    p.add_argument("--nudge-right-steps", type=int, default=0)
    p.add_argument("--stuck-nudge-steps", type=int, default=20)
    p.add_argument("--dump-hud-every-n-steps", type=int, default=0)
    p.add_argument("--perf", action="store_true")
    p.add_argument("--export-metrics", type=str, default=None)
    return p


def _apply_config_file(args: argparse.Namespace, root: Path) -> None:
    """Load --config file and set args. Supports flat dict or nested {run, ppo} snapshot. If path missing -> fail."""
    path = getattr(args, "config", None)
    if not path:
        return
    p = (root / path).resolve() if not Path(path).is_absolute() else Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file explicitly specified but not found: {p}. Fix --config or omit.")
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    # Nested snapshot (from config_snapshot.json)
    if "ppo" in data:
        for key, value in data["ppo"].items():
            if hasattr(args, key):
                setattr(args, key, value)
        if "run" in data:
            run = data["run"]
            if "run_dir" in run and hasattr(args, "run_dir"):
                setattr(args, "run_dir", run["run_dir"])
    else:
        for key, value in data.items():
            if hasattr(args, key):
                setattr(args, key, value)
            if key == "max_updates" and hasattr(args, "epochs"):
                setattr(args, "epochs", value)


def _resolve_paths(args: argparse.Namespace, root: Path) -> tuple[Path, Path, Path]:
    """Resolve run_dir, checkpoint_dir, rom_path. Create run_dir if use_runs_dir or run_dir set."""
    rom_path = root / "VAMPIRE.ROM"
    if getattr(args, "run_dir", None):
        run_dir = Path(args.run_dir).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir = run_dir / "checkpoints" / "ppo"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        return run_dir, ckpt_dir, rom_path
    if getattr(args, "use_runs_dir", False):
        from scripts.run_utils import make_run_dir
        run_name = getattr(args, "run_name", None) or "run"
        run_dir = make_run_dir(run_name, runs_base="runs")
        ckpt_dir = run_dir / "checkpoints" / "ppo"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        return run_dir, ckpt_dir, rom_path
    log_dir = getattr(args, "log_dir", None) or args.checkpoint_dir
    run_name = getattr(args, "run_name", None) or "run"
    run_dir = (root / log_dir.replace("\\", "/").strip("/")).resolve() / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = (root / args.checkpoint_dir.replace("\\", "/").strip("/")).resolve()
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, ckpt_dir, rom_path


def load_config(argv: list[str] | None = None) -> ResolvedConfig:
    """
    Parse CLI, load optional config file, merge (defaults -> file -> CLI), resolve paths, validate.
    Returns single ResolvedConfig. Writes config_snapshot.json and resolved_paths.json into run_dir.
    """
    if argv is None:
        argv = sys.argv[1:]
    parser = _build_parser()
    args = parser.parse_args(argv)

    root = ROOT
    if getattr(args, "config", None):
        _apply_config_file(args, root)

    run_dir, ckpt_dir, rom_path = _resolve_paths(args, root)
    layout = RunLayout(run_dir, LoggingConfig())

    # Reward config (strict)
    reward_cfg, reward_path = _load_reward_config_strict(
        getattr(args, "reward_config", None),
        root,
        getattr(args, "novelty_reward", None),
    )
    if reward_path is None:
        logging.getLogger(__name__).info("Using default reward config (v1); no --reward-config provided.")

    # Device
    device = getattr(args, "device", "") or ("cuda" if __import__("torch").cuda.is_available() else "cpu")

    # Env schema
    num_envs = max(1, int(getattr(args, "num_envs", 1)))
    post_delay = getattr(args, "post_action_delay_ms", None)
    if post_delay is None and num_envs > 1:
        post_delay = 50.0
    elif post_delay is None:
        post_delay = 0.0
    tmp_root_abs = str((root / getattr(args, "tmp_root", "runs/tmp")).resolve()) if num_envs > 1 else getattr(args, "tmp_root", "runs/tmp")
    window_rects_path = None
    if getattr(args, "window_rects_json", None):
        wp = Path(args.window_rects_json)
        window_rects_path = (root / wp).resolve() if not wp.is_absolute() else wp

    env_schema = EnvConfigSchema(
        rom_path=rom_path,
        frame_size=(84, 84),
        action_repeat=getattr(args, "action_repeat", 2),
        decision_fps=getattr(args, "decision_fps", None),
        capture_backend=getattr(args, "capture_backend", "png"),
        post_action_delay_ms=post_delay,
        max_episode_steps=getattr(args, "max_episode_steps", 1500),
        tmp_root=tmp_root_abs,
        soft_reset=True,
        num_envs=num_envs,
        window_title=getattr(args, "window_title_pattern", None),
        window_rects_path=window_rects_path,
        reset_handshake_stable_frames=3 if (num_envs > 1 and not getattr(args, "no_reset_handshake", False)) else 0,
        reset_handshake_conf_min=0.5,
        reset_handshake_timeout_s=15.0,
        debug=getattr(args, "debug", False),
        debug_every=max(1, getattr(args, "debug_every", 10)),
        debug_episode_max_steps=max(0, getattr(args, "debug_episode_max_steps", 0)),
        debug_dump_frames=max(0, getattr(args, "debug_dump_frames", 0)),
        debug_force_action=(getattr(args, "debug_force_action", "") or "").strip(),
        debug_room_change=getattr(args, "debug_room_change", False),
        ignore_death=getattr(args, "ignore_death", False),
        dump_hud_every_n_steps=max(0, getattr(args, "dump_hud_every_n_steps", 0)),
        perf_profile=getattr(args, "perf", False),
        quiet=not getattr(args, "no_quiet", False),
        openmsx_log_dir=run_dir,
    )

    capture_cfg = CaptureConfig(
        backend=getattr(args, "capture_backend", "png"),
        fallback_policy=getattr(args, "capture_fallback", "fallback"),
        window_crop=None,
        window_title=getattr(args, "window_title_pattern", None),
        capture_lag_ms=0.0,
        log_diagnostics=getattr(args, "perf", False),
    )

    run_cfg = RunConfig(
        run_dir=run_dir,
        experiment_name=getattr(args, "run_name", None) or "run",
        seed=42,
        device=device,
        use_runs_dir=getattr(args, "use_runs_dir", False),
    )

    export_path = None
    if getattr(args, "export_metrics", None):
        export_path = (root / args.export_metrics).resolve()

    ppo_cfg = PPOConfig(
        checkpoint_dir=ckpt_dir,
        bc_checkpoint=(root / args.bc_checkpoint).resolve() if args.bc_checkpoint else None,
        epochs=args.epochs,
        rollout_steps=args.rollout_steps,
        ppo_epochs=args.ppo_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_eps=args.clip_eps,
        entropy_coef=args.entropy_coef,
        value_loss_coef=args.value_loss_coef,
        max_episode_steps=args.max_episode_steps,
        step_penalty=args.step_penalty,
        arch=args.arch,
        entropy_floor=args.entropy_floor,
        stuck_updates=args.stuck_updates,
        resume=args.resume,
        checkpoint_every=args.checkpoint_every,
        recurrent=args.recurrent,
        lstm_hidden_size=args.lstm_hidden_size,
        sequence_length=args.sequence_length,
        nudge_right_steps=max(0, getattr(args, "nudge_right_steps", 0)),
        stuck_nudge_steps=max(0, getattr(args, "stuck_nudge_steps", 0)),
        no_reset_handshake=getattr(args, "no_reset_handshake", False),
        dry_run_seconds=max(0.0, args.dry_run_seconds),
        summary_every=max(1, args.summary_every),
        summary_interval_sec=max(1.0, args.summary_interval_sec),
        export_metrics=export_path,
    )

    validate_config(
        run_dir, rom_path, reward_path, ckpt_dir, args.resume, ppo_cfg.bc_checkpoint, layout
    )

    # Resolved paths manifest
    try:
        from scripts.run_utils import code_version
        _code_ver = code_version()
    except Exception:
        _code_ver = "unknown"
    resolved_paths = {
        "run_dir": str(run_dir.resolve()),
        "checkpoint_dir": str(ckpt_dir.resolve()),
        "rom_path": str(rom_path.resolve()),
        "metrics_csv": str(layout.metrics_csv().resolve()),
        "train_log": str(layout.train_log().resolve()),
        "config_snapshot": str(layout.config_snapshot().resolve()),
        "reward_config_path": str(reward_path.resolve()) if reward_path is not None else None,
        "code_version": _code_ver,
    }
    layout.ensure_run_dir()
    with open(layout.resolved_paths_file(), "w", encoding="utf-8") as f:
        json.dump(resolved_paths, f, indent=2, ensure_ascii=False)

    # Config snapshot (full resolved config for supervisor/trainer)
    snapshot = {
        "run": {
            "run_dir": str(run_dir),
            "experiment_name": run_cfg.experiment_name,
            "device": device,
            "code_version": _code_ver,
        },
        "ppo": {
            "checkpoint_dir": str(ckpt_dir),
            "epochs": ppo_cfg.epochs,
            "rollout_steps": ppo_cfg.rollout_steps,
            "lr": ppo_cfg.lr,
            "gamma": ppo_cfg.gamma,
            "gae_lambda": ppo_cfg.gae_lambda,
            "clip_eps": ppo_cfg.clip_eps,
            "entropy_coef": ppo_cfg.entropy_coef,
            "value_loss_coef": ppo_cfg.value_loss_coef,
            "num_envs": num_envs,
            "max_episode_steps": ppo_cfg.max_episode_steps,
            "action_repeat": env_schema.action_repeat,
            "reward_config": reward_cfg.to_dict() if hasattr(reward_cfg, "to_dict") else {},
            "recurrent": ppo_cfg.recurrent,
            "lstm_hidden_size": ppo_cfg.lstm_hidden_size,
            "arch": ppo_cfg.arch,
        },
        "reward_version": reward_cfg.version if hasattr(reward_cfg, "version") else "v1",
        "key_reward": getattr(reward_cfg, "key_reward", 0.0),
        "door_reward": getattr(reward_cfg, "door_reward", 0.0),
        "stage_reward_enabled": getattr(reward_cfg, "enable_stage_reward", False),
        "capture_backend": env_schema.capture_backend,
    }
    with open(layout.config_snapshot(), "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)

    return ResolvedConfig(
        run=run_cfg,
        ppo=ppo_cfg,
        env_schema=env_schema,
        reward_config=reward_cfg,
        reward_config_path=reward_path,
        capture=capture_cfg,
        logging=LoggingConfig(),
        layout=layout,
    )


def build_resolved_config_from_args(args: argparse.Namespace, root: Path | None = None) -> ResolvedConfig:
    """
    Build ResolvedConfig from an already-parsed argparse.Namespace (e.g. from train_ppo.parse_args()).
    Validates, writes config_snapshot.json and resolved_paths.json into run_dir.
    Use when not using --config (legacy CLI path).
    """
    root = root or ROOT
    run_dir, ckpt_dir, rom_path = _resolve_paths(args, root)
    layout = RunLayout(run_dir, LoggingConfig())

    reward_cfg, reward_path = _load_reward_config_strict(
        getattr(args, "reward_config", None),
        root,
        getattr(args, "novelty_reward", None),
    )
    if reward_path is None:
        logging.getLogger(__name__).info("Using default reward config (v1); no --reward-config provided.")

    import torch
    device = getattr(args, "device", "") or ("cuda" if torch.cuda.is_available() else "cpu")
    num_envs = max(1, int(getattr(args, "num_envs", 1)))
    post_delay = getattr(args, "post_action_delay_ms", None)
    if post_delay is None and num_envs > 1:
        post_delay = 50.0
    elif post_delay is None:
        post_delay = 0.0
    tmp_root_abs = str((root / getattr(args, "tmp_root", "runs/tmp")).resolve()) if num_envs > 1 else getattr(args, "tmp_root", "runs/tmp")
    window_rects_path = None
    if getattr(args, "window_rects_json", None):
        wp = Path(args.window_rects_json)
        window_rects_path = (root / wp).resolve() if not wp.is_absolute() else wp

    env_schema = EnvConfigSchema(
        rom_path=rom_path,
        frame_size=(84, 84),
        action_repeat=getattr(args, "action_repeat", 2),
        decision_fps=getattr(args, "decision_fps", None),
        capture_backend=getattr(args, "capture_backend", "png"),
        post_action_delay_ms=post_delay,
        max_episode_steps=getattr(args, "max_episode_steps", 1500),
        tmp_root=tmp_root_abs,
        soft_reset=True,
        num_envs=num_envs,
        window_title=getattr(args, "window_title_pattern", None),
        window_rects_path=window_rects_path,
        reset_handshake_stable_frames=3 if (num_envs > 1 and not getattr(args, "no_reset_handshake", False)) else 0,
        reset_handshake_conf_min=0.5,
        reset_handshake_timeout_s=15.0,
        debug=getattr(args, "debug", False),
        debug_every=max(1, getattr(args, "debug_every", 10)),
        debug_episode_max_steps=max(0, getattr(args, "debug_episode_max_steps", 0)),
        debug_dump_frames=max(0, getattr(args, "debug_dump_frames", 0)),
        debug_force_action=(getattr(args, "debug_force_action", "") or "").strip(),
        debug_room_change=getattr(args, "debug_room_change", False),
        ignore_death=getattr(args, "ignore_death", False),
        dump_hud_every_n_steps=max(0, getattr(args, "dump_hud_every_n_steps", 0)),
        perf_profile=getattr(args, "perf", False),
        quiet=not getattr(args, "no_quiet", False),
        openmsx_log_dir=run_dir,
    )

    capture_cfg = CaptureConfig(
        backend=getattr(args, "capture_backend", "png"),
        fallback_policy=getattr(args, "capture_fallback", "fallback"),
        window_crop=None,
        window_title=getattr(args, "window_title_pattern", None),
        capture_lag_ms=0.0,
        log_diagnostics=getattr(args, "perf", False),
    )

    run_cfg = RunConfig(
        run_dir=run_dir,
        experiment_name=getattr(args, "run_name", None) or "run",
        seed=42,
        device=device,
        use_runs_dir=getattr(args, "use_runs_dir", False),
    )

    export_path = None
    if getattr(args, "export_metrics", None):
        export_path = (root / args.export_metrics).resolve()

    ppo_cfg = PPOConfig(
        checkpoint_dir=ckpt_dir,
        bc_checkpoint=(root / args.bc_checkpoint).resolve() if args.bc_checkpoint else None,
        epochs=args.epochs,
        rollout_steps=args.rollout_steps,
        ppo_epochs=args.ppo_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_eps=args.clip_eps,
        entropy_coef=args.entropy_coef,
        value_loss_coef=args.value_loss_coef,
        max_episode_steps=args.max_episode_steps,
        step_penalty=args.step_penalty,
        arch=args.arch,
        entropy_floor=args.entropy_floor,
        stuck_updates=args.stuck_updates,
        resume=args.resume,
        checkpoint_every=args.checkpoint_every,
        recurrent=args.recurrent,
        lstm_hidden_size=args.lstm_hidden_size,
        sequence_length=args.sequence_length,
        nudge_right_steps=max(0, getattr(args, "nudge_right_steps", 0)),
        stuck_nudge_steps=max(0, getattr(args, "stuck_nudge_steps", 0)),
        no_reset_handshake=getattr(args, "no_reset_handshake", False),
        dry_run_seconds=max(0.0, args.dry_run_seconds),
        summary_every=max(1, args.summary_every),
        summary_interval_sec=max(1.0, args.summary_interval_sec),
        export_metrics=export_path,
    )

    validate_config(
        run_dir, rom_path, reward_path, ckpt_dir, args.resume, ppo_cfg.bc_checkpoint, layout
    )

    try:
        from scripts.run_utils import code_version
        _code_ver = code_version()
    except Exception:
        _code_ver = "unknown"
    resolved_paths = {
        "run_dir": str(run_dir.resolve()),
        "checkpoint_dir": str(ckpt_dir.resolve()),
        "rom_path": str(rom_path.resolve()),
        "metrics_csv": str(layout.metrics_csv().resolve()),
        "train_log": str(layout.train_log().resolve()),
        "config_snapshot": str(layout.config_snapshot().resolve()),
        "reward_config_path": str(reward_path.resolve()) if reward_path is not None else None,
        "code_version": _code_ver,
    }
    layout.ensure_run_dir()
    with open(layout.resolved_paths_file(), "w", encoding="utf-8") as f:
        json.dump(resolved_paths, f, indent=2, ensure_ascii=False)

    snapshot = {
        "run": {"run_dir": str(run_dir), "experiment_name": run_cfg.experiment_name, "device": device, "code_version": _code_ver},
        "ppo": {
            "checkpoint_dir": str(ckpt_dir),
            "epochs": ppo_cfg.epochs,
            "rollout_steps": ppo_cfg.rollout_steps,
            "lr": ppo_cfg.lr,
            "gamma": ppo_cfg.gamma,
            "gae_lambda": ppo_cfg.gae_lambda,
            "clip_eps": ppo_cfg.clip_eps,
            "entropy_coef": ppo_cfg.entropy_coef,
            "value_loss_coef": ppo_cfg.value_loss_coef,
            "num_envs": num_envs,
            "max_episode_steps": ppo_cfg.max_episode_steps,
            "action_repeat": env_schema.action_repeat,
            "reward_config": reward_cfg.to_dict() if hasattr(reward_cfg, "to_dict") else {},
            "recurrent": ppo_cfg.recurrent,
            "lstm_hidden_size": ppo_cfg.lstm_hidden_size,
            "arch": ppo_cfg.arch,
        },
        "reward_version": reward_cfg.version if hasattr(reward_cfg, "version") else "v1",
        "key_reward": getattr(reward_cfg, "key_reward", 0.0),
        "door_reward": getattr(reward_cfg, "door_reward", 0.0),
        "stage_reward_enabled": getattr(reward_cfg, "enable_stage_reward", False),
        "capture_backend": env_schema.capture_backend,
    }
    with open(layout.config_snapshot(), "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2, ensure_ascii=False)

    return ResolvedConfig(
        run=run_cfg,
        ppo=ppo_cfg,
        env_schema=env_schema,
        reward_config=reward_cfg,
        reward_config_path=reward_path,
        capture=capture_cfg,
        logging=LoggingConfig(),
        layout=layout,
    )
