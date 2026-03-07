#!/usr/bin/env python3
"""
A/B experiment: compare action_repeat=1 vs action_repeat=2 for RL throughput and learning quality.

Runs two short training sessions (Run A: action_repeat=1, Run B: action_repeat=2), then reads
metrics.csv from each run and prints a comparison table and conclusion.

Usage:
  python tools/action_repeat_experiment.py
  python tools/action_repeat_experiment.py --updates 30 --num-envs 1
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _run_training(
    run_dir: Path,
    action_repeat: int,
    run_name: str,
    updates: int = 50,
    num_envs: int = 1,
    rollout_steps: int = 128,
    reward_config: str | None = None,
) -> bool:
    """Run train_ppo.py with given args. Returns True if exit code is 0."""
    cmd = [
        sys.executable,
        str(ROOT / "train_ppo.py"),
        "--run-dir",
        str(run_dir.resolve()),
        "--run-name",
        run_name,
        "--action-repeat",
        str(action_repeat),
        "--epochs",
        str(updates),
        "--num-envs",
        str(num_envs),
        "--rollout-steps",
        str(rollout_steps),
    ]
    if reward_config:
        cmd.extend(["--reward-config", str(reward_config)])
    run_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(cmd, cwd=str(ROOT))
    return result.returncode == 0


def _read_metrics_and_config(run_dir: Path) -> dict | None:
    """
    Read metrics.csv (last row and averages) and config_snapshot.json from run_dir.
    Returns a dict with steps_per_sec, total_steps, unique_rooms_ep_mean, key_found_rate,
    door_found_rate, stage00_exit_rate_ep (from last row or average of last N rows), or None if missing.
    """
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        return None

    snapshot_path = run_dir / "config_snapshot.json"
    rollout_steps = 128
    num_envs = 1
    if snapshot_path.exists():
        try:
            with open(snapshot_path, encoding="utf-8") as f:
                snap = json.load(f)
            ppo = snap.get("ppo", {})
            rollout_steps = int(ppo.get("rollout_steps", rollout_steps))
            num_envs = int(ppo.get("num_envs", num_envs))
        except Exception:
            pass

    rows = []
    with open(metrics_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return None
        for row in reader:
            rows.append(row)

    if not rows:
        return None

    last = rows[-1]
    last_n = rows[-10:] if len(rows) >= 10 else rows

    def _float(key: str, default: float = 0.0) -> float:
        try:
            return float(last.get(key, default))
        except (TypeError, ValueError):
            return default

    def _avg(key: str, default: float = 0.0) -> float:
        vals = []
        for r in last_n:
            try:
                v = r.get(key, "")
                if v != "":
                    vals.append(float(v))
            except (TypeError, ValueError):
                pass
        return sum(vals) / len(vals) if vals else _float(key, default)

    last_update = int(_float("update", 0))
    steps_per_update = rollout_steps * num_envs
    total_steps = last_update * steps_per_update

    return {
        "steps_per_sec": _avg("steps_per_sec", 0.0),
        "total_steps": total_steps,
        "unique_rooms_ep_mean": _avg("unique_rooms_ep_mean", 0.0),
        "key_found_rate": _avg("key_found_rate", 0.0),
        "door_found_rate": _avg("door_found_rate", 0.0),
        "stage00_exit_rate_ep": _avg("stage00_exit_rate_ep", 0.0),
    }


def _print_table(m1: dict | None, m2: dict | None) -> None:
    """Print comparison table for repeat=1 vs repeat=2."""
    metrics_order = [
        "steps_per_sec",
        "total_steps",
        "unique_rooms_ep_mean",
        "key_found_rate",
        "door_found_rate",
        "stage00_exit_rate_ep",
    ]
    labels = {
        "steps_per_sec": "steps/sec",
        "total_steps": "total_steps",
        "unique_rooms_ep_mean": "unique_rooms_ep_mean",
        "key_found_rate": "key_found_rate",
        "door_found_rate": "door_found_rate",
        "stage00_exit_rate_ep": "stage00_exit_rate_ep",
    }

    def _fmt(k: str, v: float) -> str:
        if k == "total_steps":
            return str(int(v))
        if "rate" in k:
            return f"{v:.2%}" if 0 <= v <= 1.1 else f"{v:.4f}"
        return f"{v:.4f}"

    print("\nAction Repeat Experiment Results\n")
    print(f"{'Metric':<28} {'repeat=1':>14} {'repeat=2':>14}")
    print("-" * 58)

    for key in metrics_order:
        label = labels.get(key, key)
        v1 = (m1 or {}).get(key, 0.0)
        v2 = (m2 or {}).get(key, 0.0)
        print(f"{label:<28} {_fmt(key, v1):>14} {_fmt(key, v2):>14}")

    print()


def _conclusion(m1: dict | None, m2: dict | None) -> str:
    """Return a short conclusion string."""
    if not m1 or not m2:
        return "Incomplete data (one or both runs failed or produced no metrics)."

    faster = m2.get("steps_per_sec", 0) > m1.get("steps_per_sec", 0)
    rooms_ok = m2.get("unique_rooms_ep_mean", 0) >= m1.get("unique_rooms_ep_mean", 0)
    stage_ok = m2.get("stage00_exit_rate_ep", 0) >= m1.get("stage00_exit_rate_ep", 0)

    if faster and rooms_ok and stage_ok:
        return "action_repeat=2 appears beneficial"
    return "action_repeat=2 may hurt exploration or task progress"


def main() -> None:
    parser = argparse.ArgumentParser(description="A/B experiment: action_repeat=1 vs 2")
    parser.add_argument("--updates", type=int, default=50, help="Number of PPO updates per run")
    parser.add_argument("--num-envs", type=int, default=1, help="Number of envs (default 1 for faster experiment)")
    parser.add_argument("--rollout-steps", type=int, default=128, help="Rollout steps per update")
    parser.add_argument("--reward-config", type=str, default=None, help="Path to reward config JSON (optional)")
    parser.add_argument(
        "--experiment-dir",
        type=str,
        default=None,
        help="Base dir for run dirs (default: runs/action_repeat_experiment)",
    )
    args = parser.parse_args()

    base = Path(args.experiment_dir) if args.experiment_dir else ROOT / "runs" / "action_repeat_experiment"
    base.mkdir(parents=True, exist_ok=True)

    run_dir_1 = base / "run_action_repeat_1"
    run_dir_2 = base / "run_action_repeat_2"

    reward_cfg = None
    if args.reward_config:
        p = Path(args.reward_config)
        reward_cfg = str(ROOT / p) if not p.is_absolute() else str(p)

    print("Run A: action_repeat=1 ...")
    ok1 = _run_training(
        run_dir_1,
        action_repeat=1,
        run_name="run_action_repeat_1",
        updates=args.updates,
        num_envs=args.num_envs,
        rollout_steps=args.rollout_steps,
        reward_config=reward_cfg,
    )
    print("Run B: action_repeat=2 ...")
    ok2 = _run_training(
        run_dir_2,
        action_repeat=2,
        run_name="run_action_repeat_2",
        updates=args.updates,
        num_envs=args.num_envs,
        rollout_steps=args.rollout_steps,
        reward_config=reward_cfg,
    )

    m1 = _read_metrics_and_config(run_dir_1)
    m2 = _read_metrics_and_config(run_dir_2)

    _print_table(m1, m2)
    print("Conclusion:", _conclusion(m1, m2))
    if not ok1 or not ok2:
        sys.exit(1)


if __name__ == "__main__":
    main()
