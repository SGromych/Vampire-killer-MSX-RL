"""
Бенчмарк env: N шагов, steps/sec, сравнение бэкендов capture (png, single, window, dxcam).
Метрики: capture_time, preprocessing_time, total step — avg и p95 (мс).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from msx_env.env import EnvConfig, VampireKillerEnv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark Vampire Killer env")
    p.add_argument("--backend", choices=["png", "single", "window", "dxcam"], default="dxcam")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--action-repeat", type=int, default=1)
    p.add_argument("--decision-fps", type=float, default=None)
    p.add_argument("--workdir", type=str, default=None)
    p.add_argument("--window-crop", type=int, nargs=4, metavar=("X", "Y", "W", "H"), default=None)
    p.add_argument("--window-title", type=str, default=None)
    p.add_argument("--capture-lag-ms", type=float, default=0)
    return p.parse_args()


def _avg_p95_ms(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    avg = sum(values) / len(values)
    sorted_vals = sorted(values)
    p95_idx = min(int(len(sorted_vals) * 0.95), len(sorted_vals) - 1)
    p95 = sorted_vals[p95_idx]
    return avg * 1000, p95 * 1000


def main() -> None:
    args = parse_args()
    rom = ROOT / "VAMPIRE.ROM"
    if not rom.exists():
        raise FileNotFoundError(f"ROM не найден: {rom}")

    workdir = args.workdir or str(ROOT / "checkpoints" / "ppo" / "run")
    Path(workdir).mkdir(parents=True, exist_ok=True)

    cfg_kw = dict(
        rom_path=str(rom),
        workdir=workdir,
        frame_size=(84, 84),
        capture_backend=args.backend,
        action_repeat=args.action_repeat,
        decision_fps=args.decision_fps,
    )
    if args.backend in ("window", "dxcam"):
        if args.window_crop is not None and args.backend == "window":
            cfg_kw["window_crop"] = tuple(args.window_crop)
        if args.window_title is not None:
            cfg_kw["window_title"] = args.window_title
        cfg_kw["capture_lag_ms"] = args.capture_lag_ms

    env = VampireKillerEnv(EnvConfig(**cfg_kw))

    try:
        obs, _ = env.reset()
        print(f"Benchmark: backend={args.backend} steps={args.steps} action_repeat={args.action_repeat} decision_fps={args.decision_fps}")

        t0 = time.perf_counter()
        for i in range(args.steps):
            action = 0
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()
        elapsed = time.perf_counter() - t0

        steps_per_sec = args.steps / elapsed if elapsed > 0 else 0
        steps_per_min = steps_per_sec * 60
        print(f"Done: {args.steps} steps in {elapsed:.2f}s")
        print(f"Steps/sec: {steps_per_sec:.2f}")
        print(f"Steps/min: {steps_per_min:.1f}")

        timings = list(getattr(env, "_perf_timings", []))
        if timings:
            for key in ("capture_time", "preprocessing_time", "env_step_time"):
                vals = [t[key] for t in timings if key in t]
                if vals:
                    avg_ms, p95_ms = _avg_p95_ms(vals)
                    print(f"  {key}: avg={avg_ms:.1f}ms p95={p95_ms:.1f}ms")
    finally:
        env.close()


if __name__ == "__main__":
    main()
