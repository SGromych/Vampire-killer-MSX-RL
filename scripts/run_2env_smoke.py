"""
Ручной smoke: 2 env, по 200 шагов каждый. Проверка отсутствия коллизий (уникальные workdir).
Требует VAMPIRE.ROM и openMSX.
  python scripts/run_2env_smoke.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from msx_env.env import EnvConfig, VampireKillerEnv
from msx_env.make_env import make_env


def main() -> None:
    rom = ROOT / "VAMPIRE.ROM"
    if not rom.exists():
        print("VAMPIRE.ROM not found, skipping (config-only check).")
        base = EnvConfig(rom_path="dummy", workdir="/tmp/d", tmp_root=str(ROOT / "runs" / "tmp"))
        fn0, fn1 = make_env(0, base), make_env(1, base)
        e0, e1 = fn0(), fn1()
        print(f"  env0 workdir={e0.cfg.workdir} instance_id={e0.cfg.instance_id}")
        print(f"  env1 workdir={e1.cfg.workdir} instance_id={e1.cfg.instance_id}")
        assert e0.cfg.workdir != e1.cfg.workdir
        e0.close()
        e1.close()
        return

    tmp_root = ROOT / "runs" / "tmp"
    base_cfg = EnvConfig(
        rom_path=str(rom),
        workdir=str(ROOT / "checkpoints" / "ppo" / "run"),
        frame_size=(84, 84),
        tmp_root=str(tmp_root),
        capture_backend="png",
    )
    env_fns = [make_env(0, base_cfg), make_env(1, base_cfg)]
    envs = [fn() for fn in env_fns]
    assert envs[0].cfg.workdir != envs[1].cfg.workdir
    print(f"env0 workdir={envs[0].cfg.workdir}")
    print(f"env1 workdir={envs[1].cfg.workdir}")

    for i, e in enumerate(envs):
        obs, _ = e.reset()
        for step in range(200):
            obs, _, term, trunc, _ = e.step(0)
            if term or trunc:
                obs, _ = e.reset()
        print(f"  env{i} completed 200 steps")

    for e in envs:
        e.close()
    print("run_2env_smoke: OK (no collisions)")


if __name__ == "__main__":
    main()
