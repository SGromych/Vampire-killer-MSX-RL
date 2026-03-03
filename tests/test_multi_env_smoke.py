"""
Smoke test: 2 env с разными workdir, 200 шагов каждый (последовательно и проверка изоляции).
Без эмулятора: проверяем только что make_env создаёт конфиги с разными workdir.
С эмулятором: scripts/run_2env_smoke.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from msx_env.env import EnvConfig
from msx_env.make_env import make_env


def test_make_env_isolates_workdir() -> None:
    base = EnvConfig(
        rom_path="dummy.rom",
        workdir="/tmp/ppo_run",
        tmp_root=str(ROOT / "runs" / "tmp"),
    )
    fn0 = make_env(0, base)
    fn1 = make_env(1, base)
    env0 = fn0()
    env1 = fn1()
    w0 = env0.cfg.workdir
    w1 = env1.cfg.workdir
    assert w0 != w1, f"workdirs must differ: {w0} vs {w1}"
    assert "0" in w0 or "0" in Path(w0).name
    assert "1" in w1 or "1" in Path(w1).name
    assert env0.cfg.instance_id == 0
    assert env1.cfg.instance_id == 1
    env0.close()
    env1.close()
    print("test_make_env_isolates_workdir: OK (unique workdirs, instance_id)")


if __name__ == "__main__":
    test_make_env_isolates_workdir()
