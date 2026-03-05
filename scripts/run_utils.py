"""
Утилиты для run directory: runs/<timestamp>_<gitshort>_<runname>/
"""
from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _git_short() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()[:8]
    except Exception:
        pass
    return "nogit"


def code_version() -> str:
    """Версия кода для логов: git short hash или 'nogit'. По ней в train.log можно убедиться, что в прогоне участвует нужный коммит."""
    return _git_short()


def make_run_dir(run_name: str | None = None, runs_base: str | Path = "runs") -> Path:
    """
    Создать runs/<timestamp>_<gitshort>_<runname>/
    run_name: суффикс (auto_night, exp01 и т.д.)
    """
    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    g = _git_short()
    suffix = (run_name or "run").replace(" ", "_").replace("/", "_")
    name = f"{ts}_{g}_{suffix}"
    base = ROOT / runs_base if isinstance(runs_base, str) else runs_base
    run_dir = base / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def run_metadata(run_dir: Path | None = None) -> tuple[str, str, int]:
    """run_id (путь), hostname, pid."""
    import socket
    run_id = str(run_dir) if run_dir else os.environ.get("RUN_DIR", "")
    host = socket.gethostname()
    pid = os.getpid()
    return (run_id, host, pid)
