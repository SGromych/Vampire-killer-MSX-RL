"""
Lightweight perf timer accumulator for throughput diagnostics.
Used when EnvConfig.perf_profile=True or diagnose_throughput.py.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any
import time


def _percentile(samples: list[float], p: float) -> float:
    """p in [0, 100]; returns p-th percentile."""
    if not samples:
        return 0.0
    sorted_ = sorted(samples)
    idx = max(0, min(len(sorted_) - 1, int(len(sorted_) * p / 100)))
    return sorted_[idx]


@dataclass
class PerfTimerAccumulator:
    """Holds samples and computes p50/p95; per-env timers."""
    max_samples: int = 5000
    # Per-env samples: key = "t_action_ms", "t_capture_ms", etc.
    samples: dict[str, list[float]] = field(default_factory=dict)
    # Per-env: key = "t_capture_ms_env0", "t_capture_ms_env1", etc.
    per_env_samples: dict[str, list[float]] = field(default_factory=dict)
    # Ring buffer for update wall times
    update_wall_sec: deque = field(default_factory=lambda: deque(maxlen=200))

    def record(self, name: str, ms: float, env_id: int | None = None) -> None:
        key = name if env_id is None else f"{name}_env{env_id}"
        dst = self.samples if env_id is None else self.per_env_samples
        if key not in dst:
            dst[key] = []
        lst = dst[key]
        lst.append(ms)
        if len(lst) > self.max_samples:
            lst.pop(0)

    def record_update_wall_sec(self, sec: float) -> None:
        self.update_wall_sec.append(sec)

    def get_stats(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, lst in self.samples.items():
            if lst:
                out[f"{key}_p50"] = _percentile(lst, 50)
                out[f"{key}_p95"] = _percentile(lst, 95)
                out[f"{key}_mean"] = sum(lst) / len(lst)
                out[f"{key}_count"] = len(lst)
        for key, lst in self.per_env_samples.items():
            if lst:
                out[f"{key}_p50"] = _percentile(lst, 50)
                out[f"{key}_p95"] = _percentile(lst, 95)
                out[f"{key}_mean"] = sum(lst) / len(lst)
        if self.update_wall_sec:
            uw = list(self.update_wall_sec)
            out["update_wall_sec_p50"] = _percentile(uw, 50)
            out["update_wall_sec_p95"] = _percentile(uw, 95)
            out["update_wall_sec_mean"] = sum(uw) / len(uw)
        return out

    def clear(self) -> None:
        self.samples.clear()
        self.per_env_samples.clear()
        self.update_wall_sec.clear()
