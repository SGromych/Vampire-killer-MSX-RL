import json
import hashlib
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


DATASET_VERSION = 1


@dataclass
class DemoMetadata:
    version: int
    game_id: str
    machine_config: str
    rom_path: str
    action_set_version: str
    obs_shape: Tuple[int, ...]
    obs_dtype: str
    step_rate_hz: float
    created_at: float
    total_steps: int
    max_minutes: float
    run_id: str


def _sha1_bytes(arr: np.ndarray) -> str:
    h = hashlib.sha1()
    h.update(arr.tobytes())
    return h.hexdigest()


def save_demo_run(
    run_dir: Path,
    obs: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    next_obs: np.ndarray,
    dones: np.ndarray,
    timestamps: np.ndarray,
    meta: DemoMetadata,
    extras: Dict[str, Any] | None = None,
) -> None:
    """Сохранить демонстрацию в .npz + manifest.json (атомарно)."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    data_path = run_dir / "data.npz"
    tmp_path = run_dir / "data.npz.tmp"

    # Важно: открываем файл сами, чтобы np.savez_compressed не дописывал .npz к имени
    with open(tmp_path, "wb") as f:
        np.savez_compressed(
            f,
            obs=obs,
            actions=actions,
            rewards=rewards,
            next_obs=next_obs,
            dones=dones,
            timestamps=timestamps,
        )
    os.replace(tmp_path, data_path)

    manifest: Dict[str, Any] = {
        "schema_version": DATASET_VERSION,
        "metadata": asdict(meta),
        "files": {},
    }

    # Хэши по нескольким первым/последним кадрам (для отладки)
    if len(obs) > 0:
        manifest["frame_hashes"] = {
            "first": _sha1_bytes(obs[0]),
            "last": _sha1_bytes(obs[-1]),
        }

    manifest["files"]["data.npz"] = {
        "sha1": hashlib.sha1(data_path.read_bytes()).hexdigest(),
        "size": data_path.stat().st_size,
    }

    if extras:
        manifest["extras"] = extras

    man_path = run_dir / "manifest.json"
    man_tmp = run_dir / "manifest.json.tmp"
    man_tmp.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(man_tmp, man_path)


def load_demo_run(run_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    run_dir = Path(run_dir)
    data = np.load(run_dir / "data.npz")
    obs = data["obs"]
    actions = data["actions"]
    rewards = data["rewards"]
    next_obs = data["next_obs"]
    dones = data["dones"]
    timestamps = data["timestamps"]
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    return obs, actions, rewards, next_obs, dones, timestamps, manifest


def validate_demo_run(
    run_dir: Path,
    min_steps: int = 500,
    max_noop_ratio: float = 0.95,
    preview: bool = False,
    preview_seconds: int = 10,
    fps: int = 10,
) -> bool:
    """Проверить базовые свойства датасета. Возвращает True/False и печатает отчёт."""
    from .replay_utils import create_preview_from_obs

    run_dir = Path(run_dir)
    data_path = run_dir / "data.npz"
    man_path = run_dir / "manifest.json"

    if not data_path.exists() or data_path.stat().st_size < 1024:
        print("VALIDATE: FAIL — data.npz отсутствует или слишком мал.")
        return False
    if not man_path.exists():
        print("VALIDATE: FAIL — manifest.json отсутствует.")
        return False

    obs, actions, rewards, next_obs, dones, timestamps, manifest = load_demo_run(run_dir)
    n = len(actions)
    print(f"VALIDATE: steps = {n}")
    if n < min_steps:
        print(f"VALIDATE: FAIL — слишком мало шагов (< {min_steps}).")
        return False

    # Распределение действий
    unique, counts = np.unique(actions, return_counts=True)
    dist = dict(zip(unique.tolist(), counts.tolist()))
    print(f"VALIDATE: action distribution = {dist}")
    if 0 in dist and dist[0] / n > max_noop_ratio:
        print("VALIDATE: FAIL — распределение действий вырождено (слишком много NOOP).")
        return False

    # Проверка, что кадры меняются
    if len(obs) >= 2:
        hashes = {_sha1_bytes(o) for o in [obs[0], obs[len(obs) // 2], obs[-1]]}
        if len(hashes) == 1:
            print("VALIDATE: FAIL — кадры выглядят одинаково (возможно, чёрный экран).")
            return False

    print("VALIDATE: PASS — базовые проверки пройдены.")

    if preview:
        try:
            create_preview_from_obs(
                obs,
                out_path=run_dir / "preview.mp4",
                seconds=preview_seconds,
                fps=fps,
            )
            print("VALIDATE: preview.mp4 создан.")
        except Exception as e:
            print(f"VALIDATE: предупреждение — не удалось создать preview: {e}")

    return True

