from pathlib import Path
from typing import Sequence

import imageio.v2 as imageio
import numpy as np


def create_preview_from_obs(
    obs: np.ndarray,
    out_path: Path,
    seconds: int = 10,
    fps: int = 10,
) -> None:
    """Создать небольшой mp4 из массива obs (HWC или CHW, uint8)."""
    out_path = Path(out_path)
    n_frames = min(len(obs), seconds * fps)
    if n_frames == 0:
        raise ValueError("Нет кадров для превью")

    frames: Sequence[np.ndarray] = []
    for i in np.linspace(0, len(obs) - 1, n_frames, dtype=int):
        frame = obs[i]
        if frame.ndim == 3 and frame.shape[0] in (1, 3):  # CHW -> HWC
            frame = np.transpose(frame, (1, 2, 0))
        if frame.ndim == 2:  # HW -> HWC grayscale
            frame = np.stack([frame] * 3, axis=-1)
        frames.append(frame)

    imageio.mimwrite(out_path, frames, fps=fps)

