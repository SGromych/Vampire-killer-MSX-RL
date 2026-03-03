"""
Проверка детектора STAGE: разбор номера этапа из HUD.
Запуск без env: передать путь к скриншоту. С env: сделать N шагов и вывести stage/stage_conf на каждом.

Пример:
  python scripts/debug_stage_detector.py --dump 20
  python scripts/debug_stage_detector.py --image path/to/frame.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from msx_env.hud_parser import parse_stage, STAGE_DIGITS_ROI


def main() -> None:
    p = argparse.ArgumentParser(description="Debug stage detector (HUD STAGE 00, 01, ...)")
    p.add_argument("--image", type=str, default=None, help="путь к скриншоту (PNG) для разбора")
    p.add_argument("--dump", type=int, default=0, help="запустить env, сделать N шагов, вывести stage/conf на каждом")
    p.add_argument("--capture-backend", choices=["png", "single", "window"], default="png")
    args = p.parse_args()

    if args.image:
        path = Path(args.image)
        if not path.exists():
            print(f"Файл не найден: {path}")
            return
        stage, conf = parse_stage(path)
        print(f"STAGE={stage} confidence={conf:.3f} ROI={STAGE_DIGITS_ROI}")
        return

    if args.dump <= 0:
        print("Укажите --image <path> или --dump N")
        return

    from msx_env.env import EnvConfig, VampireKillerEnv
    from msx_env.reward import default_v1_config

    rom = ROOT / "VAMPIRE.ROM"
    if not rom.exists():
        print(f"ROM не найден: {rom}")
        return
    workdir = str(ROOT / "checkpoints" / "ppo" / "run")
    Path(workdir).mkdir(parents=True, exist_ok=True)

    env = VampireKillerEnv(EnvConfig(
        rom_path=str(rom),
        workdir=workdir,
        frame_size=(84, 84),
        capture_backend=args.capture_backend,
        reward_config=default_v1_config(),
    ))

    obs, _ = env.reset()
    for step in range(args.dump):
        obs, _, term, trunc, info = env.step(0)
        stage = info.get("stage", 0)
        conf = info.get("stage_conf", 0.0)
        print(f"step={step} stage={stage} stage_conf={conf:.3f}")
        if term or trunc:
            break
    env.close()
    print("Done.")


if __name__ == "__main__":
    main()
