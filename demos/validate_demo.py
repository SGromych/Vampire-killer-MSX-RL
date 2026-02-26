import argparse
from pathlib import Path
import sys

# Добавляем корень проекта в sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from msx_env.dataset import validate_demo_run


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate recorded Vampire Killer demo run.")
    p.add_argument("run_id", help="имя подкаталога в demos/runs/")
    p.add_argument("--preview", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base_dir = Path(__file__).resolve().parent.parent
    run_dir = base_dir / "demos" / "runs" / args.run_id
    ok = validate_demo_run(run_dir, preview=args.preview)
    if ok:
        print("VALIDATE_DEMO: PASS")
    else:
        print("VALIDATE_DEMO: FAIL")


if __name__ == "__main__":
    main()

