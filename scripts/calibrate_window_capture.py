"""
Калибровка window capture: поиск окна по заголовку, замер FPS, сохранение сэмплов, вывод JSON-конфига.
Использование:
  python scripts/calibrate_window_capture.py --title openMSX
  python scripts/calibrate_window_capture.py --title openMSX --dump-frames 3 --out-dir calibrate_out
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from msx_env.capture import _get_window_rect_by_title


def list_windows_with_title(substring: str) -> list[tuple[str, tuple[int, int, int, int]]]:
    """Список (title, (left, top, right, bottom)) для видимых окон с substring в заголовке."""
    if sys.platform != "win32":
        print("List windows is supported on Windows only.")
        return []
    import ctypes
    from ctypes import wintypes
    user32 = ctypes.windll.user32
    results = []
    buf = ctypes.create_unicode_buffer(512)

    def enum_cb(hwnd, _):
        if not user32.IsWindowVisible(hwnd):
            return True
        user32.GetWindowTextW(hwnd, buf, 512)
        title = buf.value
        if substring.lower() in title.lower():
            rect = wintypes.RECT()
            user32.GetWindowRect(hwnd, ctypes.byref(rect))
            results.append((title, (rect.left, rect.top, rect.right, rect.bottom)))
        return True

    WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    user32.EnumWindows(WNDENUMPROC(enum_cb), 0)
    return results


def capture_region_fps(region: tuple[int, int, int, int], num_frames: int = 60) -> float:
    """Захват region, возврат FPS (кадров в секунду)."""
    left, top, right, bottom = region
    w, h = right - left, bottom - top
    if w <= 0 or h <= 0:
        return 0.0
    capturer = None
    use_dxcam = False
    try:
        import dxcam
        capturer = dxcam.create()
        use_dxcam = True
    except Exception:
        try:
            import mss
            capturer = mss.mss()
        except ImportError:
            print("Install dxcam or mss: pip install dxcam  or  pip install mss")
            return 0.0
    t0 = time.perf_counter()
    for _ in range(num_frames):
        if use_dxcam:
            capturer.grab(region=region)
        else:
            capturer.grab({"left": left, "top": top, "width": w, "height": h})
    elapsed = time.perf_counter() - t0
    if not use_dxcam:
        try:
            capturer.close()
        except Exception:
            pass
    return num_frames / elapsed if elapsed > 0 else 0.0


def grab_region_rgb(region: tuple[int, int, int, int]) -> "np.ndarray | None":
    """Один кадр region как (H,W,3) uint8 RGB."""
    left, top, right, bottom = region
    w, h = right - left, bottom - top
    if w <= 0 or h <= 0:
        return None
    import numpy as np
    try:
        import dxcam
        cam = dxcam.create()
        frame = cam.grab(region=region)
        return np.asarray(frame, dtype=np.uint8) if frame is not None else None
    except Exception:
        pass
    try:
        import mss
        with mss.mss() as sct:
            shot = sct.grab({"left": left, "top": top, "width": w, "height": h})
            arr = np.array(shot, dtype=np.uint8)
            if arr.ndim == 3 and arr.shape[2] == 4:
                arr = arr[:, :, [2, 1, 0]]
            return arr
    except Exception:
        return None


def main() -> None:
    p = argparse.ArgumentParser(description="Calibrate window capture for openMSX")
    p.add_argument("--title", default="openMSX", help="Substring in window title")
    p.add_argument("--dump-frames", type=int, default=0, help="Save N sample frames to disk")
    p.add_argument("--out-dir", type=str, default=None, help="Directory for dumped frames and config snippet")
    p.add_argument("--fps-frames", type=int, default=60, help="Frames to measure FPS")
    args = p.parse_args()

    candidates = list_windows_with_title(args.title)
    if not candidates:
        print(f"No windows found with title containing '{args.title}'.")
        print("Start openMSX first, or use another --title.")
        sys.exit(1)

    print(f"Found {len(candidates)} window(s):")
    for i, (title, (left, top, right, bottom)) in enumerate(candidates):
        w, h = right - left, bottom - top
        print(f"  [{i}] {title!r}  rect=({left}, {top}, {right}, {bottom})  size={w}x{h}")

    choice = candidates[0]
    if len(candidates) > 1:
        idx = input("Enter index to use [0]: ").strip() or "0"
        try:
            choice = candidates[int(idx)]
        except (ValueError, IndexError):
            choice = candidates[0]
    title, rect = choice
    left, top, right, bottom = rect
    w, h = right - left, bottom - top
    print(f"Using rect=({left}, {top}, {right}, {bottom})  size={w}x{h}")

    print("Measuring capture FPS...")
    fps = capture_region_fps(rect, num_frames=args.fps_frames)
    print(f"Capture FPS: {fps:.1f}")

    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "calibrate_out"
    if args.dump_frames > 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        from PIL import Image
        for i in range(args.dump_frames):
            frame = grab_region_rgb(rect)
            if frame is not None:
                path = out_dir / f"calibrate_frame_{i:02d}.png"
                Image.fromarray(frame).save(path)
                print(f"Saved {path}")
            time.sleep(0.05)

    config = {
        "capture_backend": "window",
        "window_title": args.title,
        "window_crop": [left, top, w, h],
        "capture_lag_ms": 0,
    }
    snippet = json.dumps(config, indent=2)
    print("\nConfig snippet (EnvConfig / CLI):")
    print(snippet)
    if args.dump_frames > 0 or args.out_dir:
        config_path = out_dir / "window_capture_config.json"
        config_path.write_text(snippet, encoding="utf-8")
        print(f"Written to {config_path}")


if __name__ == "__main__":
    main()
