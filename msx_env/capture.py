"""
Модульный бэкенд захвата кадров для RL env.
Интерфейс: start(), grab() -> np.ndarray (H,W,3) RGB, close().
Формат obs после препроцессинга (resize, grayscale) остаётся (84, 84) uint8.
"""
from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Tuple

import numpy as np

if TYPE_CHECKING:
    from openmsx_bridge import OpenMSXFileControl

logger = logging.getLogger(__name__)


def _get_window_rect_by_pid(pid: int) -> Tuple[int, int, int, int] | None:
    """Windows: (left, top, right, bottom) окна по PID или None."""
    if os.name != "nt":
        return None
    try:
        import ctypes
        from ctypes import wintypes
        user32 = ctypes.windll.user32
        result = [None]

        def enum_cb(hwnd, _):
            if not user32.IsWindowVisible(hwnd):
                return True
            p = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(p))
            if p.value == pid:
                result[0] = hwnd
                return False
            return True

        WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        user32.EnumWindows(WNDENUMPROC(enum_cb), 0)
        hwnd = result[0]
        if hwnd is None:
            return None
        rect = wintypes.RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(rect))
        return (rect.left, rect.top, rect.right, rect.bottom)
    except Exception:
        return None


def _bring_window_to_foreground(pid: int) -> bool:
    """Windows: вывести окно процесса на передний план (для захвата при 2+ env). Возвращает True если нашли окно."""
    if os.name != "nt" or pid is None:
        return False
    try:
        import ctypes
        from ctypes import wintypes
        user32 = ctypes.windll.user32
        result = [None]

        def enum_cb(hwnd, _):
            if not user32.IsWindowVisible(hwnd):
                return True
            p = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(p))
            if p.value == pid:
                result[0] = hwnd
                return False
            return True

        WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        user32.EnumWindows(WNDENUMPROC(enum_cb), 0)
        hwnd = result[0]
        if hwnd is None:
            return False
        user32.SetForegroundWindow(hwnd)
        time.sleep(0.03)
        return True
    except Exception:
        return False


def _get_window_rect_by_title(substring: str) -> Tuple[int, int, int, int] | None:
    """Windows: (left, top, right, bottom) первого видимого окна с substring в заголовке."""
    if os.name != "nt" or not substring:
        return None
    try:
        import ctypes
        from ctypes import wintypes
        user32 = ctypes.windll.user32
        result = [None]
        buf = ctypes.create_unicode_buffer(256)

        def enum_cb(hwnd, _):
            if not user32.IsWindowVisible(hwnd):
                return True
            user32.GetWindowTextW(hwnd, buf, 256)
            if substring.lower() in buf.value.lower():
                result[0] = hwnd
                return False
            return True

        WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        user32.EnumWindows(WNDENUMPROC(enum_cb), 0)
        hwnd = result[0]
        if hwnd is None:
            return None
        rect = wintypes.RECT()
        user32.GetWindowRect(hwnd, ctypes.byref(rect))
        return (rect.left, rect.top, rect.right, rect.bottom)
    except Exception:
        return None


class FrameCaptureBackend(ABC):
    """Интерфейс бэкенда захвата. grab() возвращает RGB (H, W, 3) uint8."""

    @abstractmethod
    def start(self) -> None:
        """Инициализация (опционально)."""
        pass

    @abstractmethod
    def grab(self) -> np.ndarray:
        """Захват кадра. Возвращает (H, W, 3) uint8 RGB, полное разрешение экрана."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Освобождение ресурсов."""
        pass

    def __enter__(self) -> "FrameCaptureBackend":
        self.start()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


class FileCapturePNG(FrameCaptureBackend):
    """
    PNG через openMSX screenshot; каждый раз в один и тот же файл.
    Используется для отладки и скриптов; в training path по умолчанию используется dxcam.
    """

    def __init__(
        self,
        emu: "OpenMSXFileControl",
        workdir: str | Path,
        filename: str = "step_frame.png",
    ):
        self._emu = emu
        self._workdir = Path(workdir)
        self._filename = filename

    def start(self) -> None:
        pass

    def grab(self) -> np.ndarray:
        import io
        import time
        from PIL import Image

        self._emu.screenshot(self._filename)
        img_path = self._workdir / self._filename
        last_err = None
        for attempt in range(5):
            try:
                time.sleep(0.05 * (attempt + 1))
                data = img_path.read_bytes()
                if not data:
                    raise OSError("screenshot file is empty")
                buf = io.BytesIO(data)
                img = Image.open(buf)
                img.load()
                rgb = img.convert("RGB")
                return np.array(rgb, dtype=np.uint8)
            except OSError as e:
                last_err = e
                if attempt < 4:
                    continue
                try:
                    size = img_path.stat().st_size if img_path.exists() else -1
                    head = img_path.read_bytes()[:12] if img_path.exists() else b""
                    magic = head.hex() if len(head) >= 4 else "?"
                except Exception:
                    size, magic = -1, "?"
                raise RuntimeError(
                    "Failed to read screenshot after 5 attempts: path=%s size=%s magic_hex=%s last_error=%s"
                    % (img_path, size, magic, last_err)
                ) from last_err

    def close(self) -> None:
        pass


class DxcamCapture(FrameCaptureBackend):
    """
    Прямой захват области экрана (окно openMSX) через dxcam в память.
    Без записи/чтения PNG. Полный кадр окна (включая HUD), без дополнительного crop.
    При ошибке инициализации или захвата — исключение, без fallback.
    """

    def __init__(
        self,
        emu: "OpenMSXFileControl",
        workdir: str | Path,
        filename: str = "step_frame.png",
        *,
        window_title_substring: str = "openMSX",
        pid: int | None = None,
        capture_lag_ms: float = 0,
    ):
        self._emu = emu
        self._workdir = Path(workdir)
        self._filename = filename
        self._window_title = window_title_substring
        self._pid = pid or (getattr(emu, "proc", None) and getattr(emu.proc, "pid", None))
        self._capture_lag_ms = max(0.0, float(capture_lag_ms))
        self._region: Tuple[int, int, int, int] | None = None
        self._camera = None

    def start(self) -> None:
        if os.name != "nt":
            raise RuntimeError("DxcamCapture is only supported on Windows (dxcam requires it).")
        # Окно openMSX может появиться с задержкой после старта процесса,
        # поэтому несколько раз пробуем найти его по PID/заголовку.
        self._region = None
        deadline = time.time() + 10.0
        last_err: str | None = None
        while self._region is None and time.time() < deadline:
            region = None
            if self._pid is not None:
                region = _get_window_rect_by_pid(self._pid)
            if region is None and self._window_title:
                region = _get_window_rect_by_title(self._window_title)
            if region is not None:
                self._region = region
                break
            last_err = f"pid={self._pid} title_substring={self._window_title!r}"
            time.sleep(0.2)
        if self._region is None:
            raise RuntimeError(
                "DxcamCapture: could not find window (by pid/title). "
                "Ensure openMSX is running and visible. "
                f"Last attempt: {last_err}"
            )
        left, top, right, bottom = self._region
        if right <= left or bottom <= top:
            raise RuntimeError(
                "DxcamCapture: invalid window region (left=%s top=%s right=%s bottom=%s)."
                % (left, top, right, bottom)
            )
        try:
            import dxcam
            self._camera = dxcam.create(output_color="RGB")
        except ImportError as e:
            raise RuntimeError(
                "DxcamCapture: dxcam is not installed. Install with: pip install dxcam"
            ) from e
        except Exception as e:
            raise RuntimeError("DxcamCapture: dxcam.create() failed: %s" % e) from e
        logger.info(
            "[capture] dxcam capture enabled. Region (left, top, right, bottom)=%s. "
            "PNG disk capture is disabled for this backend.",
            self._region,
        )

    def grab(self) -> np.ndarray:
        if self._camera is None or self._region is None:
            raise RuntimeError("DxcamCapture: not started or window not found.")
        if self._capture_lag_ms > 0:
            time.sleep(self._capture_lag_ms / 1000.0)
        # При 2+ env выводим окно на передний план, иначе dxcam может захватить другое окно
        if self._pid is not None:
            _bring_window_to_foreground(self._pid)
        # При нескольких env окно может быть перекрыто; даём больше попыток и пауз
        max_attempts = 25
        for attempt in range(max_attempts):
            frame = self._camera.grab(region=self._region)
            if frame is not None:
                return np.asarray(frame, dtype=np.uint8)
            time.sleep(0.02 * (attempt + 1))
        raise RuntimeError(
            "DxcamCapture: grab() returned None repeatedly (no new frame). "
            "Check that the openMSX window is visible and not minimized (with 2 env, place windows side-by-side)."
        )

    def close(self) -> None:
        self._camera = None


class FileCaptureSinglePath(FileCapturePNG):
    """
    То же что FileCapturePNG: overwrite одного файла step_frame.png.
    Алиас для явного указания режима (один путь, без создания новых файлов).
    """

    pass


class WindowCapture(FrameCaptureBackend):
    """
    Захват из области экрана (окно openMSX или ручной crop).
    dxcam (Windows) или mss в качестве fallback. При ошибке — fallback на FileCapturePNG.
    """

    def __init__(
        self,
        emu: "OpenMSXFileControl",
        workdir: str | Path,
        filename: str = "step_frame.png",
        *,
        crop_rect: Tuple[int, int, int, int] | None = None,
        window_title_substring: str = "openMSX",
        pid: int | None = None,
        capture_lag_ms: float = 0,
        fallback_to_file: bool = True,
    ):
        self._emu = emu
        self._workdir = Path(workdir)
        self._filename = filename
        self._crop_rect = crop_rect
        self._window_title = window_title_substring
        self._pid = pid or (getattr(emu, "proc", None) and getattr(emu.proc, "pid", None))
        self._capture_lag_ms = max(0.0, float(capture_lag_ms))
        self._fallback_to_file = fallback_to_file
        self._region: Tuple[int, int, int, int] | None = None
        self._capturer = None
        self._use_dxcam = False
        self._file_fallback: FileCapturePNG | None = FileCapturePNG(emu, workdir, filename) if fallback_to_file else None
        self._last_fallback_warn = False

    def start(self) -> None:
        region = self._crop_rect
        if region is not None and len(region) == 4:
            x, y, a, b = region
            if a > 0 and b > 0:
                self._region = (x, y, x + a, y + b)
            else:
                self._region = (x, y, a, b)
        if self._region is None and self._pid is not None:
            self._region = _get_window_rect_by_pid(self._pid)
        if self._region is None and self._window_title:
            self._region = _get_window_rect_by_title(self._window_title)
        if self._region is None:
            if self._fallback_to_file:
                if not self._last_fallback_warn:
                    logger.warning("[WindowCapture] Window not found, using file backend fallback.")
                    self._last_fallback_warn = True
                return
            raise RuntimeError(
                "WindowCapture: could not find window (by pid or title). "
                "Set window_crop (x,y,w,h) or run calibrate_window_capture.py."
            )
        left, top, right, bottom = self._region
        if right <= left or bottom <= top:
            if self._fallback_to_file:
                if not self._last_fallback_warn:
                    logger.warning("[WindowCapture] Invalid region, using file backend fallback.")
                    self._last_fallback_warn = True
                return
            raise RuntimeError("WindowCapture: invalid region (right<=left or bottom<=top).")
        try:
            import dxcam
            self._capturer = dxcam.create()
            self._use_dxcam = True
        except Exception:
            try:
                import mss
                self._capturer = mss.mss()
                self._use_dxcam = False
            except ImportError as e:
                if self._fallback_to_file:
                    if not self._last_fallback_warn:
                        logger.warning("[WindowCapture] dxcam/mss not available (%s), using file backend.", e)
                        self._last_fallback_warn = True
                    self._capturer = None
                    return
                raise RuntimeError(
                    "WindowCapture: install dxcam or mss (pip install dxcam or pip install mss)."
                ) from e

    def grab(self) -> np.ndarray:
        if self._capture_lag_ms > 0:
            time.sleep(self._capture_lag_ms / 1000.0)
        if self._capturer is None and self._file_fallback is not None:
            return self._file_fallback.grab()
        if self._region is None:
            if self._file_fallback is not None:
                return self._file_fallback.grab()
            raise RuntimeError("WindowCapture: not started or window not found.")
        left, top, right, bottom = self._region
        try:
            if self._use_dxcam:
                frame = self._capturer.grab(region=self._region)
                if frame is None:
                    raise RuntimeError("dxcam.grab returned None")
                return np.asarray(frame, dtype=np.uint8)
            else:
                mon = {"left": left, "top": top, "width": right - left, "height": bottom - top}
                shot = self._capturer.grab(mon)
                if shot is None:
                    raise RuntimeError("mss.grab returned None")
                arr = np.array(shot, dtype=np.uint8)
                if arr.ndim == 3 and arr.shape[2] == 4:
                    arr = arr[:, :, [2, 1, 0]]
                return arr
        except Exception as e:
            if self._file_fallback is not None:
                if not self._last_fallback_warn:
                    logger.warning("[WindowCapture] Capture failed (%s), using file backend.", e)
                    self._last_fallback_warn = True
                return self._file_fallback.grab()
            raise

    def close(self) -> None:
        if self._capturer is not None and not self._use_dxcam:
            try:
                self._capturer.close()
            except Exception:
                pass
        self._capturer = None


def make_capture_backend(
    backend_name: str,
    emu: "OpenMSXFileControl",
    workdir: str | Path,
    filename: str = "step_frame.png",
    *,
    window_crop: Tuple[int, int, int, int] | None = None,
    window_title: str | None = None,
    capture_lag_ms: float = 0,
) -> FrameCaptureBackend:
    """Фабрика бэкендов. backend_name: 'png' | 'single' | 'window' | 'dxcam'."""
    if backend_name in ("png", "single"):
        return FileCapturePNG(emu, workdir, filename)
    if backend_name == "dxcam":
        pid = getattr(emu, "proc", None) and getattr(emu.proc, "pid", None)
        return DxcamCapture(
            emu,
            workdir,
            filename,
            window_title_substring=window_title or "openMSX",
            pid=pid,
            capture_lag_ms=capture_lag_ms,
        )
    if backend_name == "window":
        pid = getattr(emu, "proc", None) and getattr(emu.proc, "pid", None)
        return WindowCapture(
            emu,
            workdir,
            filename,
            crop_rect=window_crop,
            window_title_substring=window_title or "openMSX",
            pid=pid,
            capture_lag_ms=capture_lag_ms,
            fallback_to_file=True,
        )
    raise ValueError(f"Unknown capture backend: {backend_name}. Use: png | single | window | dxcam")
