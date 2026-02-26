"""Вывести список окон и PID — для отладки поиска openMSX."""
import sys

try:
    import pygetwindow as gw
except ImportError:
    print("pip install pygetwindow")
    sys.exit(1)

if sys.platform == "win32":
    import ctypes
    from ctypes import wintypes
    user32 = ctypes.windll.user32

    def get_pid(hwnd):
        p = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(p))
        return p.value
else:
    def get_pid(hwnd):
        return 0

print("Visible windows (title, pid):")
for w in gw.getAllWindows():
    if w.visible and w.title.strip():
        pid = get_pid(w._hWnd) if hasattr(w, '_hWnd') else "?"
        print(f"  pid={pid}  {w.title!r}")
