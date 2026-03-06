import subprocess
import time
import uuid
import os
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import pyautogui
    _HOST_INPUT_AVAILABLE = True
except ImportError:
    _HOST_INPUT_AVAILABLE = False

try:
    import numpy as np
except ImportError:
    np = None

try:
    import pygetwindow
    _PYGETWINDOW_AVAILABLE = True
except ImportError:
    _PYGETWINDOW_AVAILABLE = False

if os.name == "nt":
    import ctypes
    from ctypes import wintypes
    _WIN_USER32 = ctypes.windll.user32
else:
    _WIN_USER32 = None

OPENMSX_EXE = r"C:\Program Files\openMSX\openmsx.exe"  # поправь если нужно

# Маппинг для press_host: MSX key -> pyautogui key
_HOST_KEYMAP = {
    "SPACE": "space",
    "LEFT": "left",
    "RIGHT": "right",
    "UP": "up",
    "DOWN": "down",
    "RET": "enter",
}


def _posix(p: Path) -> str:
    return str(p.resolve()).replace("\\", "/")


def _kill_orphan_openmsx(workdir: Path) -> None:
    """
    Убить старый openMSX, запущенный в этом workdir (PID в workdir/openmsx.pid).
    Нужно перед стартом нового процесса, чтобы не накапливались «забытые» openMSX.
    """
    pid_file = Path(workdir) / "openmsx.pid"
    if not pid_file.exists():
        return
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        return
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/PID", str(pid)],
            capture_output=True,
            timeout=5,
        )
    else:
        try:
            os.kill(pid, 9)
        except (ProcessLookupError, PermissionError):
            pass
    try:
        pid_file.unlink(missing_ok=True)
    except OSError:
        pass


class OpenMSXFileControl:
    """
    Windows-stable file-based control for openMSX.

    - Starts openMSX with -script bootstrap.tcl
    - bootstrap.tcl polls commands.tcl and sources it
    - Python writes commands.tcl atomically (tmp -> os.replace)
    - commands.tcl writes reply to ABSOLUTE reply path embedded as literal
    - screenshot uses 'screenshot <filename>' (supported by your build)
    - input uses keymatrixdown/up <row> <bitmask> (supported by your build)
    - press() uses 'after realtime' callback for key hold timing
    """

    # MSX key matrix (international): row 8 arrows+SPACE, row 7 RET
    _KEYMAP = {
        "SPACE": (8, 1 << 0),
        "LEFT":  (8, 1 << 4),
        "UP":    (8, 1 << 5),
        "DOWN":  (8, 1 << 6),
        "RIGHT": (8, 1 << 7),
        "RET":   (7, 1 << 7),  # Enter
    }

    def __init__(
        self,
        rom_path: str,
        workdir: str = ".",
        poll_ms: int = 20,
        boot_timeout_s: float = 20.0,
        log_dir: str | Path | None = None,
        instance_id: int | str | None = None,
    ):
        self.workdir = Path(workdir).resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)

        self.rom_path = _posix(Path(rom_path))

        self.commands_tcl = self.workdir / "commands.tcl"
        self.commands_tmp = self.workdir / "commands.tcl.tmp"
        self.reply_txt = self.workdir / "reply.txt"
        self.bootstrap_tcl = self.workdir / "bootstrap.tcl"

        self.poll_ms = int(poll_ms)

        self.reply_abs = _posix(self.reply_txt)
        self.cmd_abs = _posix(self.commands_tcl)

        self._write_bootstrap()

        # clear files
        self.commands_tcl.write_text("", encoding="utf-8")
        self.reply_txt.write_text("", encoding="utf-8")

        # kill orphan openMSX from previous run (PID in workdir/openmsx.pid)
        _kill_orphan_openmsx(self.workdir)

        # log files: if log_dir + instance_id given, one file per instance under log_dir; else workdir
        if log_dir is not None and instance_id is not None:
            log_dir_path = Path(log_dir).resolve()
            log_dir_path.mkdir(parents=True, exist_ok=True)
            self.log_out_path = log_dir_path / f"openmsx_{instance_id}.log"
            self.log_err_path = self.log_out_path
            self._log_out = open(self.log_out_path, "w", encoding="utf-8", errors="ignore")
            self._log_err = self._log_out
        else:
            self.log_out_path = self.workdir / "openmsx_stdout.log"
            self.log_err_path = self.workdir / "openmsx_stderr.log"
            self._log_out = open(self.log_out_path, "w", encoding="utf-8", errors="ignore")
            self._log_err = open(self.log_err_path, "w", encoding="utf-8", errors="ignore")

        self.proc = subprocess.Popen(
            [OPENMSX_EXE, "-script", str(self.bootstrap_tcl)],
            cwd=str(self.workdir),
            stdout=self._log_out,
            stderr=self._log_err,
        )
        (self.workdir / "openmsx.pid").write_text(str(self.proc.pid), encoding="utf-8")

        self._wait_boot(boot_timeout_s)

    # -------------------------
    # Bootstrap + low-level IO
    # -------------------------

    def _write_bootstrap(self) -> None:
        # FIX 1: correct catch handling via return code,
        # not via "err string is non-empty"
        tcl = f"""
# bootstrap.tcl (diagnostic + polling)

set CMD_FILE "{self.cmd_abs}"

proc __write_reply {{msg}} {{
    set f [open {{{self.reply_abs}}} w]
    puts $f $msg
    close $f
}}

set __poll_count 0
set __last_status_ms 0

proc __poll_commands {{}} {{
    global CMD_FILE
    global __poll_count
    global __last_status_ms

    incr __poll_count

    # Heartbeat every ~500ms — в bootstrap_status, НЕ в reply.txt (не затирать ответы)
    set now [clock milliseconds]
    if {{$__last_status_ms == 0 || ($now - $__last_status_ms) > 500}} {{
        set exists [file exists $CMD_FILE]
        set size 0
        if {{$exists}} {{ set size [file size $CMD_FILE] }}
        if {{[catch {{set stf [open "bootstrap_status.txt" w]; puts $stf "ok:bootstrap poll=$__poll_count cmd_exists=$exists cmd_size=$size"; close $stf}}] == 0}} {{}}
        set __last_status_ms $now
    }}

    if {{[file exists $CMD_FILE] && [file size $CMD_FILE] > 0}} {{
        set err ""
        set code [catch {{source $CMD_FILE}} err]
        if {{$code != 0}} {{
            __write_reply "err:source poll=$__poll_count msg=$err"
        }}
        # truncate after exec
        set f [open $CMD_FILE w]
        close $f
    }}

    after {self.poll_ms} __poll_commands
}}

__write_reply "ok:bootstrap starting"
after {self.poll_ms} __poll_commands

# Load ROM
carta {{{self.rom_path}}}

# Настройка джойстика — в catch, чтобы ошибка не ломала загрузку
if {{[catch {{
    plug joyporta msxjoystick1
    dict set msxjoystick1_config LEFT {{keyb LEFT}}
    dict set msxjoystick1_config RIGHT {{keyb RIGHT}}
    dict set msxjoystick1_config UP {{keyb UP}}
    dict set msxjoystick1_config DOWN {{keyb DOWN}}
    dict set msxjoystick1_config A {{keyb SPACE}}
    dict set msxjoystick1_config B {{keyb LCTRL}}
}} err] != 0}} {{
    # игнорируем — картридж уже загружен
}}
"""
        self.bootstrap_tcl.write_text(tcl.strip() + "\n", encoding="utf-8")

    def _atomic_write_commands(self, content: str) -> None:
        self.commands_tmp.write_text(content, encoding="utf-8")
        # На Windows openMSX может держать commands.tcl открытым при опросе — retry при PermissionError
        for attempt in range(5):
            try:
                os.replace(str(self.commands_tmp), str(self.commands_tcl))
                return
            except PermissionError:
                if attempt < 4:
                    time.sleep(0.05 * (attempt + 1))
                else:
                    raise

    def _new_rid(self) -> str:
        return uuid.uuid4().hex[:10]

    def _read_reply(self) -> str:
        try:
            return self.reply_txt.read_text(encoding="utf-8", errors="ignore").strip()
        except Exception:
            return ""

    def _wait_contains(self, needle: str, timeout_s: float, file: Path | None = None) -> str:
        f = file or self.reply_txt
        deadline = time.time() + timeout_s
        last = ""
        while time.time() < deadline:
            try:
                txt = f.read_text(encoding="utf-8", errors="ignore").strip()
            except Exception:
                txt = ""
            last = txt
            if needle in txt:
                return txt
            time.sleep(0.05)
        raise TimeoutError(
            f"{f.name} did not contain '{needle}' within {timeout_s:.1f}s. Last='{last}'"
        )

    def _wait_reply_rid(self, rid: str, timeout_s: float) -> str:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            txt = self._read_reply()
            if txt.startswith("err:source"):
                raise RuntimeError(f"Tcl source error reported: {txt}")
            if txt and f"RID={rid}" in txt:
                return txt
            time.sleep(0.01)
        raise TimeoutError(
            f"No reply for RID={rid} within {timeout_s:.2f}s. "
            f"Last reply.txt='{self._read_reply()}'. Check {self.log_err_path.name}."
        )

    def _send_tcl_with_replyproc(self, tcl_body: str) -> None:
        header = f"""
proc __reply {{msg}} {{
    set f [open {{{self.reply_abs}}} w]
    puts $f $msg
    close $f
}}
"""
        full = header + "\n" + tcl_body.strip() + "\n"
        self._atomic_write_commands(full)

    def _wait_boot(self, timeout_s: float) -> None:
        self._wait_contains("ok:bootstrap", timeout_s)
        self._wait_contains("poll=", timeout_s, file=self.workdir / "bootstrap_status.txt")
        self.ping(timeout_s=2.0)
        if self.proc.poll() is not None:
            raise RuntimeError(
                "openMSX process exited during boot (code=%s). workdir=%s Check log: %s"
                % (self.proc.returncode, self.workdir, self.log_err_path)
            )

    # -------------------------
    # Public API
    # -------------------------

    def ping(self, timeout_s: float = 1.0) -> str:
        rid = self._new_rid()
        self._send_tcl_with_replyproc(f"""
__reply "RID={rid} ok:ping t=[clock milliseconds]"
""")
        return self._wait_reply_rid(rid, timeout_s)

    def screenshot(self, filename: str = "frame.png", timeout_s: float = 3.0) -> str:
        """
        Your build supports: screenshot <filename>
        """
        rid = self._new_rid()
        fn = _posix(self.workdir / filename)
        self._send_tcl_with_replyproc(f"""
screenshot {{{fn}}}
__reply "RID={rid} ok:screenshot file={fn} t=[clock milliseconds]"
""")
        out = self._wait_reply_rid(rid, timeout_s)
        time.sleep(0.05)
        return out

    def press(self, key: str, hold_ms: int = 120, timeout_s: float = None) -> str:
        """
        Press one of: SPACE, LEFT, RIGHT, UP, DOWN
        via keymatrixdown/up <row> <bitmask>.

        Uses after realtime callback: keydown -> wait hold_ms (host sec) -> keyup -> reply.
        openMSX "after N" does NOT block; keyup would run immediately. Use realtime callback.
        """
        key = key.upper().strip()
        if key not in self._KEYMAP:
            raise ValueError(f"Unsupported key '{key}'. Supported: {sorted(self._KEYMAP.keys())}")

        row, mask = self._KEYMAP[key]
        rid = self._new_rid()
        hold_ms = int(hold_ms)
        hold_sec = hold_ms / 1000.0
        if timeout_s is None:
            timeout_s = max(5.0, hold_sec + 2.0)  # ответ приходит после hold

        # Callback: keyup + reply. catch чтобы reply всегда ушёл даже при ошибке keymatrixup.
        callback = f'catch {{ keymatrixup {row} {mask} }}; __reply "RID={rid} ok:press key={key} row={row} mask={mask} hold_ms={hold_ms} t=[clock milliseconds]"'
        callback_escaped = callback.replace('"', r'\"')

        self._send_tcl_with_replyproc(f"""
keymatrixdown {row} {mask}
after realtime {hold_sec} "{callback_escaped}"
""")
        return self._wait_reply_rid(rid, timeout_s)

    def press_repeat(
        self, key: str, times: int = 10, press_ms: int = 80, gap_ms: int = 60
    ) -> None:
        """
        Серия коротких нажатий — для игр, реагирующих на key repeat.
        """
        for _ in range(times):
            self.press(key, hold_ms=press_ms)
            if gap_ms > 0:
                time.sleep(gap_ms / 1000.0)

    def press_type(self, key: str, timeout_s: float = 2.0) -> str:
        """
        Нажать через команду openMSX `type` — альтернатива keymatrix.
        type "simulates typing" и всегда должен работать (docs).
        -release: клавиши отпускаются перед новой — нужно некоторым играм.
        Поддерживает: SPACE (пробел), RET (\\r), остальное — keymatrix.
        """
        key = key.upper().strip()
        if key == "SPACE":
            cmd = 'type -release " "'
        elif key == "RET":
            cmd = 'type -release "\\r"'
        else:
            return self.press(key, timeout_s=timeout_s)
        rid = self._new_rid()
        self._send_tcl_with_replyproc(f"""
{cmd}
__reply "RID={rid} ok:type key={key} t=[clock milliseconds]"
""")
        return self._wait_reply_rid(rid, timeout_s)

    def press_combo(
        self,
        keys: list[str],
        hold_ms: int = 150,
        timeout_s: float = 2.0,
    ) -> str:
        """
        Одновременное нажатие нескольких клавиш через keymatrix.
        Работает только с клавишами из _KEYMAP (SPACE, LEFT, RIGHT, UP, DOWN, RET).
        """
        if not keys:
            return "ok:combo empty"

        rows: dict[int, int] = {}
        norm_keys: list[str] = []
        for k in keys:
            k = k.upper().strip()
            if k not in self._KEYMAP:
                raise ValueError(
                    f"Unsupported key '{k}' in press_combo. Supported: {sorted(self._KEYMAP.keys())}"
                )
            row, mask = self._KEYMAP[k]
            rows[row] = rows.get(row, 0) | mask
            norm_keys.append(k)

        rid = self._new_rid()
        hold_ms = int(hold_ms)
        hold_sec = hold_ms / 1000.0
        if timeout_s is None:
            timeout_s = max(5.0, hold_sec + 2.0)

        downs = "\n".join(f"keymatrixdown {r} {m}" for r, m in rows.items())
        ups = "; ".join(f"catch {{ keymatrixup {r} {m} }}" for r, m in rows.items())
        combo_name = "+".join(norm_keys)
        callback = (
            f'{ups}; __reply "RID={rid} ok:combo keys={combo_name} hold_ms={hold_ms} '
            't=[clock milliseconds]"'
        )
        callback_escaped = callback.replace('"', r"\"")

        self._send_tcl_with_replyproc(f"""
{downs}
after realtime {hold_sec} "{callback_escaped}"
""")
        return self._wait_reply_rid(rid, timeout_s)

    def keydown(self, keys: list[str], timeout_s: float = 2.0) -> str:
        """
        Нажать и держать клавиши (без автоотпускания).
        Для непрерывного движения: keydown при смене действия, keyup при NOOP или смене.
        """
        if not keys:
            return "ok:keydown empty"
        rows: dict[int, int] = {}
        for k in keys:
            k = k.upper().strip()
            if k not in self._KEYMAP:
                raise ValueError(
                    f"Unsupported key '{k}'. Supported: {sorted(self._KEYMAP.keys())}"
                )
            r, m = self._KEYMAP[k]
            rows[r] = rows.get(r, 0) | m
        rid = self._new_rid()
        downs = "\n".join(f"keymatrixdown {r} {m}" for r, m in rows.items())
        self._send_tcl_with_replyproc(f"""
{downs}
__reply "RID={rid} ok:keydown t=[clock milliseconds]"
""")
        return self._wait_reply_rid(rid, timeout_s)

    def keyup(self, keys: list[str], timeout_s: float = 2.0) -> str:
        """Отпустить клавиши."""
        if not keys:
            return "ok:keyup empty"
        rows: dict[int, int] = {}
        for k in keys:
            k = k.upper().strip()
            if k not in self._KEYMAP:
                raise ValueError(
                    f"Unsupported key '{k}'. Supported: {sorted(self._KEYMAP.keys())}"
                )
            r, m = self._KEYMAP[k]
            rows[r] = rows.get(r, 0) | m
        rid = self._new_rid()
        ups = "; ".join(f"catch {{ keymatrixup {r} {m} }}" for r, m in rows.items())
        self._send_tcl_with_replyproc(f"""
{ups}
__reply "RID={rid} ok:keyup t=[clock milliseconds]"
""")
        return self._wait_reply_rid(rid, timeout_s)

    # -------------------------
    # Host input (окно эмулятора)
    # -------------------------

    def _find_window_by_pid(self, pid: int):
        """Найти HWND окна по PID процесса (Windows, ctypes)."""
        if _WIN_USER32 is None:
            return None
        result = [None]

        def enum_cb(hwnd, _):
            if not _WIN_USER32.IsWindowVisible(hwnd):
                return True
            p = wintypes.DWORD()
            _WIN_USER32.GetWindowThreadProcessId(hwnd, ctypes.byref(p))
            if p.value == pid:
                result[0] = hwnd
                return False
            return True

        WNDENUMPROC = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        _WIN_USER32.EnumWindows(WNDENUMPROC(enum_cb), 0)
        return result[0]

    def _click_window_center(self, hwnd) -> None:
        """Клик в центр окна для активации (Windows)."""
        if _WIN_USER32 is None or hwnd is None:
            return
        rect = wintypes.RECT()
        _WIN_USER32.GetWindowRect(hwnd, ctypes.byref(rect))
        cx = (rect.left + rect.right) // 2
        cy = (rect.top + rect.bottom) // 2
        pyautogui.click(cx, cy)

    def ensure_focus(self) -> bool:
        """
        Переводит фокус на окно openMSX.
        Сначала по PID, затем по заголовку (pygetwindow).
        Клик в центр окна для надёжной активации.
        """
        if not _HOST_INPUT_AVAILABLE:
            raise RuntimeError("Для ensure_focus нужен pyautogui: pip install pyautogui")
        pyautogui.FAILSAFE = False
        hwnd = None
        if _WIN_USER32 is not None:
            hwnd = self._find_window_by_pid(self.proc.pid)
        if hwnd is not None:
            _WIN_USER32.SetForegroundWindow(hwnd)
            time.sleep(0.1)
            self._click_window_center(hwnd)
            time.sleep(0.15)
            return True
        if _PYGETWINDOW_AVAILABLE:
            for part in ("openMSX", "MSX", "openmsx", "C-BIOS"):
                for w in pygetwindow.getWindowsWithTitle(part):
                    if w.visible and w.title.strip():
                        try:
                            w.activate()
                            time.sleep(0.1)
                            pyautogui.click(w.left + w.width // 2, w.top + w.height // 2)
                            time.sleep(0.15)
                            return True
                        except Exception:
                            pass
        return False

    def press_host(self, key: str, hold_ms: int = 100) -> None:
        """
        Нажать клавишу через окно эмулятора (как реальный пользователь).
        Фокус на openMSX по PID, потом pyautogui.
        """
        if not _HOST_INPUT_AVAILABLE:
            raise RuntimeError("Для press_host нужен pyautogui: pip install pyautogui")
        key = key.upper().strip()
        if key not in _HOST_KEYMAP:
            raise ValueError(f"press_host: ключ '{key}'. Доступны: {list(_HOST_KEYMAP.keys())}")
        ok = self.ensure_focus()
        if not ok:
            raise RuntimeError(
                f"Не удалось найти окно openMSX (pid={self.proc.pid}). "
                "Убедитесь, что окно не свёрнуто."
            )
        pyautogui.FAILSAFE = False
        time.sleep(0.1)
        pyautogui.keyDown(_HOST_KEYMAP[key])
        time.sleep(hold_ms / 1000.0)
        pyautogui.keyUp(_HOST_KEYMAP[key])

    def press_host_combo(self, keys: list[str], hold_ms: int = 150) -> None:
        """
        Одновременное нажатие нескольких клавиш (например UP+RIGHT для прыжка вправо).
        """
        if not _HOST_INPUT_AVAILABLE:
            raise RuntimeError("Для press_host_combo нужен pyautogui: pip install pyautogui")
        pykeys = []
        for k in keys:
            k = k.upper().strip()
            if k not in _HOST_KEYMAP:
                raise ValueError(f"press_host_combo: ключ '{k}'. Доступны: {list(_HOST_KEYMAP.keys())}")
            pykeys.append(_HOST_KEYMAP[k])
        self.ensure_focus()
        time.sleep(0.1)
        for k in pykeys:
            pyautogui.keyDown(k)
        time.sleep(hold_ms / 1000.0)
        for k in reversed(pykeys):
            pyautogui.keyUp(k)

    def get_screen_array(self, filename: str = "frame.png"):
        """
        Скриншот и загрузка как массив для анализа.
        Возвращает (path, PIL.Image) или (path, None) при ошибке.
        """
        path_str = self.screenshot(filename)
        if Image is None:
            return path_str, None
        try:
            img = Image.open(path_str).convert("RGB")
            return path_str, img
        except Exception:
            return path_str, None

    def wait_for_nonblue_screen(
        self,
        timeout_s: float = 30.0,
        check_every_s: float = 0.5,
        blue_ratio_threshold: float = 0.80,
        get_frame=None,
    ) -> str:
        """
        Wait until screen is NOT mostly the BIOS blue screen.

        If get_frame is provided (callable returning RGB array H,W,3), uses it
        in-memory and does not write any screenshot files.
        Otherwise takes screenshots to _bluecheck_*.png and uses Pillow.

        Returns path to the last screenshot that passed, or "" when get_frame was used.
        """
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            ratio = 0.0
            if get_frame is not None and np is not None:
                try:
                    frame = get_frame()
                    if frame is not None and frame.ndim == 3:
                        r, g, b = frame[:, :, 0], frame[:, :, 1], frame[:, :, 2]
                        blue_mask = (b > 150) & (b > r) & (b > g)
                        npx = frame.shape[0] * frame.shape[1]
                        ratio = float(blue_mask.sum()) / npx if npx else 0
                except Exception:
                    ratio = 0
            else:
                if Image is None:
                    raise RuntimeError(
                        "wait_for_nonblue_screen needs Pillow or get_frame with numpy"
                    )
                fn = f"_bluecheck_{int(time.time() * 1000)}.png"
                self.screenshot(fn, timeout_s=5.0)
                path = self.workdir / fn
                if not path.exists():
                    time.sleep(check_every_s)
                    continue
                try:
                    img = Image.open(path).convert("RGB")
                    pixels = list(img.getdata())
                    blue_count = sum(
                        1 for r, g, b in pixels if b > 150 and b > r and b > g
                    )
                    ratio = blue_count / len(pixels) if pixels else 0
                except Exception:
                    ratio = 0
                if ratio >= blue_ratio_threshold:
                    try:
                        path.unlink(missing_ok=True)
                    except OSError:
                        pass

            if ratio < blue_ratio_threshold:
                return "" if (get_frame is not None) else str(path)

            time.sleep(check_every_s)

        raise TimeoutError(
            f"Screen stayed blue for {timeout_s}s (threshold={blue_ratio_threshold})"
        )

    def close(self) -> None:
        try:
            if getattr(self, "proc", None) and self.proc.poll() is None:
                self.proc.terminate()
        except Exception:
            pass
        try:
            self._log_out.close()
            self._log_err.close()
        except Exception:
            pass


# -----------------------------------------------------------------------------
# OpenMSXStdioControl — альтернатива через -control stdio (XML-протокол)
# На Windows может быть нестабильно; если file-based не работает — попробовать.
# Документация: https://openmsx.org/manual/openmsx-control.html
# -----------------------------------------------------------------------------

class OpenMSXStdioControl:
    """
    Управление через -control stdio: XML <command>...</command> в stdin.
    На Windows: openmsx -control pipe (или stdio в зависимости от сборки).
    """

    def __init__(self, rom_path: str, workdir: str = "."):
        self.workdir = Path(workdir).resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)
        rom = _posix(Path(rom_path))
        # -control stdio: без окна по умолчанию, нужен set renderer SDL
        self.proc = subprocess.Popen(
            [OPENMSX_EXE, "-control", "stdio"],
            cwd=str(self.workdir),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        time.sleep(1.0)
        self._read_until_openmsx_output()
        self._send_cmd("set renderer SDL")
        self._send_cmd(f"carta {{{rom}}}")
        time.sleep(2.0)

    def _read_until_openmsx_output(self) -> None:
        """Дать openMSX время на вывод."""
        time.sleep(1.0)

    def _send_cmd(self, cmd: str) -> str:
        xml = f"<command>{cmd}</command>\n"
        self.proc.stdin.write(xml)
        self.proc.stdin.flush()
        time.sleep(0.05)
        out = []
        while True:
            line = self.proc.stdout.readline()
            if not line:
                break
            out.append(line)
            if "<reply" in line:
                break
        return "".join(out)

    def press(self, key: str) -> str:
        """keymatrix через stdio."""
        row, mask = OpenMSXFileControl._KEYMAP.get(
            key.upper(), (8, 1)
        )  # SPACE default
        self._send_cmd(f"keymatrixdown {row} {mask}")
        time.sleep(0.12)
        self._send_cmd(f"keymatrixup {row} {mask}")
        return "ok"

    def screenshot(self, filename: str = "frame.png") -> str:
        fn = _posix(self.workdir / filename)
        self._send_cmd(f"screenshot {{{fn}}}")
        return str(self.workdir / filename)

    def close(self) -> None:
        if getattr(self, "proc", None) and self.proc.poll() is None:
            self.proc.terminate()