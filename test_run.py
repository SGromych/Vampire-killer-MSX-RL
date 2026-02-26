from pathlib import Path
import time
from openmsx_bridge import OpenMSXFileControl

rom = Path(__file__).with_name("VAMPIRE.ROM").resolve()
if not rom.exists():
    raise FileNotFoundError(f"VAMPIRE.ROM not found next to test_run.py: {rom}")

print("Starting openMSX (file-control loop)...")
emu = OpenMSXFileControl(str(rom), workdir=Path(__file__).parent, poll_ms=20, boot_timeout_s=25.0)

print("Ping:", emu.ping())

print("Wait for game to leave BIOS blue screen...")
emu.wait_for_nonblue_screen(timeout_s=30.0, check_every_s=0.5)
time.sleep(1.0)  # дать заголовку/меню стабилизироваться

# press_type — команда openMSX "type" (альтернатива keymatrix)
print("SPACE на заставке Konami (type)...")
emu.press_type("SPACE")
time.sleep(2.0)
print("SPACE на PRESS START (type)...")
emu.press_type("SPACE")
time.sleep(3.0)  # ждём загрузки основного экрана

# Движение — только через keymatrix (MSX-клавиатура), без джойстика/pyautogui

print("RIGHT x15 (дискретно)...")
for _ in range(15):
    emu.press("RIGHT", hold_ms=120)
    time.sleep(0.08)
time.sleep(0.3)

print("LEFT x5 (дискретно)...")
for _ in range(5):
    emu.press("LEFT", hold_ms=120)
    time.sleep(0.08)
time.sleep(0.3)

print("RIGHT hold 1.5s (непрерывное движение)...")
emu.press("RIGHT", hold_ms=1500)
time.sleep(0.5)

print("Прыжок вправо x3 (RIGHT+UP, combo)...")
for _ in range(3):
    emu.press_combo(["RIGHT", "UP"], hold_ms=220)
    time.sleep(0.4)

print("Прыжок влево x3 (LEFT+UP, combo)...")
for _ in range(3):
    emu.press_combo(["LEFT", "UP"], hold_ms=220)
    time.sleep(0.4)

print("Прыжок вправо с ударом x2 (RIGHT+UP+SPACE)...")
for _ in range(2):
    emu.press_combo(["RIGHT", "UP", "SPACE"], hold_ms=220)
    time.sleep(0.5)

print("Прыжок влево с ударом x2 (LEFT+UP+SPACE)...")
for _ in range(2):
    emu.press_combo(["LEFT", "UP", "SPACE"], hold_ms=220)
    time.sleep(0.5)

print("Присесть x2 (DOWN)...")
for _ in range(2):
    emu.press("DOWN", hold_ms=350)
    time.sleep(0.5)

print("Удар/хлыст x3 (SPACE отдельно)...")
for _ in range(3):
    emu.press("SPACE", hold_ms=120)
    time.sleep(0.35)

print("Watch the game for 10 seconds, then close...")
time.sleep(10.0)

emu.close()
print("Done.")