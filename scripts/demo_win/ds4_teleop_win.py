#!/usr/bin/env python3
"""
DualShock 4 teleop for an OSCC drive-by-wire vehicle -- Windows 11 build.

  py -m pip install pygame python-can

Requires the PEAK PCAN-Basic driver (PCANBasic.dll), which ships with the
"PCAN-Basic API" / device driver package from peak-system.com. There is no
SocketCAN on Windows, so the bus is opened through python-can's pcan backend
and the bitrate is set here rather than with `ip link`.

  py ds4_teleop_win.py --probe        # identify axis/button indices, then exit
  py ds4_teleop_win.py --list-can     # show detected CAN channels, then exit
  py ds4_teleop_win.py --no-can       # gamepad only, nothing touches the bus
  py ds4_teleop_win.py                # live on PCAN_USBBUS1 @ 500 kbit/s

Racing layout:
  R2            throttle
  L2            brake
  left stick X  steering
  OPTIONS       arm    (enable OSCC control)
  SHARE         disarm (disable OSCC control)

Commands are always echoed to the terminal, so the gamepad can be verified
with the CAN adapter absent, the driver missing, or the vehicle powered down.
"""

import argparse
import atexit
import ctypes
import os
import struct
import sys
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")      # no window needed
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")  # no import banner
import pygame  # noqa: E402

# ============================================================================
# LIMITS AND TUNING
# ============================================================================

THROTTLE_MAX = 0.15      # [0..1] fraction of full pedal
BRAKE_MAX = 0.30         # [0..1]
STEER_MAX = 0.25         # [0..1] fraction of full torque, both directions

STEER_DEADZONE = 0.08    # stick units ignored around centre
STEER_EXPO = 0.5         # 0 = linear, 1 = heavily softened near centre

# Exponential filters, per tick. These take jitter out of the gamepad; they are
# NOT a rate limit. The brake module closes a PID loop on measured wheel
# pressure and drives the fill/spill solenoids by PWM, with its output already
# bounded by BRAKE_PID_OUTPUT_MIN/MAX. Rate-limiting the setpoint on top only
# adds lag going down and delays release. Command what you want.
BRAKE_FILTER = 0.35
THROTTLE_FILTER = 0.35
STEER_FILTER = 0.25

# Brake wins: above this brake command the throttle is forced to zero.
# (A choice made here, not something the OSCC firmware enforces.)
BRAKE_OVERRIDES_THROTTLE_AT = 0.02

RATE_HZ = 50.0           # stay well above 5 Hz; module watchdog is 200 ms
PRINT_HZ = 10.0          # terminal status refresh

# One zero+disable burst is sent when the gamepad drops, then the script goes
# silent. Set False to go silent immediately and let the watchdog time out.
SEND_DISABLE_ON_LOST = True

# ============================================================================
# GAMEPAD MAPPING
#
# SDL maps the DS4 differently on Windows than on Linux. On Windows the usual
# layout is 0=LX 1=LY 2=RX 3=L2 4=R2 5=RY; the hid-sony layout on Linux puts
# the triggers on 2 and 5 instead. If DS4Windows or ViGEm is running, the pad
# presents as an Xbox 360 controller and the mapping changes again.
#
# Run --probe once and set these. It takes ten seconds and removes all doubt.
# ============================================================================

AXIS_STEER = 0           # left stick X
AXIS_BRAKE = 4           # L2 analog
AXIS_THROTTLE = 5        # R2 analog

BTN_ARM = 6              # OPTIONS
BTN_DISARM = 4           # SHARE
BTN_HOLD = 9             # L1 -- dead-man; commands only flow while it is held

# Arming enables the OSCC modules but sends nothing but zeros. Commands only
# reach the vehicle while BTN_HOLD is down. Releasing it zeroes everything
# immediately while keeping the modules enabled, so re-engaging is one press
# rather than a full re-arm.
#
# Zeros keep flowing rather than stopping: silence would trip the 200 ms
# firmware watchdog, fault the modules and force a re-arm every time you let
# go. Seconds of continuous release before a full disarm; 0 disables that.
HOLD_RELEASE_DISARM_AFTER = 5.0

# DS4 triggers usually rest at -1.0 and reach +1.0, but an XInput-emulated pad
# rests at 0.0. Leave AUTO on and the resting value is sampled at startup.
TRIGGER_REST_AUTO = True
TRIGGER_REST = -1.0
TRIGGER_FULL = 1.0

# ============================================================================
# OSCC CONTROL CAN PROTOCOL
# ============================================================================

MAGIC = b"\x05\xcc"

BRAKE = {"enable": 0x070, "disable": 0x071, "command": 0x072, "report": 0x073}
STEER = {"enable": 0x080, "disable": 0x081, "command": 0x082, "report": 0x083}
THROTTLE = {"enable": 0x090, "disable": 0x091, "command": 0x092, "report": 0x093}
FAULT_ID = 0x0AF

CAN_INTERFACE = "pcan"
CAN_CHANNEL = "PCAN_USBBUS1"
CAN_BITRATE = 500000


# ============================================================================
# Windows timer resolution
#
# The default scheduler tick is 15.6 ms, so time.sleep(0.02) can overshoot to
# 31 ms and the command rate collapses to ~32 Hz. Still inside the 200 ms
# watchdog, but the jitter is visible in the car. timeBeginPeriod(1) pulls it
# to 1 ms for the lifetime of the process.
# ============================================================================

def sharpen_timer():
    if sys.platform != "win32":
        return
    try:
        winmm = ctypes.WinDLL("winmm")
        winmm.timeBeginPeriod(1)
        atexit.register(winmm.timeEndPeriod, 1)
    except Exception as exc:                          # noqa: BLE001
        print(f"Could not raise timer resolution ({exc}); expect rate jitter.")


# ============================================================================
# CAN transport -- degrades to a no-op so the gamepad can still be checked
# ============================================================================

class Bus:
    def __init__(self, interface, channel, bitrate, enabled=True):
        self.bus = None
        self.reason = "disabled by --no-can"

        if not enabled:
            return

        try:
            import can
        except ImportError:
            self.reason = "python-can not installed (py -m pip install python-can)"
            return

        self.can = can
        try:
            self.bus = can.interface.Bus(
                interface=interface, channel=channel, bitrate=bitrate)
            self.reason = None
        except Exception as exc:                      # noqa: BLE001
            name = type(exc).__name__
            hint = ""
            if "PCANBasic" in str(exc) or "DLL" in str(exc):
                hint = " -- install the PEAK PCAN-Basic driver package"
            elif "initialize" in str(exc).lower():
                hint = " -- adapter unplugged, or the channel is open elsewhere"
            self.reason = f"{name}: {exc}{hint}"

    @property
    def live(self):
        return self.bus is not None

    def send(self, can_id, payload=b"\x00" * 6):
        if not self.live:
            return
        try:
            self.bus.send(self.can.Message(
                arbitration_id=can_id, data=MAGIC + payload, is_extended_id=False))
        except Exception:                             # noqa: BLE001
            pass  # a full TX queue must not take the control loop down

    def command(self, can_id, value):
        self.send(can_id, struct.pack("<f", value) + b"\x00\x00")

    def shutdown(self):
        if self.live:
            try:
                self.bus.shutdown()
            except Exception:                         # noqa: BLE001
                pass

    def drain(self):
        """Return {'steer': (enabled, override, dtc), ...} plus any fault."""
        state = {}
        if not self.live:
            return state

        while True:
            try:
                message = self.bus.recv(timeout=0)
            except Exception:                         # noqa: BLE001
                break
            if message is None:
                break
            if len(message.data) < 8 or message.data[:2] != MAGIC:
                continue

            if message.arbitration_id == BRAKE["report"]:
                state["brake"] = (message.data[2], message.data[3], message.data[4])
            elif message.arbitration_id == STEER["report"]:
                state["steer"] = (message.data[2], message.data[3], message.data[4])
            elif message.arbitration_id == THROTTLE["report"]:
                state["throttle"] = (message.data[2], message.data[3], message.data[4])
            elif message.arbitration_id == FAULT_ID:
                origin = int.from_bytes(message.data[2:6], "little")
                names = ["brake", "steering", "throttle"]
                who = names[origin] if origin < len(names) else f"id{origin}"
                state["fault"] = (who, message.data[6])

        return state


def list_can():
    try:
        import can
    except ImportError:
        print("python-can not installed: py -m pip install python-can")
        return 1

    print("Configured channels visible to python-can:\n")
    try:
        found = can.detect_available_configs(["pcan", "vector", "kvaser", "ixxat"])
    except Exception as exc:                          # noqa: BLE001
        print(f"  detection failed: {exc}")
        found = []

    if not found:
        print("  (none) -- adapter unplugged, or the vendor driver is missing.")
        print("  For PCAN-USB, install the PCAN-Basic / device driver package.")
    for config in found:
        print(f"  {config}")
    return 0


# ============================================================================
# Gamepad
# ============================================================================

def trigger_to_unit(raw, rest):
    unit = (raw - rest) / (TRIGGER_FULL - rest) if TRIGGER_FULL != rest else 0.0
    return min(1.0, max(0.0, unit))


def apply_deadzone_expo(raw):
    if abs(raw) < STEER_DEADZONE:
        return 0.0
    sign = 1.0 if raw > 0 else -1.0
    scaled = (abs(raw) - STEER_DEADZONE) / (1.0 - STEER_DEADZONE)
    shaped = (1.0 - STEER_EXPO) * scaled + STEER_EXPO * scaled ** 3
    return sign * min(1.0, shaped)


def open_pad():
    """Return joystick 0 if one is attached, else None.

    Deliberately does NOT call pygame.joystick.quit(). Tearing the subsystem
    down clears pygame's SDL-instance-id table while device events may still
    be queued against it; translating one of those then raises KeyError deep
    inside pygame.event.get(), which surfaces as an unhandled SystemError.
    """
    if not pygame.joystick.get_init():
        pygame.joystick.init()

    pygame.event.pump()

    if pygame.joystick.get_count() == 0:
        return None

    try:
        pad = pygame.joystick.Joystick(0)
        pad.init()
        return pad
    except pygame.error:
        return None


def pad_alive(pad):
    """True if the pad is still attached and readable."""
    if pad is None or pygame.joystick.get_count() == 0:
        return False
    try:
        pad.get_axis(AXIS_STEER)
        return True
    except (pygame.error, AttributeError):
        return False


def wait_for_pad(timeout=3.0):
    """Give SDL a moment to enumerate; Bluetooth pads are not instant."""
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        pad = open_pad()
        if pad is not None:
            return pad
        time.sleep(0.1)
    return None


def calibrate_triggers(pad):
    """Sample the resting value of both trigger axes for half a second."""
    if not TRIGGER_REST_AUTO:
        return TRIGGER_REST

    lowest = 1.0
    deadline = time.perf_counter() + 0.5
    while time.perf_counter() < deadline:
        pygame.event.pump()
        lowest = min(lowest, pad.get_axis(AXIS_BRAKE), pad.get_axis(AXIS_THROTTLE))
        time.sleep(0.01)

    # Snap to the two values that actually occur, so noise near zero does not
    # shift the whole scale.
    rest = -1.0 if lowest < -0.5 else 0.0
    print(f"Trigger rest calibrated to {rest:+.1f} (observed {lowest:+.2f})")
    return rest


def probe():
    pygame.init()
    pygame.joystick.init()
    pad = wait_for_pad(3.0)
    if pad is None:
        print("No gamepad found.")
        print("USB: just plug the DS4 in. Bluetooth: hold SHARE+PS until the")
        print("bar flashes, then pair from Settings > Bluetooth & devices.")
        return 1

    print(f"{pad.get_name()}  axes={pad.get_numaxes()}  buttons={pad.get_numbuttons()}")
    print("Pull L2, then R2, then move the left stick. Press OPTIONS and SHARE.")
    print("Note the indices, put them at the top of this file. Ctrl-C to stop.\n")

    try:
        while True:
            pygame.event.pump()
            axes = " ".join(f"{i}:{pad.get_axis(i):+.2f}" for i in range(pad.get_numaxes()))
            pressed = [i for i in range(pad.get_numbuttons()) if pad.get_button(i)]
            print(f"\r{axes}  btn={pressed}   ", end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print()
    return 0


# ============================================================================
# Main loop
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", action="store_true",
                        help="print live axis/button indices and exit")
    parser.add_argument("--list-can", action="store_true",
                        help="list detected CAN channels and exit")
    parser.add_argument("--no-can", action="store_true",
                        help="gamepad only; never touch the bus")
    parser.add_argument("--interface", default=CAN_INTERFACE,
                        help="python-can backend (pcan, vector, kvaser, slcan)")
    parser.add_argument("--channel", default=CAN_CHANNEL)
    parser.add_argument("--bitrate", type=int, default=CAN_BITRATE)
    args = parser.parse_args()

    if args.list_can:
        return list_can()
    if args.probe:
        return probe()

    sharpen_timer()
    pygame.init()
    pygame.joystick.init()
    # We never call event.get(), so stop pygame building event objects at all.
    # This also removes the code path that raised KeyError on hotplug.
    pygame.event.set_allowed(None)

    bus = Bus(args.interface, args.channel, args.bitrate, enabled=not args.no_can)
    if bus.live:
        print(f"CAN: {args.interface}:{args.channel} @ {args.bitrate} bit/s")
    else:
        print(f"CAN: DRY RUN ({bus.reason})")
        print("     try --list-can to see what the driver can see")

    print(f"Limits: throttle<={THROTTLE_MAX:.2f} brake<={BRAKE_MAX:.2f} "
          f"steer<=+/-{STEER_MAX:.2f}")
    print("OPTIONS = arm, SHARE = disarm. Triggers must be at rest to arm.")
    print("HOLD L1 for commands to reach the vehicle; release = everything zero.\n")

    print("Gamepad: looking...", end="", flush=True)
    pad = wait_for_pad(3.0)
    trigger_rest = TRIGGER_REST
    if pad:
        print(f"\rGamepad: {pad.get_name()}          ")
        trigger_rest = calibrate_triggers(pad)
    else:
        print("\rGamepad: none yet, will keep looking.")

    armed = False
    lost = (pad is None)
    was_held = False
    released_at = None
    steer_cmd = 0.0
    brake_cmd = 0.0
    throttle_cmd = 0.0
    period = 1.0 / RATE_HZ
    next_print = 0.0
    reports = {}

    def disarm(reason):
        nonlocal armed, steer_cmd, brake_cmd, throttle_cmd, was_held, released_at

        # Zero, then disable. No ramp-down: a brake setpoint of 0.0 is what
        # tells the module's PID to open the spill solenoid. Walking the
        # setpoint down would keep the brakes held while it walked.
        if armed:
            for _ in range(5):
                bus.command(BRAKE["command"], 0.0)
                bus.command(THROTTLE["command"], 0.0)
                bus.command(STEER["command"], 0.0)
                time.sleep(0.005)
            for group in (BRAKE, STEER, THROTTLE):
                bus.send(group["disable"])

        armed = False
        was_held = False
        released_at = None
        steer_cmd = 0.0
        brake_cmd = 0.0
        throttle_cmd = 0.0
        print(f"\nDISARMED ({reason})")

    try:
        while True:
            tick = time.perf_counter()

            # ---- gamepad presence -------------------------------------------
            # Polled, not event-driven: pygame.event.get() translates device
            # events through an instance-id table that goes stale on hotplug,
            # and a miss there raises out of pygame itself. pump() keeps SDL's
            # joystick state fresh without producing objects to translate.
            pygame.event.pump()

            if not pad_alive(pad):
                if not lost:
                    if armed and SEND_DISABLE_ON_LOST:
                        disarm("gamepad lost")
                    else:
                        armed = False
                    lost = True
                    pad = None
                    print("\nGAMEPAD LOST -- transmitting nothing. "
                          "Module watchdog (200 ms) will disable control.")
                else:
                    pad = open_pad()
                    if pad is not None:
                        lost = False
                        trigger_rest = calibrate_triggers(pad)
                        print(f"Gamepad reconnected: {pad.get_name()}"
                              " -- press OPTIONS to arm again")
                time.sleep(period)
                continue

            # ---- read sticks -------------------------------------------------
            throttle_in = trigger_to_unit(pad.get_axis(AXIS_THROTTLE), trigger_rest)
            brake_in = trigger_to_unit(pad.get_axis(AXIS_BRAKE), trigger_rest)
            steer_in = apply_deadzone_expo(pad.get_axis(AXIS_STEER))

            if pad.get_button(BTN_DISARM) and armed:
                disarm("SHARE")
            elif pad.get_button(BTN_ARM) and not armed:
                if throttle_in > 0.02 or brake_in > 0.02:
                    print("\nRefusing to arm: release both triggers first.")
                    time.sleep(0.3)
                else:
                    for group in (BRAKE, STEER, THROTTLE):
                        bus.send(group["enable"])
                    armed = True
                    was_held = False
                    released_at = tick
                    print("\nARMED -- hold L1 to take control")

            # ---- dead-man ----------------------------------------------------
            held = bool(pad.get_button(BTN_HOLD))

            if armed and held != was_held:
                print("\nLIVE -- L1 held" if held else "\nHOLD RELEASED -- commands zeroed")
                released_at = None if held else tick
            was_held = held

            if armed and not held and HOLD_RELEASE_DISARM_AFTER > 0.0:
                if released_at is not None and tick - released_at > HOLD_RELEASE_DISARM_AFTER:
                    disarm(f"L1 released for {HOLD_RELEASE_DISARM_AFTER:.0f} s")

            # ---- scale by the limits, then filter ---------------------------
            brake_cmd += BRAKE_FILTER * (brake_in * BRAKE_MAX - brake_cmd)
            throttle_cmd += THROTTLE_FILTER * (throttle_in * THROTTLE_MAX - throttle_cmd)
            steer_cmd += STEER_FILTER * (steer_in * STEER_MAX - steer_cmd)

            # Brake wins. Cut throttle outright rather than filtering it down.
            if brake_cmd > BRAKE_OVERRIDES_THROTTLE_AT:
                throttle_cmd = 0.0

            # ---- transmit ----------------------------------------------------
            if armed and held:
                bus.command(BRAKE["command"], brake_cmd)
                bus.command(THROTTLE["command"], throttle_cmd)
                bus.command(STEER["command"], steer_cmd)
            elif armed:
                # Enabled but not held: keep the modules fed with zeros so the
                # watchdog stays happy and one press of L1 re-engages.
                brake_cmd = 0.0
                throttle_cmd = 0.0
                steer_cmd = 0.0
                bus.command(BRAKE["command"], 0.0)
                bus.command(THROTTLE["command"], 0.0)
                bus.command(STEER["command"], 0.0)
            else:
                brake_cmd = 0.0
                throttle_cmd = 0.0
                steer_cmd = 0.0

            new = bus.drain()
            reports.update(new)

            if "fault" in new:
                who, dtc = new["fault"]
                print(f"\nFAULT from {who}: dtc=0b{dtc:08b}")
                disarm("fault report")

            for name in ("brake", "steer", "throttle"):
                if name in new and new[name][1]:
                    print(f"\nOPERATOR OVERRIDE on {name}")
                    disarm("operator override")
                    break

            # ---- terminal echo, always, CAN or not ---------------------------
            if tick >= next_print:
                next_print = tick + 1.0 / PRINT_HZ
                flags = "".join(
                    key[0].upper() if reports.get(key, (0,))[0] else "-"
                    for key in ("brake", "steer", "throttle"))
                cut = "!" if (brake_cmd > BRAKE_OVERRIDES_THROTTLE_AT
                              and throttle_in > 0.02) else " "
                link = "can" if bus.live else "dry"
                state = ("LIVE  " if held else "ARMED ") if armed else "idle  "
                print(f"\r{state} thr {throttle_cmd:5.2f}{cut} brk {brake_cmd:5.2f}  "
                      f"str {steer_cmd:+5.2f} | en:{flags} | {link}   ",
                      end="", flush=True)

            time.sleep(max(0.0, period - (time.perf_counter() - tick)))

    except KeyboardInterrupt:
        pass
    finally:
        disarm("shutdown")
        bus.shutdown()
        print("bye")

    return 0


if __name__ == "__main__":
    sys.exit(main())
