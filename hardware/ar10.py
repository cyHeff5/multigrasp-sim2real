# IO-Adapter für die echte AR10-Hand über den Pololu Maestro Servo-Controller.
# Alle q-Werte sind normalisiert: 0.0 = vollständig offen, 1.0 = vollständig geschlossen.
# com_port=None -> Mock-Mode (kein Hardware nötig, read_q_measured gibt q_target zurück).

import json
import os
import time
from typing import List, Optional

try:
    import serial
except ImportError:
    serial = None 


# Sim-Joint-Index -> Maestro-Kanal, per Hardware verifiziert.
# Weicht von der originalen Active8-Nummerierung ab weil Sim und Hardware
# die Finger in unterschiedlicher Reihenfolge nummerieren.
#   sim j0/1  (Daumen)  -> ch18/19
#   sim j2/3  (Kleiner) -> ch16/17
#   sim j4/5  (Ring)    -> ch14/15
#   sim j6/7  (Mittel)  -> ch12/13
#   sim j8/9  (Zeige)   -> ch10/11
_CHANNELS = [18, 19, 16, 17, 14, 15, 12, 13, 10, 11]

# Originale Active8-Nummerierung, wird nur gebraucht um joint_input_calibration.json auf die Sim-Indizes umzurechnen.
_OLD_CHANNELS = [10, 11, 18, 19, 16, 17, 14, 15, 12, 13]

_DEFAULT_SERVO_MIN = [4200] * 10  # vollständig offen
_DEFAULT_SERVO_MAX = [7700] * 10  # vollständig geschlossen


class AR10Interface:

    def __init__(
        self,
        com_port: Optional[str] = None,
        servo_min: Optional[List[int]] = None,
        servo_max: Optional[List[int]] = None,
        speed: int = 100,
        acceleration: int = 0,
        input_calibration_file: Optional[str] = None,
    ):
        self._servo_min = list(servo_min) if servo_min is not None else list(_DEFAULT_SERVO_MIN)
        self._servo_max = list(servo_max) if servo_max is not None else list(_DEFAULT_SERVO_MAX)
        self._q_target: List[float] = [0.0] * 10
        self._usb: Optional[serial.Serial] = None
        self._input_cal: dict = self._load_input_calibration(input_calibration_file)

        if com_port is not None:
            if serial is None:
                raise ImportError("pyserial is required for hardware mode: pip install pyserial")
            if not self._input_cal:
                raise FileNotFoundError(
                    "joint_input_calibration.json not found — required for hardware mode."
                )
            self._usb = serial.Serial(com_port, baudrate=9600)
            for ch in _CHANNELS:
                self._set_channel_speed(ch, speed)
                time.sleep(0.05)
                self._set_channel_acceleration(ch, acceleration)
                time.sleep(0.05)

    # Pololu Maestro Protokoll
    def _send_command(self, *args: str) -> None:
        # Pololu Compact Protocol: 0xAA = Start, 0x0C = Gerätenummer, dann Befehlsbytes.
        if self._usb is None:
            return
        msg = chr(0xAA) + chr(0x0C) + "".join(args)
        self._usb.write(msg.encode("latin-1"))

    def _set_channel_speed(self, channel: int, speed: int) -> None:
        lsb = speed & 0x7F
        msb = (speed >> 7) & 0x7F
        self._send_command(chr(0x07), chr(channel), chr(lsb), chr(msb))

    def _set_channel_acceleration(self, channel: int, accel: int) -> None:
        accel = max(0, min(255, accel))
        lsb = accel & 0x7F
        msb = (accel >> 7) & 0x7F
        self._send_command(chr(0x09), chr(channel), chr(lsb), chr(msb))

    def _set_all_channel_targets(self, targets: List[int]) -> None:
        # Alle 10 Servo-Targets in einem einzigen Maestro-Befehl senden (0x1F).
        args = [chr(0x1F), chr(10), chr(10)]
        for i, t in enumerate(targets):
            t = max(self._servo_min[i], min(self._servo_max[i], t))
            args.append(chr(t & 0x7F))
            args.append(chr((t >> 7) & 0x7F))
        self._send_command(*args)

    # Sensor-Kalibrierung 
    @staticmethod
    def _load_input_calibration(path: Optional[str]) -> dict:
        # Liest joint_input_calibration.json und rechnet die alten Active8-Joint-Indizes auf die aktuellen Sim-Indizes um.
        if path is None:
            path = os.path.join(os.path.dirname(__file__), "joint_input_calibration.json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            return {}
        cal = {}
        for joint_str, jd in data.get("joints", {}).items():
            old_j = int(joint_str)
            old_ch = _OLD_CHANNELS[old_j]
            try:
                new_j = _CHANNELS.index(old_ch)
            except ValueError:
                continue
            cal[new_j] = {
                "input_channel": int(jd["input_channel"]),
                "open_real":     float(jd["opened"]["mapped_input"]),
                "closed_real":   float(jd["closed"]["mapped_input"]),
            }
        return cal

    def _read_input_channel(self, channel: int) -> int:
        # Liest analogen Sensorwert vom Maestro-Eingangskanal.
        if self._usb is None:
            return 0
        self._send_command(chr(0x10), chr(channel))
        lsb = ord(self._usb.read())
        msb = ord(self._usb.read())
        return (msb << 8) + lsb

    def _normalize_input(self, value: float, open_val: float, closed_val: float) -> float:
        denom = closed_val - open_val
        if denom == 0.0:
            return 0.0
        return max(0.0, min(1.0, (value - open_val) / denom))

    # Normalisierung
    def _to_servo(self, q_norm: float, joint_idx: int) -> int:
        lo = self._servo_min[joint_idx]
        hi = self._servo_max[joint_idx]
        return int(round(lo + max(0.0, min(1.0, q_norm)) * (hi - lo)))

    def _to_norm(self, servo_val: int, joint_idx: int) -> float:
        lo = self._servo_min[joint_idx]
        hi = self._servo_max[joint_idx]
        if hi == lo:
            return 0.0
        return max(0.0, min(1.0, (servo_val - lo) / (hi - lo)))

    # Public Interface
    def send_q_target(self, q_target: List[float]) -> None:
        # Sendet 10 normalisierte Zielwerte [0, 1] an die echte Hand.
        if len(q_target) != 10:
            raise ValueError(f"q_target must have 10 values, got {len(q_target)}.")
        self._q_target = [max(0.0, min(1.0, v)) for v in q_target]
        channel_targets = [0] * 10
        for joint_idx, v in enumerate(self._q_target):
            ch = _CHANNELS[joint_idx]
            channel_targets[ch - 10] = self._to_servo(v, joint_idx)
        self._set_all_channel_targets(channel_targets)

    def read_q_measured(self) -> List[float]:
        # Liest aktuelle Gelenkpositionen von den analogen Positionssensoren, normalisiert auf [0, 1].
        # Im Mock-Mode wird q_target zurückgegeben.
        if self._usb is None:
            return list(self._q_target)
        result = []
        for i in range(10):
            if i in self._input_cal:
                cal = self._input_cal[i]
                raw = self._read_input_channel(cal["input_channel"])
                result.append(self._normalize_input(raw, cal["open_real"], cal["closed_real"]))
            else:
                result.append(0.0)
        return result

    def position_error_norm(self) -> float:
        # Mittlerer absoluter Fehler zwischen q_target und q_measured.
        measured = self.read_q_measured()
        errors = [abs(t - m) for t, m in zip(self._q_target, measured)]
        return sum(errors) / len(errors)

    def close(self) -> None:
        if self._usb is not None:
            self._usb.close()
            self._usb = None
