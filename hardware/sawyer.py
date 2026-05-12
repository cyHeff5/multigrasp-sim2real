# Sawyer-Hardware: ROS-Interface, SimLimb-Mock und Motion-Primitives.
# Für Visualisierung und IK-Berechnungen wird sim.sawyer importiert.
from __future__ import annotations

import time

from hardware.gamepad import read_inputs
from sim.sawyer import (
    _SAWYER_JOINT_NAMES, SawyerHelper, set_pb_joints, update_hand_from_ee,
)


_MOVE_SPEED = 0.25    # alpha/s beim X-halten
_DT         = 1.0 / 50.0
_STICK_M    = 0.001   # EE-Versatz pro Tick (Feinjustierung)
_BUMP_M     = 0.001


class SimLimb:
    # Mock für intera_interface.Limb, für --sim-only Modus ohne echten Sawyer.

    def __init__(self, robot_id: int, helper: SawyerHelper) -> None:
        self._robot_id = robot_id
        self._helper   = helper

    def joint_angles(self) -> dict:
        import pybullet as p
        return {n: float(p.getJointState(self._robot_id, j)[0])
                for n, j in zip(_SAWYER_JOINT_NAMES, self._helper.joints)}

    def set_joint_positions(self, angles_dict: dict) -> None:
        pass  # PyBullet wird direkt über set_pb_joints gesteuert


def init_ros_sawyer():
    # Initialisiert ROS-Node und Sawyer-Limb.
    try:
        import rospy
        import intera_interface
    except ImportError as exc:
        raise RuntimeError("ROS/intera nicht verfügbar. --sim-only verwenden.") from exc
    rospy.init_node("multigrasp_sawyer", anonymous=True)
    limb = intera_interface.Limb("right")
    print("[hw] Sawyer bereit.")
    return limb


def read_q(limb) -> list[float]:
    d = limb.joint_angles()
    return [float(d[n]) for n in _SAWYER_JOINT_NAMES]


def send_q(limb, q: list[float]) -> None:
    limb.set_joint_positions({n: float(v) for n, v in zip(_SAWYER_JOINT_NAMES, q)})


def drive_x_held(
    limb,
    robot_id: int,
    helper: SawyerHelper,
    js,
    q_real_start: list[float],
    q_real_target: list[float],
    q_pb_start: list[float],
    q_pb_target: list[float],
    label: str = "Zielpose",
    auto_proceed: bool = False,
) -> str:
    # Fährt linear zur Zielpose solange X gehalten wird.
    # auto_proceed=True: gibt "ok" zurück sobald alpha=1.0, ohne A-Bestätigung.
    # Gibt "ok" | "cancel" | "quit" zurück.

    # Gehaltene Knöpfe entleeren
    while True:
        inp = read_inputs(js)
        if not inp["a"] and not inp["b"]:
            break
        time.sleep(_DT)

    alpha  = 0.0
    prev_a = prev_b = False
    print(f"  X halten -> {label} anfahren   |   A = Bestätigen   |   B = Abbrechen")

    while True:
        inp    = read_inputs(js)
        a_trig = inp["a"] and not prev_a
        b_trig = inp["b"] and not prev_b
        prev_a = inp["a"]
        prev_b = inp["b"]

        if inp["menu"]:
            return "quit"
        if b_trig:
            return "cancel"

        if inp["x"] and alpha < 1.0:
            alpha  = min(1.0, alpha + _MOVE_SPEED * _DT)
            q_real = [s + alpha * (t - s) for s, t in zip(q_real_start, q_real_target)]
            q_pb   = [s + alpha * (t - s) for s, t in zip(q_pb_start,   q_pb_target)]
            print(f"\r  Fortschritt: {int(alpha * 100):3d}%"
                  f"{'  [FERTIG]' if alpha >= 1.0 else ''}",
                  end="", flush=True)
            if alpha >= 1.0:
                print()
            set_pb_joints(robot_id, helper, q_pb)
            send_q(limb, q_real)

        time.sleep(_DT)

        if auto_proceed and alpha >= 1.0:
            return "ok"
        if a_trig:
            return "ok"


def ik_fine_tune(
    limb,
    robot_id: int,
    helper: SawyerHelper,
    js,
    q_start: list[float],
    label: str = "Feinjustierung",
    hand_id: int = -1,
) -> tuple[str, list[float]]:
    # Joystick-gesteuerte EE-Feinjustierung via IK.
    # Gibt ("ok"|"cancel"|"quit", q_pb) zurück.
    while True:
        inp = read_inputs(js)
        if not inp["a"] and not inp["b"]:
            break
        time.sleep(_DT)

    set_pb_joints(robot_id, helper, q_start)
    _, ee_rpy = helper.ee_pose()
    ee_rpy    = [float(v) for v in ee_rpy]
    q_pb      = list(q_start)
    prev_a = prev_b = False

    print(f"\n{'='*60}\n  {label}\n{'='*60}")
    print("  Linker Stick X/Y -> EE lateral")
    print("  RB / LB          -> EE hoch / runter")
    print("  A -> Bestätigen   B -> Abbrechen\n")

    while True:
        inp    = read_inputs(js)
        a_trig = inp["a"] and not prev_a
        b_trig = inp["b"] and not prev_b
        prev_a = inp["a"]
        prev_b = inp["b"]

        if inp["menu"]:
            return "quit", q_pb
        if b_trig:
            return "cancel", q_pb

        dz = (float(inp["rb"]) - float(inp["lb"])) * _BUMP_M
        sx, sy = float(inp["sx"]), float(inp["sy"])

        if abs(sx) > 0 or abs(sy) > 0 or abs(dz) > 1e-6:
            ee_xyz, _ = helper.ee_pose()
            target = [float(ee_xyz[0]) + sx * _STICK_M,
                      float(ee_xyz[1]) + sy * _STICK_M,
                      float(ee_xyz[2]) + dz]
            try:
                q_pb = helper._ik(target, ee_rpy).tolist()
            except Exception as exc:
                print(f"\n[IK Fehler] {exc}")

        set_pb_joints(robot_id, helper, q_pb)
        if hand_id >= 0:
            update_hand_from_ee(robot_id, helper, hand_id)
        send_q(limb, q_pb)
        time.sleep(_DT)

        if a_trig:
            return "ok", q_pb


def cal_fine_tune(
    limb,
    robot_id: int,
    helper: SawyerHelper,
    js,
    q_start: list[float],
    offset_init: list[float],
    label: str = "Kalibrierungs-Feinjustierung",
    hand_id: int = -1,
) -> tuple[str, list[float], list[float]]:
    # Feinjustierung mit Offset-Akkumulation für Hand-Kalibrierung.
    # new_offset = offset_init + (EE_jetzt - EE_referenz).
    # Gibt ("ok"|"cancel"|"quit", q_pb, new_offset) zurück.
    while True:
        inp = read_inputs(js)
        if not inp["a"] and not inp["b"]:
            break
        time.sleep(_DT)

    set_pb_joints(robot_id, helper, q_start)
    ee_ref, ee_rpy = helper.ee_pose()
    ee_ref = [float(v) for v in ee_ref]
    ee_rpy = [float(v) for v in ee_rpy]
    q_pb   = list(q_start)
    prev_a = prev_b = False

    print(f"\n{'='*60}\n  {label}\n{'='*60}")
    print("  Linker Stick X/Y -> EE lateral")
    print("  RB / LB          -> EE hoch / runter")
    print("  A -> Bestätigen & Offset speichern   B -> Abbrechen\n")

    while True:
        inp    = read_inputs(js)
        a_trig = inp["a"] and not prev_a
        b_trig = inp["b"] and not prev_b
        prev_a = inp["a"]
        prev_b = inp["b"]

        if inp["menu"]:
            return "quit", q_pb, offset_init
        if b_trig:
            return "cancel", q_pb, offset_init

        dz = (float(inp["rb"]) - float(inp["lb"])) * _BUMP_M
        sx, sy = float(inp["sx"]), float(inp["sy"])

        if abs(sx) > 0 or abs(sy) > 0 or abs(dz) > 1e-6:
            ee_now, _ = helper.ee_pose()
            target = [float(ee_now[0]) + sx * _STICK_M,
                      float(ee_now[1]) + sy * _STICK_M,
                      float(ee_now[2]) + dz]
            try:
                q_pb = helper._ik(target, ee_rpy).tolist()
            except Exception as exc:
                print(f"\n[IK Fehler] {exc}")

        set_pb_joints(robot_id, helper, q_pb)
        if hand_id >= 0:
            update_hand_from_ee(robot_id, helper, hand_id)
        send_q(limb, q_pb)

        ee_now, _  = helper.ee_pose()
        delta      = [float(ee_now[i]) - ee_ref[i] for i in range(3)]
        new_offset = [offset_init[i] + delta[i] for i in range(3)]
        print(f"\r  Delta: {[round(v*1000,1) for v in delta]} mm  "
              f"Offset: {[round(v*1000,1) for v in new_offset]} mm  ",
              end="", flush=True)
        time.sleep(_DT)

        if a_trig:
            print()
            return "ok", q_pb, new_offset
