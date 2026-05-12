# Greifpunkt auswählen, Pregrasp anfahren und Lift testen.
# Greifpunkte und Pregrasp-Posen werden aus der Lookup Table per IK berechnet.
#
# Ablauf:
#   1. SELECT      : RB/LB navigieren, A bestätigen
#   2. SYNC        : X halten -> Home-Pose (Q_SYNC)
#   3. PREGRASP    : X halten -> Pregrasp-Pose (IK aus Lookup Table)
#   4. AT PREGRASP : Y=Feinjustierung  A=Lift  B=neu wählen  Menu=Beenden
#
# Gespeicherte Offsets: artifacts/calibration/gp_offsets.yaml
#
# Verwendung:
#   python -m hardware.drive_pregrasp
#   python -m hardware.drive_pregrasp --sim-only
#   python -m hardware.drive_pregrasp --robot-rpy 0 0 180
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pybullet as p
import pybullet_data
import yaml

from assets import benchmark_part_urdf
from sim.sawyer import (
    SawyerHelper, init_pybullet, init_hand, set_pb_joints, update_hand_from_ee,
)
from hardware.sawyer import (
    SimLimb, init_ros_sawyer, read_q, send_q, drive_x_held, cal_fine_tune,
)
from hardware.gamepad import read_inputs, init_pygame_joystick


# Sichere Home-Pose (Sawyer-Neutral nahe Tisch)
Q_SYNC = [-0.3531, 0.1428, -0.8645, 0.8330, -1.2630, -2.3962, -0.8262]

_OBJ_XY      = [0.50, 0.00]
_PED_H       = 0.04
_LIFT_DZ     = 0.05   # Hubhöhe in Metern

_LOOKUP_YAML  = "artifacts/grasp_lookup_table.yaml"
_OFFSETS_FILE = Path("artifacts/calibration/gp_offsets.yaml")

_ROBOT_BASE_RPY_DEG = [0.0, 0.0, 0.0]
_DT = 1.0 / 50.0


# Lookup Table I/O

def _load_parts(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)["parts"]


def _flat_gp_list(parts: list[dict]) -> list[dict]:
    # Flacht die verschachtelte Parts-Struktur zu einer linearen Liste ab
    # damit RB/LB einfach einen Index inkrementieren/dekrementieren kann.
    result = []
    for part in parts:
        pid = int(part["part_id"])
        for gp in part.get("grasp_points", []):
            result.append({"part_id": pid, "gp": gp})
    return result


# GP-Offset I/O

def _offset_key(part_id: int, gp_id: str) -> str:
    # Schlüsselformat in gp_offsets.yaml: "partid_gpid"
    return f"{part_id}_{gp_id}"


def _load_gp_offset(part_id: int, gp_id: str) -> list[float]:
    if _OFFSETS_FILE.exists():
        with _OFFSETS_FILE.open(encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        off = (data.get("offsets") or {}).get(_offset_key(part_id, gp_id))
        if off:
            return [float(v) for v in off]
    return [0.0, 0.0, 0.0]


def _save_gp_offset(part_id: int, gp_id: str, offset: list[float]) -> None:
    _OFFSETS_FILE.parent.mkdir(parents=True, exist_ok=True)
    data: dict = {}
    if _OFFSETS_FILE.exists():
        with _OFFSETS_FILE.open(encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    data.setdefault("offsets", {})[_offset_key(part_id, gp_id)] = \
        [round(float(v), 6) for v in offset]
    with _OFFSETS_FILE.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False)
    print(f"[gespeichert] gp_offset {_offset_key(part_id, gp_id)} -> {_OFFSETS_FILE}")


# Geometrie-Hilfsfunktionen

def _seated_spawn_pos(part_id: int, orientation: list[float]) -> list[float]:
    # Temporäre DIRECT-Instanz: AABB-Min bestimmen und Z so setzen
    # dass das Teil exakt auf dem Podest steht.
    cid = p.connect(p.DIRECT)
    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)
        tmp = p.loadURDF(
            str(benchmark_part_urdf(part_id)),
            [_OBJ_XY[0], _OBJ_XY[1], _PED_H + 0.5],
            list(orientation),
            useFixedBase=True, physicsClientId=cid,
        )
        aabb_min, _ = p.getAABB(tmp, physicsClientId=cid)
        cur_pos, _  = p.getBasePositionAndOrientation(tmp, physicsClientId=cid)
        obj_z = float(cur_pos[2]) + (_PED_H - float(aabb_min[2]))
    finally:
        p.disconnect(cid)
    return [_OBJ_XY[0], _OBJ_XY[1], obj_z]


def _world_hand_pose(obj_pos: list, gp: dict) -> tuple[list, list]:
    # Transformiert Pregrasp-Pose vom Objekt-Frame in den Welt-Frame.
    obj_quat = [float(v) for v in gp["object_orientation_xyzw"]]
    hp_w, hq_w = p.multiplyTransforms(
        obj_pos, obj_quat,
        [float(v) for v in gp["pregrasp_position_obj_xyz"]],
        [float(v) for v in gp["pregrasp_orientation_obj_xyzw"]],
    )
    return list(hp_w), list(hq_w)


def _hand_to_ik_target(hand_pos: list, hand_quat: list) -> tuple[list, list]:
    # Inverse von update_hand_from_ee: Hand-Pose -> Sawyer-EE-Ziel.
    rot180z = p.getQuaternionFromEuler([0.0, 0.0, math.pi])
    _, ee_quat = p.multiplyTransforms([0, 0, 0], hand_quat, [0, 0, 0], rot180z)
    ee_pos, _ = p.multiplyTransforms(hand_pos, ee_quat, [0.0, 0.0, -0.03], [0, 0, 0, 1])
    return list(ee_pos), list(p.getEulerFromQuaternion(ee_quat))


def _compute_pregrasp_q(helper: SawyerHelper, part_id: int,
                         gp: dict, offset: list[float]) -> list[float]:
    # rest=Q_SYNC: IK wählt reproduzierbar die der Home-Pose nächste Lösung,
    # unabhängig von der aktuellen Arm-Position.
    obj_orn = [float(v) for v in gp["object_orientation_xyzw"]]
    obj_pos = _seated_spawn_pos(part_id, obj_orn)
    hand_pos, hand_quat = _world_hand_pose(obj_pos, gp)
    ee_pos, ee_rpy = _hand_to_ik_target(hand_pos, hand_quat)
    ee_off = [ee_pos[i] + offset[i] for i in range(3)]
    return helper._ik(ee_off, ee_rpy, rest=Q_SYNC).tolist()


# Visualisierung

class _Vis:
    # Verwaltet das Benchmark-Objekt in der GUI.
    # URDF wird nur neu geladen wenn Part oder Orientierung wechselt.

    def __init__(self) -> None:
        self.obj_id: int = -1
        self.obj_part_id: int = -1
        self.obj_orn: list = [0, 0, 0, 1]

    def refresh(self, part_id: int, gp: dict,
                hand_id: int = -1, hand_model=None) -> None:
        orn = [float(v) for v in gp["object_orientation_xyzw"]]
        if self.obj_id < 0 or self.obj_part_id != part_id or self.obj_orn != orn:
            self._remove_obj()
            spawn_pos = _seated_spawn_pos(part_id, orn)
            try:
                self.obj_id = p.loadURDF(
                    str(benchmark_part_urdf(part_id)),
                    basePosition=spawn_pos,
                    baseOrientation=orn,
                    useFixedBase=True,
                )
            except Exception as exc:
                print(f"[vis] URDF part {part_id}: {exc}")
            self.obj_part_id = part_id
            self.obj_orn = orn

        if hand_id >= 0:
            spawn_pos = _seated_spawn_pos(part_id, orn)
            hand_pos, hand_quat = _world_hand_pose(spawn_pos, gp)
            p.resetBasePositionAndOrientation(hand_id, hand_pos, hand_quat)
            if hand_model is not None:
                hand_model.reset_open_pose()
            p.stepSimulation()

    def _remove_obj(self) -> None:
        if self.obj_id >= 0:
            try:
                p.removeBody(self.obj_id)
            except Exception:
                pass
            self.obj_id = -1

    def clear(self) -> None:
        self._remove_obj()


# Phase: Greifpunkt auswählen

def _phase_select(robot_id, helper, js, flat_gps: list[dict], idx: int,
                  vis: _Vis, hand_id: int = -1,
                  hand_model=None) -> tuple[str, int]:
    # RB/LB navigieren, A bestätigen. Gibt ("ok"|"quit", idx) zurück.
    while True:
        inp = read_inputs(js)
        if not inp["a"] and not inp["b"]:
            break
        time.sleep(_DT)

    print("\n" + "=" * 60)
    print("  GREIFPUNKT WÄHLEN")
    print("=" * 60)
    print("  RB -> nächster   LB -> vorheriger   A -> Bestätigen   Menu -> Beenden\n")

    prev = dict(a=False, rb=False, lb=False)

    def _show(i: int) -> None:
        e = flat_gps[i]
        print(f"\r  [{i+1}/{len(flat_gps)}]  Part {e['part_id']}  |  "
              f"{e['gp']['id']}  |  {e['gp']['grasp_type']}          ",
              end="", flush=True)
        vis.refresh(e["part_id"], e["gp"], hand_id=hand_id, hand_model=hand_model)

    _show(idx)

    while True:
        inp = read_inputs(js)
        a_trig  = inp["a"]  and not prev["a"]
        rb_trig = inp["rb"] and not prev["rb"]
        lb_trig = inp["lb"] and not prev["lb"]
        prev.update(a=inp["a"], rb=inp["rb"], lb=inp["lb"])

        if inp["menu"]:
            print()
            return "quit", idx
        if a_trig:
            print()
            return "ok", idx
        if rb_trig:
            idx = (idx + 1) % len(flat_gps)
            _show(idx)
        if lb_trig:
            idx = (idx - 1) % len(flat_gps)
            _show(idx)

        p.stepSimulation()
        time.sleep(_DT)


# Phase: At Pregrasp

def _phase_at_pregrasp(
    limb, robot_id, helper, js,
    hand_id: int,
    q_pregrasp: list[float],
    part_id: int,
    gp: dict,
) -> tuple[str, list[float]]:
    # Wartet auf Eingabe: Y=Feinjustierung, A=Lift, B=zurück.
    # Nach Feinjustierung wird q_pregrasp neu per IK berechnet.
    # Gibt ("ok_lift"|"back"|"quit", q_pregrasp) zurück.
    print("\n" + "=" * 60)
    print("  AT PREGRASP")
    print("=" * 60)
    print("  Y    -> Feinjustierung (Offset speichern)")
    print("  A    -> Lift-Modus (+5 cm, X hoch/runter)")
    print("  B    -> Greifpunkt neu wählen")
    print("  Menu -> Beenden\n")

    prev = dict(a=False, b=False, y=False)

    while True:
        inp = read_inputs(js)
        a_trig = inp["a"] and not prev["a"]
        b_trig = inp["b"] and not prev["b"]
        y_trig = inp["y"] and not prev["y"]
        prev.update(a=inp["a"], b=inp["b"], y=inp["y"])

        if inp["menu"]:
            return "quit", q_pregrasp
        if b_trig:
            return "back", q_pregrasp

        if y_trig:
            offset_old = _load_gp_offset(part_id, gp["id"])
            print("\n  [Feinjustierung]  RB/LB = Z  Stick = XY  A = Speichern  B = Abbrechen\n")
            status, _, new_offset = cal_fine_tune(
                limb, robot_id, helper, js, q_pregrasp, offset_old,
                label="Feinjustierung Pregrasp", hand_id=hand_id,
            )
            if status == "ok":
                _save_gp_offset(part_id, gp["id"], new_offset)
                try:
                    q_pregrasp = _compute_pregrasp_q(helper, part_id, gp, new_offset)
                    set_pb_joints(robot_id, helper, q_pregrasp)
                    if hand_id >= 0:
                        update_hand_from_ee(robot_id, helper, hand_id)
                except Exception as exc:
                    print(f"[warn] IK nach Kalibrierung fehlgeschlagen: {exc}")
            elif status == "quit":
                return "quit", q_pregrasp

            print("\n" + "=" * 60)
            print("  AT PREGRASP")
            print("=" * 60)
            print("  Y -> Feinjustierung   A -> Lift   B -> Auswahl   Menu -> Beenden\n")
            prev = dict(a=False, b=False, y=False)

        if a_trig:
            return "ok_lift", q_pregrasp

        p.stepSimulation()
        time.sleep(_DT)


# Phase: Lift

def _phase_lift(
    limb, robot_id, helper, js,
    hand_id: int,
    q_pregrasp: list[float],
) -> str:
    # X-Toggle: jeder X-Druck wechselt zwischen Pregrasp und Pregrasp+LIFT_DZ.
    # drive_x_held mit auto_proceed=True fährt bis 100% und kehrt dann zurück.
    # Gibt "back"|"quit" zurück.
    set_pb_joints(robot_id, helper, q_pregrasp)
    ee_xyz, ee_rpy = helper.ee_pose()
    try:
        q_lifted = helper._ik(
            [float(ee_xyz[0]), float(ee_xyz[1]), float(ee_xyz[2]) + _LIFT_DZ],
            [float(v) for v in ee_rpy],
        ).tolist()
    except Exception as exc:
        print(f"[warn] Lift-IK fehlgeschlagen: {exc}")
        return "back"

    lifted = False

    print("\n" + "=" * 60)
    print("  LIFT-MODUS")
    print("=" * 60)
    print("  X halten -> hoch / runter (togglet bei jedem Druck)")
    print("  B        -> zurück")
    print("  Menu     -> Beenden\n")

    def _status():
        direction = "runter" if lifted else "hoch"
        print(f"  Aktuell: {'OBEN' if lifted else 'UNTEN'}  |  X halten = {direction}\n")

    _status()
    prev_x = prev_b = False

    while True:
        inp   = read_inputs(js)
        x_trig = inp["x"] and not prev_x
        b_trig = inp["b"] and not prev_b
        prev_x = inp["x"]
        prev_b = inp["b"]

        if inp["menu"]:
            return "quit"
        if b_trig:
            return "back"

        if x_trig:
            target_lifted = not lifted
            target_q = q_lifted if target_lifted else q_pregrasp
            label    = f"Lift (+{int(_LIFT_DZ*100)} cm)" if target_lifted else "Zurück (Pregrasp)"

            result = drive_x_held(
                limb, robot_id, helper, js,
                q_real_start=read_q(limb), q_real_target=target_q,
                q_pb_start=read_q(limb),   q_pb_target=target_q,
                label=label, auto_proceed=True,
            )
            if result == "quit":
                return "quit"
            if result != "cancel":
                lifted = target_lifted

            if hand_id >= 0:
                update_hand_from_ee(robot_id, helper, hand_id)

            # Warten bis X losgelassen wird damit kein sofortiger Re-Trigger.
            while read_inputs(js)["x"]:
                time.sleep(_DT)
            prev_x = prev_b = False

            _status()

        p.stepSimulation()
        time.sleep(_DT)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Greifpunkt auswählen, Pregrasp anfahren, Lift testen.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sim-only",   action="store_true",
                        help="Nur PyBullet -- kein echtes Sawyer.")
    parser.add_argument("--robot-rpy",  nargs=3, type=float,
                        default=_ROBOT_BASE_RPY_DEG, metavar=("R", "P", "Y"),
                        help="Sawyer-Basis-Orientierung in Grad.")
    parser.add_argument("--obj-xy",     nargs=2, type=float,
                        default=_OBJ_XY, metavar="M",
                        help="Objekt-Position auf dem Tisch in Metern.")
    args = parser.parse_args()

    js = init_pygame_joystick()

    robot_id, helper    = init_pybullet(args.robot_rpy, args.obj_xy)
    hand_id, hand_model = init_hand(robot_id)

    if args.sim_only:
        limb = SimLimb(robot_id, helper)
        print("[init] Sim-Only Modus -- kein echtes Sawyer.")
    else:
        limb = init_ros_sawyer()

    parts    = _load_parts(_LOOKUP_YAML)
    flat_gps = _flat_gp_list(parts)
    if not flat_gps:
        print("[error] Keine Greifpunkte in der Lookup Table.")
        p.disconnect()
        return

    print(f"[init] {len(flat_gps)} Greifpunkte geladen.")

    vis = _Vis()
    idx = 0

    try:
        while True:
            # Schritt 1: Greifpunkt auswählen
            result, idx = _phase_select(robot_id, helper, js, flat_gps, idx,
                                         vis, hand_id=hand_id, hand_model=hand_model)
            if result == "quit":
                break

            entry   = flat_gps[idx]
            part_id = entry["part_id"]
            gp      = entry["gp"]
            print(f"\n  Part {part_id}  |  {gp['id']}  |  {gp['grasp_type']}")

            # Pregrasp-IK berechnen
            offset = _load_gp_offset(part_id, gp["id"])
            if any(abs(v) > 1e-9 for v in offset):
                print(f"  Vorhandener Offset: {[round(v*1000, 1) for v in offset]} mm")
            else:
                print("  Kein gespeicherter Offset.")

            try:
                q_pregrasp = _compute_pregrasp_q(helper, part_id, gp, offset)
            except Exception as exc:
                print(f"[error] IK fehlgeschlagen: {exc}")
                continue

            # Hand in GUI zur Pregrasp-Pose teleportieren
            obj_orn = [float(v) for v in gp["object_orientation_xyzw"]]
            obj_pos = _seated_spawn_pos(part_id, obj_orn)
            hand_pos, hand_quat = _world_hand_pose(obj_pos, gp)
            if hand_id >= 0:
                p.resetBasePositionAndOrientation(hand_id, hand_pos, hand_quat)
                hand_model.reset_open_pose()
                p.stepSimulation()

            # Schritt 2: Home (Q_SYNC)
            print("\nSchritt 1: Home-Pose (Q_SYNC).")
            q_now = read_q(limb)
            set_pb_joints(robot_id, helper, q_now)
            result = drive_x_held(
                limb, robot_id, helper, js,
                q_real_start=q_now, q_real_target=Q_SYNC,
                q_pb_start=q_now,   q_pb_target=Q_SYNC,
                label="Home (Q_SYNC)", auto_proceed=True,
            )
            if result == "quit":
                break
            if result == "cancel":
                continue

            # Schritt 3: Pregrasp
            print("\nSchritt 2: Pregrasp anfahren.")
            q_now = read_q(limb)
            result = drive_x_held(
                limb, robot_id, helper, js,
                q_real_start=q_now, q_real_target=q_pregrasp,
                q_pb_start=q_now,   q_pb_target=q_pregrasp,
                label="Pregrasp", auto_proceed=True,
            )
            if result == "quit":
                break
            if result == "cancel":
                continue

            if hand_id >= 0:
                update_hand_from_ee(robot_id, helper, hand_id)

            # Schritt 4: At Pregrasp (Feinjustierung / Lift)
            while True:
                status, q_pregrasp = _phase_at_pregrasp(
                    limb, robot_id, helper, js,
                    hand_id, q_pregrasp, part_id, gp,
                )
                if status == "quit":
                    break
                if status == "back":
                    break
                if status == "ok_lift":
                    result = _phase_lift(limb, robot_id, helper, js,
                                         hand_id, q_pregrasp)
                    if result == "quit":
                        status = "quit"

            if status == "quit":
                break

    finally:
        vis.clear()
        p.disconnect()
        try:
            import pygame
            pygame.quit()
        except Exception:
            pass


if __name__ == "__main__":
    main()
