# Zeigefingerspitze auf Part 3 kalibrieren.
#
# 3-Schritt-Ablauf:
#   1. Home (Q_SYNC)     : X halten -> Arm fährt zur Home-Pose
#   2. Kalibrierungspose : X halten -> Arm fährt zur berechneten Pose
#   3. Feinjustierung    : Joystick korrigiert EE; akkumulierter Offset wird gespeichert
#
# Gespeichert in: artifacts/calibration/sawyer_offset.yaml

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pybullet as p
import pybullet_data
import yaml

from assets import AR10_URDF, benchmark_part_urdf
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
_CAL_PART_ID = 3
_CAL_N_OBJ   = [0.0, 0.0, 1.0]   # Zielnormale: Finger zeigt von oben auf Part
_CAL_T_OBJ   = [-1.0, 0.0, 0.0]  # Zieltangente: Fingerachse in -X-Richtung

# PyBullet Link-Index der Zeigefingerspitze.
_CAL_TIP_LINK = 23
_CAL_TIP_FWD  = 0.033   # Vorwärts-Offset ab Link-Frame (m)
_CAL_TIP_SIDE = 0.005   # Seiten-Offset ab Link-Frame (m)

_OFFSET_FILE        = Path("artifacts/calibration/sawyer_offset.yaml")
_ROBOT_BASE_RPY_DEG = [0.0, 0.0, 0.0]


# Mathe-Hilfsfunktionen
def _normalize(v: list) -> list:
    n = math.sqrt(sum(float(x) * float(x) for x in v))
    if n <= 1e-12:
        return [0.0, 0.0, 0.0]
    return [float(x) / n for x in v]


def _cross(a: list, b: list) -> list:
    return [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]


def _dot(a: list, b: list) -> float:
    return float(a[0] * b[0] + a[1] * b[1] + a[2] * b[2])


def _orthonormal_tangent(normal: list, tangent_hint: list) -> list:
    # Gram-Schmidt: tangent_hint auf die Ebene senkrecht zu normal projizieren.
    d = _dot(tangent_hint, normal)
    t = _normalize([tangent_hint[i] - d * normal[i] for i in range(3)])
    if _dot(t, t) < 1e-12:
        c = [1.0, 0.0, 0.0] if abs(normal[0]) < 0.9 else [0.0, 1.0, 0.0]
        t = _normalize(_cross(c, normal))
    return t


def _mat_from_basis(x_axis, y_axis, z_axis):
    return [
        [float(x_axis[0]), float(y_axis[0]), float(z_axis[0])],
        [float(x_axis[1]), float(y_axis[1]), float(z_axis[1])],
        [float(x_axis[2]), float(y_axis[2]), float(z_axis[2])],
    ]


def _mat_mul(a, b):
    out = [[0.0, 0.0, 0.0] for _ in range(3)]
    for i in range(3):
        for j in range(3):
            out[i][j] = float(a[i][0]*b[0][j] + a[i][1]*b[1][j] + a[i][2]*b[2][j])
    return out


def _mat_transpose(a):
    return [
        [float(a[0][0]), float(a[1][0]), float(a[2][0])],
        [float(a[0][1]), float(a[1][1]), float(a[2][1])],
        [float(a[0][2]), float(a[1][2]), float(a[2][2])],
    ]


def _mat_vec_mul(mat, vec) -> list:
    return [
        float(mat[0][0]*vec[0] + mat[0][1]*vec[1] + mat[0][2]*vec[2]),
        float(mat[1][0]*vec[0] + mat[1][1]*vec[1] + mat[1][2]*vec[2]),
        float(mat[2][0]*vec[0] + mat[2][1]*vec[1] + mat[2][2]*vec[2]),
    ]


def _quat_from_mat(m) -> list:
    # Shepperd-Methode: numerisch stabile Quaternion aus Rotationsmatrix.
    # Wählt den Zweig mit dem größten Nenner um Division durch ~0 zu vermeiden.
    tr = m[0][0] + m[1][1] + m[2][2]
    if tr > 0.0:
        s  = math.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2][1] - m[1][2]) / s
        qy = (m[0][2] - m[2][0]) / s
        qz = (m[1][0] - m[0][1]) / s
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s  = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
        qw = (m[2][1] - m[1][2]) / s
        qx = 0.25 * s
        qy = (m[0][1] + m[1][0]) / s
        qz = (m[0][2] + m[2][0]) / s
    elif m[1][1] > m[2][2]:
        s  = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
        qw = (m[0][2] - m[2][0]) / s
        qx = (m[0][1] + m[1][0]) / s
        qy = 0.25 * s
        qz = (m[1][2] + m[2][1]) / s
    else:
        s  = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
        qw = (m[1][0] - m[0][1]) / s
        qx = (m[0][2] + m[2][0]) / s
        qy = (m[1][2] + m[2][1]) / s
        qz = 0.25 * s
    return [float(qx), float(qy), float(qz), float(qw)]


# Kalibrierungs-Geometrie
def _get_index_tip_ref_local() -> tuple[list, list, list]:
    # Lädt AR10 in einer temporären DIRECT-Instanz um die Zeigefingerspitze
    # im Hand-Frame zu berechnen — kein GUI, kein State-Einfluss auf die Haupt-Sim.
    cid = p.connect(p.DIRECT)
    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)
        hid = p.loadURDF(str(AR10_URDF), [0, 0, 0], [0, 0, 0, 1],
                         useFixedBase=True, physicsClientId=cid)
        p.stepSimulation(physicsClientId=cid)

        hpos, hquat = p.getBasePositionAndOrientation(hid, physicsClientId=cid)
        ls = p.getLinkState(hid, _CAL_TIP_LINK,
                            computeForwardKinematics=True, physicsClientId=cid)
        twpos  = list(ls[4])
        twquat = list(ls[5])

        m    = p.getMatrixFromQuaternion(twquat)
        zdir = [float(m[2]), float(m[5]), float(m[8])]
        xdir = [float(m[0]), float(m[3]), float(m[6])]
        tip  = [twpos[i] - _CAL_TIP_FWD * zdir[i] - _CAL_TIP_SIDE * xdir[i]
                for i in range(3)]

        inv_pos, inv_quat = p.invertTransform(list(hpos), list(hquat))
        p_local, _ = p.multiplyTransforms(inv_pos, inv_quat, tip, [0, 0, 0, 1])

        mh = p.getMatrixFromQuaternion(hquat)

        def w2h(v):
            return [mh[0]*v[0] + mh[3]*v[1] + mh[6]*v[2],
                    mh[1]*v[0] + mh[4]*v[1] + mh[7]*v[2],
                    mh[2]*v[0] + mh[5]*v[1] + mh[8]*v[2]]

        n_local = _normalize(w2h([-zdir[0], -zdir[1], -zdir[2]]))
        t_local = _orthonormal_tangent(n_local, w2h([-1.0, 0.0, 0.0]))
        return [float(v) for v in p_local], n_local, t_local
    finally:
        p.disconnect(cid)


def _compute_hand_pose_from_ref(
    target_pos, n_obj, t_obj, ref_pos_h, ref_n_h, ref_t_h,
) -> tuple[list, list]:
    # Berechnet Hand-Pose so dass Referenzpunkt (Zeigefingerspitze im Hand-Frame)
    # auf target_pos ausgerichtet ist. Baut je ein Koordinatensystem für
    # Ziel-Frame (n_obj, t_obj) und Referenz-Frame (ref_n_h, ref_t_h) und
    # löst R_hand = R_target * R_ref^T.
    n_t  = [-n_obj[i] for i in range(3)]
    t_t  = _orthonormal_tangent(n_t, t_obj)
    b_t  = _normalize(_cross(n_t, t_t))
    t_t  = _normalize(_cross(b_t, n_t))
    r_wt = _mat_from_basis(t_t, b_t, n_t)

    n_r  = _normalize(ref_n_h)
    t_r  = _orthonormal_tangent(n_r, ref_t_h)
    b_r  = _normalize(_cross(n_r, t_r))
    t_r  = _normalize(_cross(b_r, n_r))
    r_rl = _mat_from_basis(t_r, b_r, n_r)

    r_hw      = _mat_mul(r_wt, _mat_transpose(r_rl))
    hand_quat = _quat_from_mat(r_hw)
    offset    = _mat_vec_mul(r_hw, ref_pos_h)
    hand_pos  = [target_pos[i] - offset[i] for i in range(3)]
    return list(hand_pos), list(hand_quat)


def _hand_to_ik_target(hand_pos, hand_quat) -> tuple[list, list]:
    # Inverse von update_hand_from_ee: Hand-Pose -> Sawyer-EE-Ziel.
    # Rückwärts: rot180z rückgängig machen, -3cm Offset.
    rot180z = p.getQuaternionFromEuler([0.0, 0.0, math.pi])
    _, ee_quat = p.multiplyTransforms([0, 0, 0], hand_quat, [0, 0, 0], rot180z)
    ee_pos, _ = p.multiplyTransforms(hand_pos, ee_quat, [0.0, 0.0, -0.03], [0, 0, 0, 1])
    return list(ee_pos), list(p.getEulerFromQuaternion(ee_quat))


def _seated_obj_pos() -> list:
    # AABB-Min von Part 3 bestimmen und Z so setzen dass das Objekt exakt auf dem Podest steht.
    cid = p.connect(p.DIRECT)
    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)
        urdf = str(benchmark_part_urdf(_CAL_PART_ID))
        tmp = p.loadURDF(urdf, [_OBJ_XY[0], _OBJ_XY[1], _PED_H + 0.5],
                         [0, 0, 0, 1], useFixedBase=True, physicsClientId=cid)
        aabb_min, _ = p.getAABB(tmp, physicsClientId=cid)
        cur_pos, _  = p.getBasePositionAndOrientation(tmp, physicsClientId=cid)
        obj_z = float(cur_pos[2]) + (_PED_H - float(aabb_min[2]))
    finally:
        p.disconnect(cid)
    return [_OBJ_XY[0], _OBJ_XY[1], obj_z]


def _cal_aabb_top() -> list:
    # AABB-Top-Center von Part 3 auf Podest = Kalibrierungsziel für Zeigefingerspitze.
    spawn_pos = _seated_obj_pos()
    cid = p.connect(p.DIRECT)
    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)
        urdf = str(benchmark_part_urdf(_CAL_PART_ID))
        tmp = p.loadURDF(urdf, spawn_pos, [0, 0, 0, 1],
                         useFixedBase=True, physicsClientId=cid)
        _, aabb_max = p.getAABB(tmp, physicsClientId=cid)
    finally:
        p.disconnect(cid)
    return [spawn_pos[0], spawn_pos[1], float(aabb_max[2])]


# Offset I/O
def _load_offset() -> list[float]:
    if _OFFSET_FILE.exists():
        with _OFFSET_FILE.open(encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        off = data.get("cal_ee_offset_xyz")
        if off:
            return [float(v) for v in off]
    return [0.0, 0.0, 0.0]


def _save_offset(offset: list[float]) -> None:
    _OFFSET_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = {"cal_ee_offset_xyz": [round(float(v), 6) for v in offset]}
    with _OFFSET_FILE.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False)
    print(f"[cal] Offset gespeichert: {_OFFSET_FILE}")
    print(f"      cal_ee_offset_xyz = {[round(v*1000, 2) for v in offset]} mm")


# Kalibrierungs-Ablauf
def run_calibration(limb, robot_id: int, helper: SawyerHelper,
                    hand_id: int, hand_model, js) -> str:
    # Führt den vollständigen 3-Schritt-Ablauf durch. Gibt "ok"|"cancel"|"quit" zurück.
    print("\n" + "=" * 60)
    print("  KALIBRIERUNG -- Zeigefinger auf Part 3")
    print("=" * 60)

    # Part 3 in GUI laden (visueller Referenzpunkt)
    obj_vis_id = -1
    try:
        obj_vis_id = p.loadURDF(
            str(benchmark_part_urdf(_CAL_PART_ID)),
            basePosition=_seated_obj_pos(),
            baseOrientation=[0, 0, 0, 1],
            useFixedBase=True,
        )
    except Exception as exc:
        print(f"[warn] Part {_CAL_PART_ID} konnte nicht geladen werden: {exc}")

    try:
        # Kalibrierungsziel aus URDF-Geometrie berechnen
        aabb_top = _cal_aabb_top()
        print(f"  Kalibrierungsziel (AABB-Top Part {_CAL_PART_ID}): "
              f"{[round(v, 4) for v in aabb_top]}")

        tip_pos_h, tip_n_h, tip_t_h = _get_index_tip_ref_local()

        hand_pos, hand_quat = _compute_hand_pose_from_ref(
            aabb_top, _CAL_N_OBJ, _CAL_T_OBJ, tip_pos_h, tip_n_h, tip_t_h,
        )

        ee_offset = _load_offset()
        print(f"  Gespeicherter Offset: {[round(v*1000, 1) for v in ee_offset]} mm")

        if hand_id >= 0:
            p.resetBasePositionAndOrientation(hand_id, hand_pos, hand_quat)
            if hand_model is not None:
                hand_model.reset_open_pose()
            p.stepSimulation()

        ee_pos, ee_rpy = _hand_to_ik_target(hand_pos, hand_quat)
        ee_pos_off = [ee_pos[i] + ee_offset[i] for i in range(3)]

        q_current = [float(p.getJointState(robot_id, j)[0]) for j in helper.joints]
        try:
            q_cal = helper._ik(ee_pos_off, ee_rpy, rest=q_current).tolist()
        except Exception as exc:
            print(f"  [IK Fehler] {exc}")
            return "cancel"

        print(f"  IK-Lösung (deg): {[round(math.degrees(v), 1) for v in q_cal]}")

        # Schritt 1: Home-Pose (Q_SYNC)
        print("\nSchritt 1: Home-Pose anfahren.")
        q_now = read_q(limb)
        set_pb_joints(robot_id, helper, q_now)
        result = drive_x_held(
            limb, robot_id, helper, js,
            q_real_start=q_now,    q_real_target=Q_SYNC,
            q_pb_start=q_now,      q_pb_target=Q_SYNC,
            label="Home (Q_SYNC)", auto_proceed=True,
        )
        if result in ("cancel", "quit"):
            return result

        # Schritt 2: Kalibrierungspose
        print("\nSchritt 2: Kalibrierungspose anfahren.")
        q_now = read_q(limb)
        result = drive_x_held(
            limb, robot_id, helper, js,
            q_real_start=q_now,  q_real_target=q_cal,
            q_pb_start=q_now,    q_pb_target=q_cal,
            label="Kalibrierungspose", auto_proceed=True,
        )
        if result in ("cancel", "quit"):
            return result

        if hand_id >= 0:
            update_hand_from_ee(robot_id, helper, hand_id)

        # Schritt 3: Feinjustierung + Offset speichern
        print("\nSchritt 3: Feinjustierung.")
        status, _, new_offset = cal_fine_tune(
            limb, robot_id, helper, js, q_cal, ee_offset,
            label="Feinjustierung -- Kalibrierung", hand_id=hand_id,
        )
        if status in ("cancel", "quit"):
            print("[abgebrochen] Kein Offset gespeichert.")
            return status

        print(f"\n[offset] {[round(v*1000, 2) for v in new_offset]} mm")
        _save_offset(new_offset)
        return "ok"

    finally:
        if obj_vis_id >= 0:
            try:
                p.removeBody(obj_vis_id)
            except Exception:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Zeigefingerspitze auf Part 3 kalibrieren -- kein manuelles YAML nötig.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sim-only",   action="store_true",
                        help="Nur PyBullet -- kein echtes Sawyer.")
    parser.add_argument("--robot-rpy",  nargs=3, type=float,
                        default=_ROBOT_BASE_RPY_DEG, metavar=("R", "P", "Y"),
                        help="Sawyer-Basis-Orientierung in Grad.")
    parser.add_argument("--obj-xy",     nargs=2, type=float,
                        default=_OBJ_XY, metavar="M",
                        help="Objekt-/Referenz-Position auf dem Tisch in Metern.")
    args = parser.parse_args()

    js = init_pygame_joystick()

    robot_id, helper    = init_pybullet(args.robot_rpy, args.obj_xy)
    hand_id, hand_model = init_hand(robot_id)

    if args.sim_only:
        limb = SimLimb(robot_id, helper)
        print("[init] Sim-Only Modus -- kein echtes Sawyer.")
    else:
        limb = init_ros_sawyer()

    try:
        result = run_calibration(limb, robot_id, helper, hand_id, hand_model, js)
        print(f"\n[fertig] Ergebnis: {result}")
    finally:
        p.disconnect()
        try:
            import pygame
            pygame.quit()
        except Exception:
            pass


if __name__ == "__main__":
    main()
