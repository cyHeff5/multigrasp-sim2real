from __future__ import annotations

import math
import numpy as np


# Referenzpunkte des AR10 im Hand-Frame, einmal in der Sim kalibriert.
# Power:     Handflächenmitte
# Precision: thumb_tip
POWER_REF = {
    "position": np.array([-0.01475,  0.013732, 0.115982]),
    "normal":   np.array([ 0.0,      0.992775, 0.11999]),
    "tangent":  np.array([ 1.0,      0.0,      0.0]),
}

PRECISION_REF = {
    "position":          np.array([-0.03875,  0.01283,  0.121998]),
    "normal":            np.array([-0.027817, 0.992852, 0.116067]),
    "tangent":           np.array([ 0.231399, 0.119354,-0.96551]),
    "thumb_tip":         np.array([-0.040079, 0.081893, 0.100338]),
    "thumb_tip_normal":  np.array([ 0.0,      0.0,      1.0]),
    "thumb_tip_tangent": np.array([ 0.0,      1.0,      0.0]),
}


def compute_pregrasp(
    grasp_type: str,
    obj_pos:    list[float],
    obj_spec:   dict,
    rng:        np.random.Generator,
    distance_m: float,
    jitter_xy_m: float,
    jitter_z_m:  float,
) -> tuple[list[float], list[float], list[float]]:
    # Berechnet Hand-Pose (Position + Quaternion) und Kontaktpunkt für die Pregrasp-Position.
    # Der Referenzpunkt (Handfläche bei Power, Daumen-Tip bei Precision) wird auf
    # den Kontaktpunkt am Objekt ausgerichtet, mit distance_m Abstand zur Oberfläche.
    if grasp_type not in ("power", "precision"):
        raise ValueError(f"Unknown grasp type: {grasp_type!r}")

    ref     = POWER_REF if grasp_type == "power" else PRECISION_REF
    shape   = obj_spec["shape"]
    radius  = _object_radius(obj_spec)

    approach_dir  = _approach_direction(grasp_type, shape, obj_spec, rng)
    contact_point = np.array(obj_pos) + radius * approach_dir

    if grasp_type == "precision":
        hand_pos, R = _precision_pose(obj_pos, approach_dir, radius, ref, shape, distance_m)
    else:
        hand_pos, R = _power_pose(obj_pos, approach_dir, radius, ref, shape, distance_m)

    hand_pos[0] += rng.uniform(-jitter_xy_m, jitter_xy_m)
    hand_pos[1] += rng.uniform(-jitter_xy_m, jitter_xy_m)
    hand_pos[2] += rng.uniform(-jitter_z_m,  jitter_z_m)

    return hand_pos.tolist(), _rot_to_quat(R), contact_point.tolist()


# Anflugrichtung (formabhängig)
def _approach_direction(grasp_type, shape, obj_spec, rng):
    # URDF-Objekte kommen hier nie an — ihr Greifpunkt steht in der Lookup Table.
    if grasp_type == "power" and shape in ("sphere", "cube"):
        # Power Grasp: Handfläche von oben, Finger schliessen seitlich.
        return np.array([0.0, 0.0, 1.0])
    if shape in ("sphere", "cylinder"):
        # Rotationssymmetrisch -> zufälliger seitlicher Anflugwinkel.
        theta = rng.uniform(0.0, 2.0 * math.pi)
        return np.array([math.cos(theta), math.sin(theta), 0.0])
    # Cube/rect_cylinder: Anflug senkrecht zu einer Seitenfläche.
    yaw = float(obj_spec["yaw_rad"])
    n   = 4 if shape == "cube" else 2
    k   = rng.integers(0, n)
    angle = yaw + k * (2 * math.pi / n)
    return np.array([math.cos(angle), math.sin(angle), 0.0])


# Hand-Pose Berechnung
def _precision_pose(obj_pos, approach_dir, radius, ref, shape, distance_m):
    # Richtet den Daumen-Tip-Frame auf den Kontaktpunkt aus (Frame-Alignment via Rotationsmatrix).
    n_target = np.array([-approach_dir[0], -approach_dir[1], 0.0])
    t_hint   = (np.cross([0, 0, 1.0], approach_dir)
                if shape in ("cylinder", "rect_cylinder")
                else np.array([0.0, 0.0, -1.0]))
    b_target = _normalize(np.cross(n_target, t_hint))
    t_target = np.cross(b_target, n_target)

    n_thumb = ref["thumb_tip_normal"]
    b_thumb = _normalize(np.cross(n_thumb, ref["thumb_tip_tangent"]))
    t_thumb = np.cross(b_thumb, n_thumb)

    W = np.column_stack([t_target, b_target, n_target])
    L = np.column_stack([t_thumb,  b_thumb,  n_thumb])
    R = W @ L.T

    gp = np.array(obj_pos, dtype=float)
    gp[0] += (radius + distance_m) * approach_dir[0]
    gp[1] += (radius + distance_m) * approach_dir[1]
    hand_pos = gp - R @ ref["thumb_tip"]
    return hand_pos, R


def _power_pose(obj_pos, approach_dir, radius, ref, shape, distance_m):
    # Richtet den Handflächenreferenzpunkt auf den Kontaktpunkt aus.
    ref_world = np.array(obj_pos) + (radius + distance_m) * approach_dir
    t_local   = (-ref["tangent"]
                 if shape in ("cylinder", "rect_cylinder")
                 else ref["tangent"])
    R         = _build_rotation(ref["normal"], t_local, -approach_dir)
    hand_pos  = ref_world - R @ ref["position"]
    return hand_pos, R


# Geometrie-Hilfsfunktionen
def _object_radius(spec: dict) -> float:
    # Gibt den "Greifradius" zurück — bei rect_cylinder ist das thickness (die kürzere Seite).
    shape = spec["shape"]
    if shape in ("sphere", "cube"):
        return float(spec["size_cm"]) / 200.0
    if shape in ("cylinder", "rect_cylinder"):
        return float(spec["thickness_cm"]) / 200.0
    raise ValueError(f"Unknown shape: {shape!r}")


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def _build_rotation(n_local, t_local, n_world):
    # Baut Rotationsmatrix die n_local auf n_world abbildet.
    # Fallback auf X-Achse wenn n_world nahezu vertikal (verhindert Gimbal Lock).
    up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(up, n_world)) > 0.99:
        up = np.array([1.0, 0.0, 0.0])
    x_world = _normalize(up - np.dot(up, n_world) * n_world)
    y_world = np.cross(n_world, x_world)

    b_local = np.cross(n_local, t_local)
    L = np.column_stack([t_local, b_local, n_local])
    W = np.column_stack([x_world, y_world, n_world])
    return W @ L.T


def _rot_to_quat(R: np.ndarray) -> list[float]:
    # Rotationsmatrix -> Quaternion [x, y, z, w] nach Shepperd's Methode.
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 0.5 / math.sqrt(tr + 1.0)
        return [
            (R[2, 1] - R[1, 2]) * s,
            (R[0, 2] - R[2, 0]) * s,
            (R[1, 0] - R[0, 1]) * s,
            0.25 / s,
        ]
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        return [
            0.25 * s,
            (R[0, 1] + R[1, 0]) / s,
            (R[0, 2] + R[2, 0]) / s,
            (R[2, 1] - R[1, 2]) / s,
        ]
    if R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        return [
            (R[0, 1] + R[1, 0]) / s,
            0.25 * s,
            (R[1, 2] + R[2, 1]) / s,
            (R[0, 2] - R[2, 0]) / s,
        ]
    s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
    return [
        (R[0, 2] + R[2, 0]) / s,
        (R[1, 2] + R[2, 1]) / s,
        0.25 * s,
        (R[1, 0] - R[0, 1]) / s,
    ]
