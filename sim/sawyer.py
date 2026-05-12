# Sawyer-Arm: IK/FK-Wrapper und PyBullet-Setup.
# Wird von hardware/sawyer.py für Visualisierung und IK-Berechnungen importiert.
from __future__ import annotations

import math

import numpy as np
import pybullet as p
import pybullet_data

from assets import AR10_URDF, SAWYER_URDF
from sim.hand import HandModel


# Die 7 Arm-Joints in Reihenfolge, wird auch von hardware/sawyer.py für read_q/send_q gebraucht.
_SAWYER_JOINT_NAMES = [
    "right_j0", "right_j1", "right_j2", "right_j3",
    "right_j4", "right_j5", "right_j6",
]


class SawyerHelper:
    # IK/FK-Wrapper für den Sawyer-Arm in PyBullet.

    def __init__(self, robot_id: int, ee_link_name: str = "right_l6",
                 joint_names: list | None = None) -> None:
        self.robot_id = robot_id
        self.ee_link  = self._resolve_link(ee_link_name)
        joint_names   = joint_names or _SAWYER_JOINT_NAMES
        self.joints   = self._resolve_joints(joint_names)
        if len(self.joints) != 7:
            raise RuntimeError(f"Erwarte 7 Arm-Joints, erhalten: {self.joints}")
        self.limits           = self._read_limits()
        self._ik_solution_map = self._build_ik_map()

    def _resolve_link(self, name: str) -> int:
        if isinstance(name, int):
            return name
        for j in range(p.getNumJoints(self.robot_id)):
            if p.getJointInfo(self.robot_id, j)[12].decode() == name:
                return j
        raise RuntimeError(f"Link nicht gefunden: {name}")

    def _resolve_joints(self, names: list) -> list[int]:
        # Gibt Joint-Indizes in der Reihenfolge von names zurück (nicht URDF-Reihenfolge).
        name_to_idx = {}
        for j in range(p.getNumJoints(self.robot_id)):
            n = p.getJointInfo(self.robot_id, j)[1].decode()
            if n in names:
                name_to_idx[n] = j
        missing = [n for n in names if n not in name_to_idx]
        if missing:
            raise RuntimeError(f"Sawyer-Joints fehlen: {missing}")
        return [name_to_idx[n] for n in names]

    def _read_limits(self) -> list[tuple[float, float]]:
        # Fallback auf +-2pi wenn das URDF keine sinnvollen Limits definiert.
        limits = []
        for j in self.joints:
            lo, hi = p.getJointInfo(self.robot_id, j)[8:10]
            if not (hi > lo) or math.isinf(lo) or math.isinf(hi):
                lo, hi = -2 * math.pi, 2 * math.pi
            limits.append((float(lo), float(hi)))
        return limits

    def _build_ik_map(self) -> dict[int, int]:
        # PyBullet calculateInverseKinematics gibt eine Lösung für ALLE beweglichen
        # Joints zurück (nicht nur die 7 Arm-Joints). Wir brauchen die Position jedes
        # Arm-Joints in diesem flachen Array um die Lösung korrekt auszulesen.
        movable = [j for j in range(p.getNumJoints(self.robot_id))
                   if p.getJointInfo(self.robot_id, j)[2] in
                   (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC)]
        result = {}
        for ji in self.joints:
            if ji not in movable:
                raise RuntimeError(f"Joint {ji} ist nicht steuerbar.")
            result[ji] = movable.index(ji)
        return result

    def ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        ls  = p.getLinkState(self.robot_id, self.ee_link, computeForwardKinematics=True)
        xyz = np.array(ls[4], dtype=float)
        rpy = np.array(p.getEulerFromQuaternion(ls[5]), dtype=float)
        return xyz, rpy

    def _ik(self, xyz, rpy, rest: list | None = None) -> np.ndarray:
        # rest=aktuelle Winkel: IK bleibt nah an der aktuellen Konfiguration
        # und wählt unter mehreren Lösungen die ergonomischste aus.
        quat   = p.getQuaternionFromEuler(rpy)
        low    = [l for l, _ in self.limits]
        high   = [h for _, h in self.limits]
        ranges = [h - l for l, h in self.limits]
        if rest is None:
            rest = [p.getJointState(self.robot_id, j)[0] for j in self.joints]
        sol = p.calculateInverseKinematics(
            bodyUniqueId=self.robot_id,
            endEffectorLinkIndex=self.ee_link,
            targetPosition=xyz,
            targetOrientation=quat,
            lowerLimits=low, upperLimits=high,
            jointRanges=ranges, restPoses=rest,
            maxNumIterations=500, residualThreshold=1e-4,
        )
        q = np.array([sol[self._ik_solution_map[j]] for j in self.joints], dtype=float)
        return np.clip(q, np.array(low), np.array(high))


def init_pybullet(
    robot_base_rpy_deg: list[float],
    obj_xy: list[float] = (0.50, 0.00),
    ped_h: float = 0.04,
) -> tuple[int, SawyerHelper]:
    # Startet PyBullet GUI, lädt Sawyer, gibt (robot_id, helper) zurück.
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0.0, 0.0, -9.81)
    p.setRealTimeSimulation(0)
    # Gleiche Physik-Parameter wie GraspEnv
    p.setPhysicsEngineParameter(numSolverIterations=200,
                                fixedTimeStep=1.0 / 240.0,
                                numSubSteps=4)
    p.loadURDF("plane.urdf")
    p.resetDebugVisualizerCamera(
        cameraDistance=1.4, cameraYaw=50.0, cameraPitch=-30.0,
        cameraTargetPosition=[0.6, 0.0, 0.2],
    )

    col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.02, height=ped_h)
    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.02, length=ped_h,
                              rgbaColor=[0.75, 0.75, 0.75, 1.0])
    p.createMultiBody(0, col, vis, [obj_xy[0], obj_xy[1], ped_h * 0.5], [0, 0, 0, 1])

    quat     = p.getQuaternionFromEuler([math.radians(v) for v in robot_base_rpy_deg])
    robot_id = p.loadURDF(str(SAWYER_URDF), basePosition=[0.0, 0.0, 0.0],
                          baseOrientation=quat, useFixedBase=True)
    return robot_id, SawyerHelper(robot_id)


def init_hand(robot_id: int) -> tuple[int, HandModel]:
    # Lädt AR10 als fixen Visualisierungskörper (kein Domain-Randomization, keine Kollision).
    hand_id = p.loadURDF(str(AR10_URDF), basePosition=[0.0, 0.0, -1.0],
                         useFixedBase=True)
    # Kollision Sawyer <-> Hand deaktivieren (nur Visualisierung)
    for la in range(-1, p.getNumJoints(robot_id)):
        for lb in range(-1, p.getNumJoints(hand_id)):
            p.setCollisionFilterPair(robot_id, hand_id, la, lb, 0)

    vis_physics = {
        "motor_force":        {"min": 5.0, "max": 5.0},
        "fingertip_friction": {"min": 2.0, "max": 2.0},
        "position_gain": 10.0,
        "velocity_gain": 1.0,
        "joint_damping": 0.1,
        "max_velocity":  0.5,
    }
    hm = HandModel(hand_id, vis_physics, np.random.default_rng(0))
    hm.reset_open_pose()
    return hand_id, hm


def set_pb_joints(robot_id: int, helper: SawyerHelper, q: list[float]) -> None:
    for idx, qi in zip(helper.joints, q):
        p.resetJointState(robot_id, idx, float(qi), targetVelocity=0.0)
    p.stepSimulation()


def update_hand_from_ee(robot_id: int, helper: SawyerHelper, hand_id: int) -> None:
    # Teleportiert Hand-Modell zur aktuellen Sawyer-EE-Position.
    # rot180z: Hand-URDF-Basis ist 180° um Z gegen die Sawyer-EE-Orientierung gedreht.
    # 3cm Offset: Abstand zwischen Sawyer-EE-Flansch und Hand-Basis.
    ls      = p.getLinkState(robot_id, helper.ee_link, computeForwardKinematics=True)
    rot180z = p.getQuaternionFromEuler([0.0, 0.0, math.pi])
    _, hq   = p.multiplyTransforms([0, 0, 0], list(ls[5]), [0, 0, 0], rot180z)
    hp, _   = p.multiplyTransforms(list(ls[4]), list(ls[5]), [0.0, 0.0, 0.03], [0, 0, 0, 1])
    p.resetBasePositionAndOrientation(hand_id, hp, hq)
