from __future__ import annotations

import numpy as np
import pybullet as p


# Reihenfolge der gesteuerten Gelenke - definiert die Indexierung aller q-Vektoren.
# 0.0 = vollständig offen, 1.0 = vollständig geschlossen (normalisiert).
CONTROL_JOINTS = [
    "servo0", "servo1",   # Daumen
    "servo2", "servo3",   # Pinky
    "servo4", "servo5",   # Ring
    "servo6", "servo7",   # Mittel
    "servo8", "servo9",   # Zeige
]

# PIP-zu-DIP Kopplung: welches DIP-Gelenk welchem PIP folgt.
DIP_MIMIC_MAP = {
    "servo3": "tip1",
    "servo5": "tip2",
    "servo7": "tip3",
    "servo9": "tip4",
}

# Geometrieparameter der 4-Stab-Mechanik aus den URDF <mimic> Tags.
DIP_MULTIPLIER = 0.49
DIP_OFFSET     = 0.16   # rad

# Fingertip-Links an denen Kontakte gemessen werden.
FINGERTIP_EE_MAP = {
    "servo1": "ee5",   # Daumen
    "servo3": "ee1",   # Pinky
    "servo5": "ee2",   # Ring
    "servo7": "ee3",   # Mittel
    "servo9": "ee4",   # Zeige
}

# Startposition Daumen-Abduktion.
SERVO0_INIT = 0.5

# Gear-Constraint sehr stark dass DIP rigid an PIP gekoppelt ist (4-Stab-Mechanik der echten Hand).
# Damit zieht das PIP-Ratchet die Fingerspitze automatisch mit, ohne separate Behandlung.
_GEAR_FORCE = 1.0e9


class HandModel:
    # AR10-Hand in PyBullet, positions-geregelt über normalisierte Zielwerte.
    # Motor-Kraft und Fingerkuppen-Reibung werden pro Episode zufällig gezogen (Domain Randomization).

    def __init__(
        self,
        hand_id:    int,
        physics_cfg: dict,
        rng:        np.random.Generator,
        client_id:  int = 0,
    ) -> None:
        self.hand_id = int(hand_id)
        self._cid    = int(client_id)

        # Pro Episode randomisiert
        self._motor_force = _uniform(physics_cfg["motor_force"], rng)
        self._friction    = _uniform(physics_cfg["fingertip_friction"], rng)

        # Joint-Damping + max_velocity. positionGain/velocityGain werden bewusst NICHT genutzt
        # weil PyBullet sie ignoriert sobald maxVelocity in setJointMotorControl2 gesetzt ist
        # (rate-limitierter Tracker statt PD).
        self._damping  = float(physics_cfg["joint_damping"])
        self._max_vel  = float(physics_cfg["max_velocity"])

        self.joint_index  = self._build_joint_index()
        self.joint_limits = self._load_joint_limits()
        self._q_target: list[float] = [0.0] * len(CONTROL_JOINTS)

        # Bookkeeping für Non-Backdrivable Enforcement (siehe enforce_non_backdrivable).
        # _q_locked: Position auf die das Gelenk vom simulierten Getriebe festgehalten wird.
        # Folgt q_measured nach oben (Finger arbeitet) und q_target nach unten (Motor öffnet).
        self._q_locked = [0.0] * len(CONTROL_JOINTS)

        self._init_dynamics()
        self._setup_dip_constraints()
        self.apply_q_target(self._q_target)

    # Setup
    def _build_joint_index(self) -> dict[str, int]:
        # Liest alle Joint-Namen aus dem URDF und erstellt name → index Mapping.
        idx: dict[str, int] = {}
        for i in range(p.getNumJoints(self.hand_id, physicsClientId=self._cid)):
            name = p.getJointInfo(self.hand_id, i, physicsClientId=self._cid)[1].decode()
            idx[name] = i
        return idx

    def _load_joint_limits(self) -> dict[str, tuple[float, float]]:
        # Liest physikalische Gelenkgrenzen aus dem URDF (in rad).
        # Fallback auf gemessene AR10-Grenzen falls URDF keine gültigen Limits enthält.
        # CONTROL_JOINTS und DIP-Joints, weil non-backdrivable Enforcement beide braucht.
        limits: dict[str, tuple[float, float]] = {}
        for name in CONTROL_JOINTS + list(DIP_MIMIC_MAP.values()):
            info = p.getJointInfo(self.hand_id, self.joint_index[name], physicsClientId=self._cid)
            lo, hi = float(info[8]), float(info[9])
            if lo >= hi:
                lo, hi = 0.17, 1.57
            limits[name] = (lo, hi)
        return limits

    def _init_dynamics(self) -> None:
        # Dämpfung auf alle gesteuerten Gelenke + DIP-Gelenke setzen.
        # Reibung nur auf Fingertip-Links (Gummi-Kappen).
        all_joints = CONTROL_JOINTS + list(DIP_MIMIC_MAP.values())
        for name in all_joints:
            p.changeDynamics(self.hand_id, self.joint_index[name],
                              jointDamping=self._damping, physicsClientId=self._cid)
        for ee_name in FINGERTIP_EE_MAP.values():
            p.changeDynamics(self.hand_id, self.joint_index[ee_name],
                              lateralFriction=self._friction, physicsClientId=self._cid)

    def _setup_dip_constraints(self) -> None:
        # PyBullet ignoriert URDF <mimic> Tags, deshalb wird die PIP-DIP Kopplung
        # manuell über JOINT_GEAR Constraints nachgebaut (starre 4-Stab-Mechanik).
        # Gear-Ratio negativ weil beide Gelenkachsen im URDF in -x Richtung zeigen
        # (gleiche physikalische Richtung -> negativer Ratio für gleichsinnige Kopplung).
        for pip_name, dip_name in DIP_MIMIC_MAP.items():
            # DIP-Motor deaktivieren, sonst kämpft er gegen den Constraint.
            p.setJointMotorControl2(
                self.hand_id, self.joint_index[dip_name],
                controlMode=p.VELOCITY_CONTROL, force=0,
                physicsClientId=self._cid,
            )
            c = p.createConstraint(
                parentBodyUniqueId=self.hand_id,
                parentLinkIndex=self.joint_index[pip_name],
                childBodyUniqueId=self.hand_id,
                childLinkIndex=self.joint_index[dip_name],
                jointType=p.JOINT_GEAR,
                jointAxis=[1, 0, 0],
                parentFramePosition=[0, 0, 0],
                childFramePosition=[0, 0, 0],
                physicsClientId=self._cid,
            )
            p.changeConstraint(c, gearRatio=-1.0 / DIP_MULTIPLIER,
                                maxForce=_GEAR_FORCE, physicsClientId=self._cid)

    # Control
    def apply_q_target(self, q_target: list[float]) -> None:
        # Sendet PD-Positionsregelung an alle CONTROL_JOINTS.
        # DIP-Gelenke folgen über den Gear-Constraint automatisch.
        if len(q_target) != len(CONTROL_JOINTS):
            raise ValueError(f"Expected {len(CONTROL_JOINTS)} values, got {len(q_target)}.")

        self._q_target = [max(0.0, min(1.0, float(v))) for v in q_target]
        for idx, name in enumerate(CONTROL_JOINTS):
            angle = self._norm_to_angle(name, self._q_target[idx])
            p.setJointMotorControl2(
                self.hand_id, self.joint_index[name],
                controlMode=p.POSITION_CONTROL,
                targetPosition=angle,
                maxVelocity=self._max_vel,
                force=self._motor_force,
                physicsClientId=self._cid,
            )

    def teleport_to(self, q_target: list[float]) -> None:
        # Setzt Gelenkwinkel direkt ohne Physik, nur für Episode-Reset verwenden.
        # DIP-Gelenke werden manuell auf die korrekte Startposition gesetzt.
        q = [max(0.0, min(1.0, float(v))) for v in q_target]
        self._q_target = q
        for idx, name in enumerate(CONTROL_JOINTS):
            p.resetJointState(self.hand_id, self.joint_index[name],
                               self._norm_to_angle(name, q[idx]),
                               physicsClientId=self._cid)
        for pip_name, dip_name in DIP_MIMIC_MAP.items():
            pip_angle = float(p.getJointState(self.hand_id, self.joint_index[pip_name],
                                               physicsClientId=self._cid)[0])
            p.resetJointState(self.hand_id, self.joint_index[dip_name],
                               DIP_MULTIPLIER * pip_angle + DIP_OFFSET,
                               physicsClientId=self._cid)
        # Non-Backdrivable Bookkeeping auf die neue Startpose zurücksetzen.
        self._q_locked = list(q)
        self.apply_q_target(q)

    def enforce_non_backdrivable(self) -> None:
        # Simuliert das 100:1 Firgelli-Getriebe symmetrisch: das Gelenk kann ausschließlich
        # durch den Motor bewegt werden. Externe Kräfte können es weder zurück- noch
        # weiter-vor-drücken. q_measured ist nach jedem Substep auf [q_locked, q_target] gedeckelt.
        # _q_locked folgt q_target nach unten (Motor öffnet) und q_measured nach oben (Motor schließt).
        for idx, name in enumerate(CONTROL_JOINTS):
            joint_idx = self.joint_index[name]
            lo, hi = self.joint_limits[name]
            pos = float(p.getJointState(self.hand_id, joint_idx, physicsClientId=self._cid)[0])
            norm = max(0.0, min(1.0, (pos - lo) / (hi - lo)))
            qt   = self._q_target[idx]

            # Motor öffnet aktiv -> Lock zieht nach unten mit dem Target.
            if qt < self._q_locked[idx]:
                self._q_locked[idx] = qt
            # Finger advanciert. Lock folgt, aber nie über das kommandierte Target hinaus.
            if norm > self._q_locked[idx]:
                self._q_locked[idx] = min(norm, qt)

            # Position auf [q_locked, q_target] clampen wenn externe Kraft sie rausgedrückt hat.
            if norm < self._q_locked[idx]:
                # Externe Kraft drückt rückwärts -> festhalten.
                angle = lo + self._q_locked[idx] * (hi - lo)
                p.resetJointState(self.hand_id, joint_idx, angle, 0.0,
                                   physicsClientId=self._cid)
            elif norm > qt:
                # Externe Kraft (z.B. schräger Objekt-Kontakt) hat den Finger weiter geschlossen
                # als der Motor das kommandiert hat -> auf q_target zurücknehmen.
                angle = lo + qt * (hi - lo)
                p.resetJointState(self.hand_id, joint_idx, angle, 0.0,
                                   physicsClientId=self._cid)

        # DIPs (Fingerspitzen-Gelenke) direkt auf die gear-korrekte Position relativ zum
        # gerade geclampten PIP setzen. Das Gear-Constraint allein kann unter schrägen
        # Kontakten kurzzeitig "abreißen" bevor der Solver es wieder einfängt - durch
        # explizites resetJointState wird der Sollwert pro Substep hart durchgesetzt.
        for pip_name, dip_name in DIP_MIMIC_MAP.items():
            pip_joint_idx = self.joint_index[pip_name]
            dip_joint_idx = self.joint_index[dip_name]
            dip_lo, dip_hi = self.joint_limits[dip_name]
            pip_angle = float(p.getJointState(self.hand_id, pip_joint_idx,
                                                physicsClientId=self._cid)[0])
            target_dip_angle = DIP_MULTIPLIER * pip_angle + DIP_OFFSET
            target_dip_angle = max(dip_lo, min(dip_hi, target_dip_angle))
            p.resetJointState(self.hand_id, dip_joint_idx,
                               target_dip_angle, 0.0, physicsClientId=self._cid)

    def reset_open_pose(self) -> None:
        self.teleport_to([0.0] * len(CONTROL_JOINTS))

    # Readout
    def q_target(self) -> list[float]:
        return list(self._q_target)

    def q_measured(self) -> list[float]:
        # Liest aktuelle Gelenkwinkel aus PyBullet und normalisiert auf [0, 1].
        out: list[float] = []
        for name in CONTROL_JOINTS:
            pos = float(p.getJointState(self.hand_id, self.joint_index[name],
                                         physicsClientId=self._cid)[0])
            lo, hi = self.joint_limits[name]
            out.append(max(0.0, min(1.0, (pos - lo) / (hi - lo))))
        return out

    def q_delta_normalized(self) -> list[float]:
        # q_target - q_measured pro Gelenk, geclipt auf [0, 1].
        # Positiver Wert = Finger schließt noch, d.h. Kontakt verhindert Bewegung.
        qt = self.q_target()
        qm = self.q_measured()
        return [max(0.0, min(1.0, t - m)) for t, m in zip(qt, qm)]

    # Helpers
    def _norm_to_angle(self, name: str, norm: float) -> float:
        # Normalisierter Wert [0, 1] -> Winkel in rad.
        lo, hi = self.joint_limits[name]
        return lo + norm * (hi - lo)


def _uniform(bounds: dict, rng: np.random.Generator) -> float:
    return float(rng.uniform(bounds["min"], bounds["max"]))
