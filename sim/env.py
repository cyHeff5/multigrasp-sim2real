from __future__ import annotations

import collections
import time
from pathlib import Path

import gymnasium
import numpy as np
import pybullet as p
import pybullet_data

from sim.hand     import CONTROL_JOINTS, SERVO0_INIT, HandModel
from sim.object   import GraspObject, apply_pedestal_magnet
from sim.pregrasp import compute_pregrasp
from sim.reward   import step_reward, terminal_reward
from sim.sampler  import sample_object_spec


_PEDESTAL_RADIUS = 0.02
_SPAWN_XY        = [0.0, 0.0]

_HAND_URDF = str(
    Path(__file__).resolve().parent.parent / "assets" / "ar10_description" / "urdf" / "ar10.urdf"
)


class GraspEnv(gymnasium.Env):
    # Config-getriebenes Gymnasium-Environment für AR10 Greiftraining.
    # Observation: binär pro Finger (1 Bit = Kontakt ja/nein).
    # Action: [-1, 1] pro aktivem Gelenk; nur servo0 ist bidirektional, alle anderen nur schließend.

    metadata = {"render_modes": ["human"]}

    def __init__(self, config: dict, render_mode: str | None = None) -> None:
        super().__init__()
        self.cfg         = config
        self.grasp_type  = config["grasp_type"]
        self.render_mode = render_mode

        self._active        = config["active_joints"]
        self._finger_joints = config["finger_joints"]
        self._fingers       = list(self._finger_joints.keys())

        self.observation_space = gymnasium.spaces.Box(
            low=0.0, high=1.0,
            shape=(len(self._fingers) + len(self._active),),
            dtype=np.float32,
        )
        self.action_space = gymnasium.spaces.Box(
            low=-1.0, high=1.0, shape=(len(self._active),), dtype=np.float32,
        )

        self._rng = np.random.default_rng()
        self._cid: int | None = None

    # Reset
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Jede Episode bekommt eine frische PyBullet-Instanz um sicherzustellen dass kein State aus der letzten Episode übrig bleibt.
        if self._cid is not None:
            p.disconnect(self._cid)

        mode      = p.GUI if self.render_mode == "human" else p.DIRECT
        self._cid = p.connect(mode)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self._cid)
        p.setAdditionalSearchPath(
            str(Path(__file__).resolve().parent.parent / "assets"),
            physicsClientId=self._cid,
        )
        p.setGravity(0, 0, -9.81, physicsClientId=self._cid)
        p.setTimeStep(1.0 / self.cfg["episode"]["sim_hz"], physicsClientId=self._cid)
        p.loadURDF(pybullet_data.getDataPath() + "/plane.urdf", physicsClientId=self._cid)

        # Podest
        ped_h = self.cfg["episode"].get("pedestal_height", 0.04)
        col = p.createCollisionShape(p.GEOM_CYLINDER, radius=_PEDESTAL_RADIUS,
                                      height=ped_h, physicsClientId=self._cid)
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=_PEDESTAL_RADIUS,
                                   length=ped_h, rgbaColor=[0.5, 0.5, 0.5, 1.0],
                                   physicsClientId=self._cid)
        self._pedestal_id = p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=col, baseVisualShapeIndex=vis,
            basePosition=[_SPAWN_XY[0], _SPAWN_XY[1], ped_h / 2],
            physicsClientId=self._cid,
        )
        self._pedestal_h = ped_h

        # Objekt: zufällig sampeln oder aus options übernehmen (Benchmark-Eval).
        # Masse und Reibung werden immer neu gezogen — auch wenn obj_spec übergeben wurde.
        opts = options or {}
        self._obj_spec = dict(
            opts.get("obj_spec") or sample_object_spec(self.cfg["sampler"], self._rng)
        )
        sampler = self.cfg["sampler"]
        self._obj_spec["mass_kg"] = float(self._rng.uniform(
            sampler["mass_kg"]["min"], sampler["mass_kg"]["max"]
        ))
        self._obj_spec["lateral_friction"] = float(self._rng.uniform(
            sampler["lateral_friction"]["min"], sampler["lateral_friction"]["max"]
        ))
        self._obj = GraspObject.spawn(
            self._obj_spec, ped_h, _SPAWN_XY, self._cid,
        )

        # Pregrasp-Pose: berechnen oder aus options übernehmen (Benchmark-Eval).
        pre_cfg  = self.cfg["pregrasp"]
        pregrasp = opts.get("pregrasp")
        if pregrasp is not None:
            hand_pos, hand_quat = list(pregrasp[0]), list(pregrasp[1])
        else:
            hand_pos, hand_quat, _ = compute_pregrasp(
                self.grasp_type, self._obj.position(), self._obj_spec, self._rng,
                distance_m  = pre_cfg["distance_m"],
                jitter_xy_m = pre_cfg["jitter_xy_m"],
                jitter_z_m  = pre_cfg["jitter_z_m"],
            )

        # Hand laden und auf Pregrasp-Pose teleportieren.
        self._hand_id = p.loadURDF(
            _HAND_URDF, basePosition=hand_pos, baseOrientation=hand_quat,
            physicsClientId=self._cid,
        )
        self._hand = HandModel(
            self._hand_id, self.cfg["physics"], self._rng, client_id=self._cid,
        )
        start_q = [SERVO0_INIT if i == 0 else 0.0 for i in range(len(CONTROL_JOINTS))]
        self._hand.teleport_to(start_q)

        # Anchor-Body: unsichtbarer Körper der per JOINT_FIXED Constraint die Hand-Basis
        # in der Luft fixiert. Nur der Anchor wird bewegt (z.B. beim Lift-Test),
        # die Hand folgt durch den Constraint.
        self._anchor_id = p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=-1,
            basePosition=hand_pos, baseOrientation=hand_quat,
            physicsClientId=self._cid,
        )
        c = p.createConstraint(
            self._anchor_id, -1, self._hand_id, -1, p.JOINT_FIXED,
            [0, 0, 0], [0, 0, 0], [0, 0, 0], physicsClientId=self._cid,
        )
        p.changeConstraint(c, maxForce=500, erp=0.95, physicsClientId=self._cid)

        # Settle-Phase ohne Kollision: Hand kann sich in Startposition einpendeln
        # ohne das Objekt zu berühren. Danach Kollision wieder aktivieren.
        p.setCollisionFilterPair(self._hand_id, self._obj.object_id, -1, -1,
                                  enableCollision=0, physicsClientId=self._cid)
        for _ in range(self.cfg["episode"]["settle_steps"]):
            p.stepSimulation(physicsClientId=self._cid)
        p.setCollisionFilterPair(self._hand_id, self._obj.object_id, -1, -1,
                                  enableCollision=1, physicsClientId=self._cid)

        if self.render_mode == "human":
            p.resetDebugVisualizerCamera(
                cameraDistance=0.5, cameraYaw=-48.0, cameraPitch=-14.8,
                cameraTargetPosition=[0.0, 0.0, 0.1], physicsClientId=self._cid,
            )

        self._step_count             = 0
        self._consecutive_contact    = 0
        self._lift_triggered         = False
        self._stabilization_left     = 0
        self._pedestal_hit_ever      = False

        return self._observation(), {
            "grasp_type": self.grasp_type,
            "shape":      self._obj_spec["shape"],
        }

    # Step 
    def step(self, action: np.ndarray):
        delta  = self._action_to_delta(action)
        next_q = [max(0.0, min(1.0, q + d)) for q, d in zip(self._hand.q_target(), delta)]
        self._apply_pip_caps(next_q)
        self._hand.apply_q_target(next_q)

        # Magnet wird pro Substep angewendet damit die Kraft zur Simulationsfrequenz passt.
        for _ in range(self.cfg["episode"]["substeps"]):
            if self.cfg.get("magnet", {}).get("enabled", False):
                apply_pedestal_magnet(self._obj.object_id, _SPAWN_XY,
                                       self.cfg["magnet"]["k"], self._cid)
            p.stepSimulation(physicsClientId=self._cid)
            if self.render_mode == "human":
                time.sleep(1.0 / self.cfg["episode"]["sim_hz"])

        self._step_count += 1
        obs       = self._observation()
        n_contact = int(obs.sum())

        # Trigger State Machine (Westling & Johansson 1984):
        # trigger_confirmation_steps consecutive frames mit >= trigger_n Kontakten -> Trigger.
        if n_contact >= self.cfg["trigger_n"]:
            self._consecutive_contact += 1
        else:
            self._consecutive_contact = 0

        if (not self._lift_triggered
                and self._consecutive_contact >= self.cfg["trigger_confirmation_steps"]):
            self._lift_triggered     = True
            self._stabilization_left = self.cfg["stabilization_steps"]

        pedestal_now = self._check_pedestal_contact()
        if pedestal_now:
            self._pedestal_hit_ever = True
        reward = step_reward(n_contact, self.cfg["n_target"], pedestal_now, self.cfg["reward"])

        # Drop: Objekt zu stark gekippt oder unter Podest-Niveau gefallen.
        if self._object_dropped():
            reward += terminal_reward(lifted=False, cfg=self.cfg["reward"])
            return obs, reward, True, False, self._info(n_contact, lifted=False, dropped=True)

        # Nach Stabilisierung: Lift-Test.
        if self._lift_triggered:
            self._stabilization_left -= 1
            if self._stabilization_left <= 0:
                lifted = self._run_lift_test()
                reward += terminal_reward(lifted, cfg=self.cfg["reward"])
                return obs, reward, True, False, self._info(n_contact, lifted=lifted)

        # Kein Trigger bis max_steps -> trotzdem Lift-Test.
        if self._step_count >= self.cfg["episode"]["max_steps"]:
            lifted = self._run_lift_test()
            reward += terminal_reward(lifted, cfg=self.cfg["reward"])
            return obs, reward, True, False, self._info(n_contact, lifted=lifted)

        return obs, reward, False, False, self._info(n_contact, lifted=False)

    # Observation 
    def _observation(self) -> np.ndarray:
        # Kontakt-Bits: 1 wenn mindestens ein Joint des Fingers q_delta > threshold.
        threshold = self.cfg["observation"]["threshold"]
        delta_all = self._hand.q_delta_normalized()
        contact = np.zeros(len(self._fingers), dtype=np.float32)
        for i, finger in enumerate(self._fingers):
            for joint_name in self._finger_joints[finger]:
                joint_idx = CONTROL_JOINTS.index(joint_name)
                if delta_all[joint_idx] > threshold:
                    contact[i] = 1.0
                    break

        # Propriozeption: normalisierte q_target der aktiven Joints ([0, 1]).
        q_all = self._hand.q_target()
        q_active = np.array(
            [q_all[CONTROL_JOINTS.index(j)] for j in self._active],
            dtype=np.float32,
        )
        return np.concatenate([contact, q_active])

    # Action mapping
    def _action_to_delta(self, action: np.ndarray) -> list[float]:
        # servo0 (Daumen-Abduktion) bidirektional, alle anderen Joints nur schließend.
        delta = [0.0] * len(CONTROL_JOINTS)
        for i, joint_name in enumerate(self._active):
            a = float(action[i])
            if joint_name == "servo0":
                d = a * self.cfg["action"]["thumb_abduction_delta"]
            else:
                d = max(0.0, a) * self.cfg["action"]["delta_norm"]
            delta[CONTROL_JOINTS.index(joint_name)] = d
        return delta

    def _apply_pip_caps(self, q_target: list[float]) -> None:
        # Daumen-Abduktion auf erlaubten Bereich begrenzen.
        # PIP-Caps verhindern dass Finger das Objekt überfahren (nur Precision).
        rng_ = self.cfg["action"]["thumb_abduction_range"]
        idx0 = CONTROL_JOINTS.index("servo0")
        q_target[idx0] = max(rng_[0], min(rng_[1], q_target[idx0]))
        for joint_name, cap in self.cfg["action"]["pip_caps"].items():
            idx = CONTROL_JOINTS.index(joint_name)
            q_target[idx] = min(cap, q_target[idx])

    # Termination checks 
    def _check_pedestal_contact(self) -> bool:
        from sim.hand import FINGERTIP_EE_MAP
        for ee in FINGERTIP_EE_MAP.values():
            contacts = p.getContactPoints(
                self._hand_id, self._pedestal_id,
                linkIndexA=self._hand.joint_index[ee],
                physicsClientId=self._cid,
            )
            if contacts:
                return True
        return False

    def _object_dropped(self) -> bool:
        _, quat = p.getBasePositionAndOrientation(
            self._obj.object_id, physicsClientId=self._cid,
        )
        euler = p.getEulerFromQuaternion(quat)
        tilt  = max(abs(euler[0]), abs(euler[1]))
        return (tilt > self.cfg["lift"]["drop_tilt_rad"]
                or self._obj.height() < self._pedestal_h * 0.3)

    def _run_lift_test(self) -> bool:
        # Bewegt den Anchor-Body (nicht die Hand direkt) um lift_height nach oben.
        # Die Hand folgt über den JOINT_FIXED Constraint.
        # Erfolg wenn das Objekt >= success_m mitgehoben wird.
        pos, quat = p.getBasePositionAndOrientation(
            self._hand_id, physicsClientId=self._cid,
        )
        z0       = self._obj.height()
        lift_cfg = self.cfg["lift"]

        for step in range(lift_cfg["lift_steps"] + lift_cfg["hold_steps"]):
            if step < lift_cfg["lift_steps"]:
                frac    = (step + 1) / lift_cfg["lift_steps"]
                new_pos = [pos[0], pos[1], pos[2] + frac * lift_cfg["height_m"]]
                p.resetBasePositionAndOrientation(
                    self._anchor_id, new_pos, quat, physicsClientId=self._cid,
                )
            p.stepSimulation(physicsClientId=self._cid)
            if self.render_mode == "human":
                time.sleep(1.0 / self.cfg["episode"]["sim_hz"])

        return (self._obj.height() - z0) >= lift_cfg["success_m"]

    # Info dict 
    def _info(self, n_contact: int, lifted: bool, dropped: bool = False) -> dict:
        return {
            "n_contact":       n_contact,
            "lifted":          lifted,
            "dropped":         dropped,
            "lift_triggered":  self._lift_triggered,
            "pedestal_hit":    self._pedestal_hit_ever,
            "step_count":      self._step_count,
        }

    # Cleanup 
    def close(self) -> None:
        if self._cid is not None:
            p.disconnect(self._cid)
            self._cid = None
