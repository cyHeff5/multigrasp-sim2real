# Kalibriert den real_threshold: den q_delta-Wert der echten Hand der dem
# Sim-Kontakt entspricht. Sim und echte Hand schliessen servo9 synchron gegen
# einen Würfel. Sobald die Sim Kontakt erkennt, wird der echte q_delta gespeichert.
# Median über N Trials -> real_threshold.yaml
#
# Usage:
#   python -m eval.threshold_calibration --config configs/precision.yaml --port COM4
#   python -m eval.threshold_calibration --config configs/precision.yaml   # mock mode
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data
import yaml

from hardware.ar10     import AR10Interface
from hardware.gamepad  import init_pygame_joystick, read_inputs
from sim.hand          import CONTROL_JOINTS, HandModel, SERVO0_INIT
from sim.object        import GraspObject
from sim.pregrasp      import compute_pregrasp


_REPO_ROOT = Path(__file__).resolve().parent.parent
_HAND_URDF = str(_REPO_ROOT / "assets" / "ar10_description" / "urdf" / "ar10.urdf")
_OUT_PATH  = _REPO_ROOT / "artifacts" / "calibration" / "real_threshold.yaml"

_PED_RADIUS = 0.02
_SPAWN_XY   = [0.0, 0.0]
_DT         = 1.0 / 50.0
_CLOSE_RATE = 0.005   # gleicher Wert wie delta_norm, identische Schliessgeschwindigkeit

# Fester Kalibrierungswürfel.
_CUBE_SPEC = {
    "shape": "cube", "size_cm": 5.0,
    "mass_kg": 0.05, "lateral_friction": 0.7, "yaw_rad": 0.0,
}

# Nur servo9 (Index-PIP) wird kalibriert.
_INDEX_MCP_IDX = CONTROL_JOINTS.index("servo8")
_INDEX_PIP_IDX = CONTROL_JOINTS.index("servo9")
_HAND_DOF      = len(CONTROL_JOINTS)


def _setup_sim(cfg: dict):
    # Sim-Szene aufbauen: Podest + fixierter Würfel + Hand in Precision Pregrasp.
    # Jitter=0 damit jeder Trial exakt gleich startet.
    cid = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)
    p.setAdditionalSearchPath(str(_REPO_ROOT / "assets"), physicsClientId=cid)
    p.setGravity(0, 0, -9.81, physicsClientId=cid)
    p.setTimeStep(1.0 / cfg["episode"]["sim_hz"], physicsClientId=cid)
    p.loadURDF(pybullet_data.getDataPath() + "/plane.urdf", physicsClientId=cid)
    p.resetDebugVisualizerCamera(0.5, 45.0, -20.0, [0.0, 0.0, 0.1],
                                  physicsClientId=cid)

    ped_h = cfg["episode"].get("pedestal_height", 0.04)
    col = p.createCollisionShape(p.GEOM_CYLINDER, radius=_PED_RADIUS,
                                  height=ped_h, physicsClientId=cid)
    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=_PED_RADIUS,
                               length=ped_h, rgbaColor=[0.5, 0.5, 0.5, 1.0],
                               physicsClientId=cid)
    p.createMultiBody(0, col, vis,
                       basePosition=[_SPAWN_XY[0], _SPAWN_XY[1], ped_h / 2],
                       physicsClientId=cid)

    obj = GraspObject.spawn(_CUBE_SPEC, ped_h, _SPAWN_XY, cid)
    p.changeDynamics(obj.object_id, -1, mass=0, physicsClientId=cid)  # Würfel fixieren

    rng = np.random.default_rng(42)
    pre_cfg = cfg["pregrasp"]
    hand_pos, hand_quat, _ = compute_pregrasp(
        "precision", obj.position(), _CUBE_SPEC, rng,
        distance_m=pre_cfg["distance_m"], jitter_xy_m=0.0, jitter_z_m=0.0,
    )

    hand_id = p.loadURDF(_HAND_URDF, basePosition=hand_pos,
                          baseOrientation=hand_quat, physicsClientId=cid)
    hand    = HandModel(hand_id, cfg["physics"], rng, client_id=cid)
    pregrasp_q = [SERVO0_INIT if i == 0 else 0.0 for i in range(_HAND_DOF)]
    hand.teleport_to(pregrasp_q)

    anchor = p.createMultiBody(
        baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=-1,
        basePosition=hand_pos, baseOrientation=hand_quat, physicsClientId=cid,
    )
    c = p.createConstraint(anchor, -1, hand_id, -1, p.JOINT_FIXED,
                            [0, 0, 0], [0, 0, 0], [0, 0, 0], physicsClientId=cid)
    p.changeConstraint(c, maxForce=500, erp=0.95, physicsClientId=cid)

    p.setCollisionFilterPair(hand_id, obj.object_id, -1, -1,
                              enableCollision=0, physicsClientId=cid)
    for _ in range(cfg["episode"]["settle_steps"]):
        p.stepSimulation(physicsClientId=cid)
    p.setCollisionFilterPair(hand_id, obj.object_id, -1, -1,
                              enableCollision=1, physicsClientId=cid)

    return cid, hand_id, hand, obj.object_id, pregrasp_q


def _reset_scene(cid, hand, ar10, hand_id, pregrasp_q, settle_steps):
    hand.teleport_to(pregrasp_q)
    ar10.send_q_target(list(pregrasp_q))
    for _ in range(settle_steps):
        p.stepSimulation(physicsClientId=cid)


def _save(samples: list[float], sim_threshold: float) -> None:
    # Median als real_threshold, robuster gegen Ausreißer als der Mittelwert.
    if not samples:
        print("\n[threshold-cal] No trials recorded.")
        return
    median = statistics.median(samples)
    mean   = sum(samples) / len(samples)
    stdev  = statistics.stdev(samples) if len(samples) > 1 else 0.0

    print(f"\n[threshold-cal] {len(samples)} trials")
    print(f"  median = {median:.4f}")
    print(f"  mean   = {mean:.4f}")
    print(f"  stdev  = {stdev:.4f}")

    _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _OUT_PATH.open("w", encoding="utf-8") as f:
        yaml.dump({
            "real_threshold":    float(median),
            "sim_threshold":     float(sim_threshold),
            "calibration_joint": "servo9",
            "n_trials":          len(samples),
            "median":            float(median),
            "mean":              float(mean),
            "stdev":             float(stdev),
            "samples":           [float(s) for s in samples],
        }, f, default_flow_style=False)
    print(f"  saved → {_OUT_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate real-hand binary threshold.")
    parser.add_argument("--config",   required=True,
                        help="Grasp config (uses precision settings).")
    parser.add_argument("--port",     default=None,
                        help="AR10 COM port (e.g. COM4). Omit for mock mode.")
    parser.add_argument("--n-trials", type=int, default=10)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    sim_threshold = cfg["observation"]["threshold"]
    pip_cap       = cfg["action"]["pip_caps"].get("servo9", 1.0)

    js   = init_pygame_joystick()
    ar10 = AR10Interface(com_port=args.port)
    if args.port is None:
        print("[threshold-cal] Mock mode — real q_delta will always be 0.\n")

    print(f"[threshold-cal] sim_threshold = {sim_threshold}")
    cid, hand_id, hand, cube_id, pregrasp_q = _setup_sim(cfg)
    ar10.send_q_target(list(pregrasp_q))
    time.sleep(1.0)

    samples:   list[float] = []
    triggered              = False
    q_target               = list(pregrasp_q)
    prev_b = prev_x        = False
    substeps = cfg["episode"]["substeps"]

    print(f"\n  Target: {args.n_trials} trials")
    print("  A hold  : close index finger")
    print("  B press : reset trial")
    print("  X press : save + exit\n")

    try:
        while True:
            inp    = read_inputs(js)
            x_trig = inp["x"] and not prev_x
            b_trig = inp["b"] and not prev_b
            prev_x = inp["x"]
            prev_b = inp["b"]

            if inp["menu"]:
                print("\n[threshold-cal] Quit without save.")
                return
            if x_trig:
                break
            if b_trig:
                _reset_scene(cid, hand, ar10, hand_id, pregrasp_q,
                              cfg["episode"]["settle_steps"])
                q_target  = list(pregrasp_q)
                triggered = False
                print(f"\n[threshold-cal] Reset. ({len(samples)}/{args.n_trials} so far)")
                continue

            if inp["a"] and not triggered:
                q_target[_INDEX_MCP_IDX] = min(1.0,    q_target[_INDEX_MCP_IDX] + _CLOSE_RATE)
                q_target[_INDEX_PIP_IDX] = min(pip_cap, q_target[_INDEX_PIP_IDX] + _CLOSE_RATE)
                hand.apply_q_target(q_target)
                ar10.send_q_target(q_target)

                for _ in range(substeps):
                    p.stepSimulation(physicsClientId=cid)

                q_sim_m  = hand.q_measured()
                q_real_m = ar10.read_q_measured()
                sim_dq   = q_target[_INDEX_PIP_IDX] - q_sim_m[_INDEX_PIP_IDX]
                real_dq  = q_target[_INDEX_PIP_IDX] - q_real_m[_INDEX_PIP_IDX]

                print(f"\r  q_pip={q_target[_INDEX_PIP_IDX]:.3f}  "
                      f"sim_dq={sim_dq:.4f}  real_dq={real_dq:.4f}",
                      end="", flush=True)

                # Sim erkennt Kontakt -> echten q_delta in diesem Moment aufzeichnen.
                if sim_dq > sim_threshold:
                    samples.append(real_dq)
                    triggered = True
                    print(f"\n[threshold-cal] Trial {len(samples)}/{args.n_trials} "
                          f"→ real_q_delta = {real_dq:.4f}")
                    if len(samples) >= args.n_trials:
                        break
            else:
                for _ in range(substeps):
                    p.stepSimulation(physicsClientId=cid)

            time.sleep(_DT)

        _save(samples, sim_threshold)

    finally:
        ar10.send_q_target([0.0] * _HAND_DOF)
        time.sleep(1.0)
        ar10.close()
        import pygame
        pygame.quit()
        p.disconnect(cid)


if __name__ == "__main__":
    main()
