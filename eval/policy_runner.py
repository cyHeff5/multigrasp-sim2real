"""Shared policy runner — used by sim and real eval scripts.

Sim:
  run_episode(env, policy, seed, options) — one Gymnasium episode.

Real (Policy Laptop):
  run_real_episode(ar10, policy, cfg, real_thresh, n_steps_max, step_dt)
    — one closed-loop grasp on the real AR10 hand.

Standalone entry point (Policy Laptop):
  python -m eval.policy_runner --config configs/precision.yaml \\
                               --model  artifacts/models/precision/best/best_model \\
                               --port   COM4
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import yaml

from hardware.ar10 import AR10Interface
from sim.hand      import CONTROL_JOINTS, SERVO0_INIT


_REPO_ROOT      = Path(__file__).resolve().parent.parent
_REAL_THRESHOLD = _REPO_ROOT / "artifacts" / "calibration" / "real_threshold.yaml"
_HAND_DOF       = len(CONTROL_JOINTS)


# ── Sim helpers ───────────────────────────────────────────────────────────────

def load_policy(model_path: str):
    """Load a stable-baselines3 PPO policy from disk."""
    from stable_baselines3 import PPO
    zip_path = Path(model_path)
    if not zip_path.suffix:
        zip_path = Path(model_path + ".zip")
    if not zip_path.exists():
        raise FileNotFoundError(f"Policy not found: {zip_path}")
    return PPO.load(str(zip_path.with_suffix("")))


def run_episode(env, policy, seed: int, options: dict | None = None) -> dict:
    """Run a single sim episode with deterministic policy. Returns the final info dict."""
    obs, _ = env.reset(seed=seed, options=options)
    while True:
        action, _ = policy.predict(obs, deterministic=True)
        obs, _reward, terminated, _truncated, info = env.step(action)
        if terminated:
            return dict(info)


# ── Real-hand helpers ─────────────────────────────────────────────────────────

def _load_real_threshold(fallback_sim: float) -> float:
    if not _REAL_THRESHOLD.exists():
        print(f"[policy-runner] WARN: {_REAL_THRESHOLD} not found, using sim threshold {fallback_sim}")
        return fallback_sim
    with _REAL_THRESHOLD.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    th = float(data.get("real_threshold", fallback_sim))
    print(f"[policy-runner] Using calibrated real_threshold = {th:.4f}")
    return th


def _binary_obs(ar10: AR10Interface, q_target: list[float],
                finger_joints: dict, fingers: list[str],
                threshold: float) -> np.ndarray:
    q_meas = ar10.read_q_measured()
    out    = np.zeros(len(fingers), dtype=np.float32)
    for i, finger in enumerate(fingers):
        for joint_name in finger_joints[finger]:
            j = CONTROL_JOINTS.index(joint_name)
            if (q_target[j] - q_meas[j]) > threshold:
                out[i] = 1.0
                break
    return out


def _apply_action(action: np.ndarray, q_target: list[float], cfg: dict) -> list[float]:
    """Mirror env._action_to_delta + PIP caps on the real hand."""
    delta = [0.0] * _HAND_DOF
    for i, joint_name in enumerate(cfg["active_joints"]):
        a = float(action[i])
        if joint_name == "servo0":
            d = a * cfg["action"]["thumb_abduction_delta"]
        else:
            d = max(0.0, a) * cfg["action"]["delta_norm"]
        delta[CONTROL_JOINTS.index(joint_name)] = d

    new_q = [max(0.0, min(1.0, q + d)) for q, d in zip(q_target, delta)]

    rng_  = cfg["action"]["thumb_abduction_range"]
    idx0  = CONTROL_JOINTS.index("servo0")
    new_q[idx0] = max(rng_[0], min(rng_[1], new_q[idx0]))
    for joint_name, cap in cfg["action"]["pip_caps"].items():
        idx = CONTROL_JOINTS.index(joint_name)
        new_q[idx] = min(cap, new_q[idx])

    return new_q


def run_real_episode(ar10: AR10Interface, policy, cfg: dict,
                     real_thresh: float, n_steps_max: int,
                     step_dt: float = 0.033) -> list[float]:
    """Open hand to pregrasp, run policy closed-loop, return final q_target."""
    fingers    = list(cfg["finger_joints"].keys())
    pregrasp_q = [SERVO0_INIT if i == 0 else 0.0 for i in range(_HAND_DOF)]

    ar10.send_q_target(list(pregrasp_q))
    time.sleep(1.0)
    q_target = list(pregrasp_q)

    for _ in range(n_steps_max):
        obs      = _binary_obs(ar10, q_target, cfg["finger_joints"], fingers, real_thresh)
        action, _ = policy.predict(obs, deterministic=True)
        q_target  = _apply_action(action, q_target, cfg)
        ar10.send_q_target(q_target)
        time.sleep(step_dt)

    return q_target


# ── Standalone entry point (Policy Laptop) ────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Policy Laptop: run AR10 grasp policy. "
                    "Coordinate with Sawyer Laptop via Enter prompts."
    )
    parser.add_argument("--config",  required=True)
    parser.add_argument("--model",   required=True,
                        help="Path to trained model (with or without .zip).")
    parser.add_argument("--port",    default=None,
                        help="AR10 COM port (e.g. COM4). Omit for mock mode.")
    parser.add_argument("--step-dt", type=float, default=0.033,
                        help="Seconds between policy steps (default 30 Hz).")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    sim_thresh  = cfg["observation"]["threshold"]
    real_thresh = _load_real_threshold(sim_thresh)
    n_steps_max = cfg["episode"]["max_steps"]

    policy = load_policy(args.model)
    ar10   = AR10Interface(com_port=args.port)
    if args.port is None:
        print("[policy-runner] Mock mode — no real hand connected.\n")

    print(f"[policy-runner] config      = {Path(args.config).stem}")
    print(f"[policy-runner] real_thresh = {real_thresh:.4f}")
    print(f"[policy-runner] max_steps   = {n_steps_max}")
    print("\nReady. Waiting for Sawyer Laptop operator.\n")

    try:
        ep = 0
        while True:
            ep += 1
            input(f"[Episode {ep}] Sawyer at pregrasp? Press Enter to start grasp ...")
            q_final = run_real_episode(ar10, policy, cfg, real_thresh, n_steps_max, args.step_dt)
            print(f"  Grasp done.  q_final = {[round(v, 3) for v in q_final]}")
            input("  Press Enter after lift is complete ...")
            ar10.send_q_target([0.0] * _HAND_DOF)
            time.sleep(0.5)
            print()

    except KeyboardInterrupt:
        print("\n[policy-runner] Stopped.")
    finally:
        ar10.send_q_target([0.0] * _HAND_DOF)
        time.sleep(1.0)
        ar10.close()


if __name__ == "__main__":
    main()
