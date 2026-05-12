"""Simulation evaluation of a trained grasp policy.

Two modes:
  shapes    — systematic grid over geometric primitives (sphere, cube, ...)
              N trials per grid point, mass/friction randomised per trial.
  benchmark — 14 benchmark URDFs with pregrasps from artifacts/grasp_lookup_table.yaml
              Only grasp points with matching grasp_type are evaluated.

Output: per-config success rate + YAML/CSV log.

Usage:
    python -m eval.eval_sim --config configs/power.yaml \\
                            --model artifacts/models/power/best/best_model \\
                            --mode shapes --trials 20 --step 1.0
    python -m eval.eval_sim --config configs/precision.yaml \\
                            --model artifacts/models/precision/best/best_model \\
                            --mode benchmark --parts 1 3 5 --trials 10
"""
from __future__ import annotations

import argparse
import csv
import datetime
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data
import yaml

from assets               import benchmark_part_urdf
from eval.object_grid     import build_grid, spec_label
from eval.policy_runner   import load_policy, run_episode
from sim                  import GraspEnv


_REPO_ROOT = Path(__file__).resolve().parent.parent
_LOOKUP    = _REPO_ROOT / "artifacts" / "grasp_lookup_table.yaml"
_PEDESTAL_H_DEFAULT = 0.04
_SPAWN_XY = [0.0, 0.0]


# ── Shapes mode ───────────────────────────────────────────────────────────────

def run_shapes_mode(env: GraspEnv, policy, cfg: dict,
                    n_trials: int, step_cm: float, base_seed: int) -> list[dict]:
    specs = build_grid(cfg["sampler"], step_cm=step_cm)
    print(f"[eval-sim] {len(specs)} grid points × {n_trials} trials "
          f"= {len(specs) * n_trials} episodes")

    rows: list[dict] = []
    for spec in specs:
        successes = 0
        for trial in range(n_trials):
            info = run_episode(env, policy, seed=base_seed + trial,
                                options={"obj_spec": spec})
            if info.get("lifted", False):
                successes += 1
        rate = successes / n_trials
        rows.append({**spec, "n_success": successes, "n_trials": n_trials,
                     "success_rate": round(rate, 4)})
        print(f"  {spec_label(spec):<48s}  {successes:>3}/{n_trials}  =  {rate*100:5.1f}%")
    return rows


def print_shapes_summary(rows: list[dict]) -> None:
    by_shape: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_shape[r["shape"]].append(r["success_rate"])
    print("\n-- Summary ----------------------------------------------")
    for shape in sorted(by_shape):
        rates = by_shape[shape]
        mean  = 100.0 * sum(rates) / len(rates)
        print(f"  {shape:<16s}  {mean:5.1f}%  ({len(rates)} configs)")


# ── Benchmark mode ────────────────────────────────────────────────────────────

def _load_lookup() -> list[dict]:
    with _LOOKUP.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    flat: list[dict] = []
    for part in data["parts"]:
        pid = int(part["part_id"])
        for gp in part.get("grasp_points", []):
            flat.append({"part_id": pid, "gp": gp})
    return flat


def _seated_obj_pose(urdf_path: Path, orientation_xyzw: list[float], ped_h: float):
    """Returns (obj_pos_world, spawn_z_offset_from_pedestal, size_m_aabb)."""
    cid = p.connect(p.DIRECT)
    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)
        p.setAdditionalSearchPath(str(_REPO_ROOT / "assets"), physicsClientId=cid)
        tmp = p.loadURDF(
            str(urdf_path),
            basePosition=[_SPAWN_XY[0], _SPAWN_XY[1], ped_h + 0.5],
            baseOrientation=[float(v) for v in orientation_xyzw],
            useFixedBase=True, physicsClientId=cid,
        )
        aabb_min, aabb_max = p.getAABB(tmp, physicsClientId=cid)
        cur_pos, _ = p.getBasePositionAndOrientation(tmp, physicsClientId=cid)
        obj_z  = float(cur_pos[2]) + (ped_h - float(aabb_min[2]))
        size_m = max(aabb_max[0] - aabb_min[0], aabb_max[1] - aabb_min[1])
    finally:
        p.disconnect(cid)
    return [_SPAWN_XY[0], _SPAWN_XY[1], obj_z], obj_z - ped_h, size_m


def _build_benchmark_episode(entry: dict, ped_h: float):
    pid       = entry["part_id"]
    gp        = entry["gp"]
    urdf_path = benchmark_part_urdf(pid)
    if not urdf_path.exists():
        return None

    obj_orn = [float(v) for v in gp["object_orientation_xyzw"]]
    obj_pos, z_off, size_m = _seated_obj_pose(urdf_path, obj_orn, ped_h)

    hand_pos, hand_quat = p.multiplyTransforms(
        obj_pos, obj_orn,
        [float(v) for v in gp["pregrasp_position_obj_xyz"]],
        [float(v) for v in gp["pregrasp_orientation_obj_xyzw"]],
    )

    spec = {
        "shape":                 "urdf",
        "urdf_path":             str(urdf_path),
        "base_orientation_xyzw": obj_orn,
        "spawn_z_offset_m":      z_off,
        "mass_kg":               0.1,
        "lateral_friction":      0.5,
        "yaw_rad":               0.0,
        "size_cm":               size_m * 100.0,
    }
    return spec, (list(hand_pos), list(hand_quat))


def run_benchmark_mode(env: GraspEnv, policy, grasp_type: str,
                       parts: list[int], n_trials: int, base_seed: int,
                       ped_h: float) -> list[dict]:
    flat   = _load_lookup()
    by_pid = {pid: [e for e in flat if e["part_id"] == pid] for pid in parts}

    rows: list[dict] = []
    for pid in parts:
        entries = [e for e in by_pid.get(pid, [])
                   if e["gp"].get("grasp_type") == grasp_type]
        if not entries:
            print(f"\n[eval-sim] Part {pid}: no {grasp_type} grasp points")
            continue
        for entry in entries:
            gp_id = entry["gp"]["id"]
            built = _build_benchmark_episode(entry, ped_h)
            if built is None:
                print(f"\n[eval-sim] Part {pid}/{gp_id}: URDF missing — SKIPPED")
                continue
            spec, pregrasp = built

            successes = 0
            for trial in range(n_trials):
                info = run_episode(env, policy, seed=base_seed + trial,
                                    options={"obj_spec": spec, "pregrasp": pregrasp})
                if info.get("lifted", False):
                    successes += 1
            rate = successes / n_trials
            rows.append({"part_id": pid, "gp_id": gp_id,
                         "n_success": successes, "n_trials": n_trials,
                         "success_rate": round(rate, 4)})
            print(f"  P{pid:02d}/{gp_id:<10}  {successes:>3}/{n_trials}  =  {rate*100:5.1f}%")
    return rows


# ── Output ────────────────────────────────────────────────────────────────────

def save_results(rows: list[dict], mode: str, out_dir: Path, config_stem: str) -> None:
    if not rows:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{config_stem}_{mode}_{ts}"

    with (out_dir / f"{stem}.yaml").open("w", encoding="utf-8") as f:
        yaml.dump({"mode": mode, "results": rows}, f, default_flow_style=False)
    with (out_dir / f"{stem}.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(dict.fromkeys(k for row in rows for k in row))
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved: {out_dir / stem}.{{yaml,csv}}")


# ── GUI helpers ───────────────────────────────────────────────────────────────

def _wait_for_space(cid: int) -> None:
    print("[eval-sim] PyBullet-Fenster fokussieren und SPACE druecken...")
    while True:
        keys = p.getKeyboardEvents(physicsClientId=cid)
        if 32 in keys and keys[32] & p.KEY_WAS_TRIGGERED:
            break
        time.sleep(0.05)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Sim evaluation of a trained policy.")
    parser.add_argument("--config", required=True,
                        help="Grasp config (configs/power.yaml or precision.yaml).")
    parser.add_argument("--model",  required=True,
                        help="Path to trained model (with or without .zip).")
    parser.add_argument("--mode",   choices=["shapes", "benchmark"], required=True)
    parser.add_argument("--trials", type=int,   default=20)
    parser.add_argument("--step",   type=float, default=1.0,
                        help="(shapes) grid step in cm")
    parser.add_argument("--parts",  nargs="+", type=int, default=list(range(1, 15)),
                        help="(benchmark) part-IDs 1-14")
    parser.add_argument("--seed",   type=int,   default=1000)
    parser.add_argument("--gui",    action="store_true")
    parser.add_argument("--output", default=None,
                        help="Output directory (default: artifacts/eval_results/)")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    config_stem = Path(args.config).stem
    out_dir     = Path(args.output) if args.output else _REPO_ROOT / "artifacts" / "eval_results"

    print(f"[eval-sim] grasp_type = {cfg['grasp_type']}")
    print(f"[eval-sim] mode       = {args.mode}")
    print(f"[eval-sim] model      = {args.model}")

    policy = load_policy(args.model)
    env    = GraspEnv(cfg, render_mode="human" if args.gui else None)

    if args.gui:
        env.reset()
        _wait_for_space(env._cid)

    try:
        if args.mode == "shapes":
            rows = run_shapes_mode(env, policy, cfg,
                                     n_trials=args.trials,
                                     step_cm=args.step,
                                     base_seed=args.seed)
            print_shapes_summary(rows)
        else:
            ped_h = cfg["episode"].get("pedestal_height", _PEDESTAL_H_DEFAULT)
            rows = run_benchmark_mode(env, policy, cfg["grasp_type"],
                                        args.parts, args.trials, args.seed, ped_h)
    finally:
        env.close()

    save_results(rows, args.mode, out_dir, config_stem)


if __name__ == "__main__":
    main()
