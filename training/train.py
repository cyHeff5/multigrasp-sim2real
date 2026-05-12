#PPO training for AR10 grasp policies.

#Usage:
#    python -m training.train --config configs/power.yaml
#    python -m training.train --config configs/precision.yaml --gui
#    python -m training.train --config configs/power.yaml --resume artifacts/models/power/ppo_500000_steps

#Outputs:
#    artifacts/models/<grasp_type>/         # checkpoints + best model
#    artifacts/models/<grasp_type>/final    # final saved policy
#    artifacts/logs/<grasp_type>_<ts>/      # tensorboard logs
from __future__ import annotations

import argparse
import datetime
import os
from pathlib import Path

import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env  import SubprocVecEnv, VecMonitor

from sim import GraspEnv


_REPO_ROOT  = Path(__file__).resolve().parent.parent
_MODELS_DIR = _REPO_ROOT / "artifacts" / "models"
_LOGS_DIR   = _REPO_ROOT / "artifacts" / "logs"


def make_env(grasp_cfg: dict, seed: int, render: bool = False):
    # SubprocVecEnv erwartet Callables, die die Env erst im Subprocess erzeugen.
    def _init():
        env = GraspEnv(grasp_cfg, render_mode="human" if render else None)
        env.reset(seed=seed)
        return env
    return _init


def main() -> None:
    parser = argparse.ArgumentParser(description="PPO training for AR10 grasp.")
    parser.add_argument("--config",     required=True,
                        help="Path to grasp config (e.g. configs/power.yaml).")
    parser.add_argument("--ppo-config", default="configs/ppo.yaml")
    parser.add_argument("--gui",        action="store_true",
                        help="Render one env with GUI (forces n_envs=1).")
    parser.add_argument("--resume",     default=None,
                        help="Checkpoint path (without .zip) to resume from.")
    parser.add_argument("--seed",       type=int, default=0)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        grasp_cfg = yaml.safe_load(f)
    with open(args.ppo_config, encoding="utf-8") as f:
        ppo_cfg = yaml.safe_load(f)

    grasp_type = grasp_cfg["grasp_type"]
    n_envs     = 1 if args.gui else os.cpu_count()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name  = f"{grasp_type}_{timestamp}"
    ckpt_dir  = _MODELS_DIR / grasp_type
    log_dir   = _LOGS_DIR   / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"[train] grasp_type = {grasp_type}")
    print(f"[train] n_envs     = {n_envs}")
    print(f"[train] timesteps  = {ppo_cfg['total_timesteps']}")
    print(f"[train] checkpoints -> {ckpt_dir}")
    print(f"[train] logs       -> {log_dir}")

    # VecMonitor zeichnet Episode-Rewards und -Längen auf -> TensorBoard-Kurven.
    train_env = VecMonitor(SubprocVecEnv([
        make_env(grasp_cfg, args.seed + i, render=(args.gui and i == 0))
        for i in range(n_envs)
    ]))
    # Großer Seed-Abstand damit Eval-Episoden sich nicht mit Training-Seeds überschneiden.
    eval_env = VecMonitor(SubprocVecEnv([
        make_env(grasp_cfg, args.seed + 10_000)
    ]))

    # // n_envs weil VecEnv den Callback einmal pro vectorized step aufruft,
    # aber n_envs echte Schritte parallel gemacht werden.
    callbacks = [
        CheckpointCallback(
            save_freq=max(ppo_cfg["checkpoint_freq"] // n_envs, 1),
            save_path=str(ckpt_dir),
            name_prefix="ppo",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=str(ckpt_dir / "best"),
            log_path=str(ckpt_dir / "eval_logs"),
            eval_freq=max(ppo_cfg["eval_freq"] // n_envs, 1),
            n_eval_episodes=ppo_cfg["n_eval_episodes"],
            verbose=1,
        ),
    ]

    # Nur PPO-Hyperparameter extrahieren.
    ppo_kwargs = {k: ppo_cfg[k] for k in (
        "n_steps", "batch_size", "n_epochs", "learning_rate",
        "gamma", "gae_lambda", "clip_range",
        "ent_coef", "vf_coef", "max_grad_norm",
    )}

    if args.resume:
        print(f"[train] Resuming from {args.resume}")
        model = PPO.load(args.resume, env=train_env, tensorboard_log=str(log_dir))
        # Checkpoint-Dateiname: ppo_<steps>_steps.zip -> [-2] = Schrittzahl
        trained_so_far = int(Path(args.resume).stem.split("_")[-2])
        remaining = max(0, ppo_cfg["total_timesteps"] - trained_so_far)
        # reset_num_timesteps=False: TensorBoard-Achse läuft weiter statt von 0.
        model.learn(total_timesteps=remaining, callback=callbacks,
                     reset_num_timesteps=False, tb_log_name=run_name,
                     progress_bar=True)
    else:
        model = PPO("MlpPolicy", train_env, verbose=1,
                     tensorboard_log=str(log_dir), **ppo_kwargs)
        model.learn(total_timesteps=ppo_cfg["total_timesteps"],
                     callback=callbacks, tb_log_name=run_name, progress_bar=True)

    final_path = ckpt_dir / "final"
    model.save(str(final_path))
    print(f"\n[train] Done. Final model saved to {final_path}.zip")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
