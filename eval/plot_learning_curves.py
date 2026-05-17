# Erzeugt publikationsreife Learning-Curves aus den TensorBoard-Logs.
# Zeigt pro Greiftyp: Trainings-Reward (geglättet), Eval-Reward, Episodenlänge.
#
# Usage:
#   python -m eval.plot_learning_curves \
#       --precision artifacts/logs/precision_20260515_114935/precision_20260515_114935_1 \
#       --power     artifacts/logs/power_20260515_224309/power_20260515_224309_1 \
#       --out       artifacts/eval_results/learning_curves.png
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def _load(path: str, tag: str):
    ea = EventAccumulator(path, size_guidance={"scalars": 0})
    ea.Reload()
    ev = ea.Scalars(tag)
    return np.array([e.step for e in ev]), np.array([e.value for e in ev])


def _smooth(y: np.ndarray, k: int = 9) -> np.ndarray:
    # Kantenkorrektes gleitendes Mittel: y und eine Eins-Maske werden gleich
    # gefaltet, dann geteilt. Dadurch ist der geglättete Wert am Rand der echte
    # Mittelwert über die tatsächlich vorhandenen Punkte (kein Abfall gegen 0).
    if len(y) < k:
        return y
    kernel = np.ones(k)
    num = np.convolve(y, kernel, mode="same")
    den = np.convolve(np.ones_like(y), kernel, mode="same")
    return num / den


def _panel(ax, log_dir: str, title: str, color: str):
    s_tr, v_tr = _load(log_dir, "rollout/ep_rew_mean")
    sm = _smooth(v_tr)

    ax.plot(s_tr / 1e6, v_tr, color=color, alpha=0.20, lw=1,
            label="Training reward (raw)")
    ax.plot(s_tr / 1e6, sm, color=color, lw=2,
            label="Training reward (smoothed)")

    ax.axhline(0, color="0.7", lw=0.8)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("Environment steps (millions)")
    ax.set_ylabel("Episode reward")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--precision", required=True)
    ap.add_argument("--power", required=True)
    ap.add_argument("--out", default="artifacts/eval_results/learning_curves.png")
    args = ap.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    _panel(axes[0], args.precision, "Precision Grasp", "#1f77b4")
    _panel(axes[1], args.power, "Power Grasp", "#d62728")
    fig.tight_layout()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"saved -> {out}")
    print(f"saved -> {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
