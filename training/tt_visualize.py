"""
Visualization utilities for the Trajectory Transformer training run.

Usage:
    python tt_visualize.py                          # regenerate all plots
    python tt_visualize.py --metrics path/to/metrics.json --preds path/to/preds.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Individual plot functions
# ---------------------------------------------------------------------------

def plot_training_curves(metrics: dict, output_path: str) -> None:
    """CE loss (train/val) over epochs."""
    epochs      = [m["epoch"]            for m in metrics["epochs"]]
    train_loss  = [m["train"]["loss"]    for m in metrics["epochs"]]
    val_loss    = [m["val"]["loss"]      for m in metrics["epochs"]]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, train_loss, label="Train CE", color="#1f77b4")
    ax.plot(epochs, val_loss,   label="Val CE",   color="#ff7f0e")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Trajectory Transformer — Training Curves")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_token_accuracy(metrics: dict, output_path: str) -> None:
    """Token prediction accuracy (train/val) over epochs."""
    epochs     = [m["epoch"]                      for m in metrics["epochs"]]
    train_acc  = [m["train"]["token_acc"] * 100   for m in metrics["epochs"]]
    val_acc    = [m["val"]["token_acc"]   * 100   for m in metrics["epochs"]]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, train_acc, label="Train", color="#1f77b4")
    ax.plot(epochs, val_acc,   label="Val",   color="#ff7f0e")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Token Accuracy (%)")
    ax.set_title("Trajectory Transformer — Token Prediction Accuracy")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_ade_fde(preds: dict, output_path: str) -> None:
    """Per-sample ADE and FDE bar chart with mean lines."""
    ade = preds["ade_m"]
    fde = preds["fde_m"]
    ids = np.arange(len(ade))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, vals, label, color in zip(
        axes, [ade, fde], ["ADE (m)", "FDE (m)"], ["#2196F3", "#FF5722"]
    ):
        ax.bar(ids, vals, color=color, alpha=0.8)
        ax.axhline(np.nanmean(vals), color="black", linestyle="--", linewidth=1.2,
                   label=f"Mean: {np.nanmean(vals):.2f}m")
        ax.set_xlabel("Sample index"); ax.set_ylabel(label)
        ax.set_title(f"Per-sample {label}")
        ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.suptitle("Trajectory Transformer — Test Set Errors", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"Saved {output_path}")


def plot_trajectories(preds: dict, output_path: str, n_show: int = 6) -> None:
    """
    Predicted vs ground-truth trajectories in the ego-vehicle frame.
    All samples are shown (including stationary); axes are scaled per-plot
    so near-zero motion is still visible.
    """
    pred_xy = preds["pred_xy"]    # (N, T, 2)
    true_xy = preds["true_xy"]    # (N, T, 2)
    valid   = preds["valid_mask"] # (N, T)
    ade     = preds["ade_m"]
    fde     = preds["fde_m"]

    n_show = min(n_show, len(pred_xy))
    ncols = 3
    nrows = (n_show + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.5 * nrows))
    axes_flat = axes.flat if nrows > 1 else [axes] if ncols == 1 else axes.flat

    for ax, i in zip(axes_flat, range(n_show)):
        mask = valid[i].astype(bool)
        tx, ty = true_xy[i, mask, 0], true_xy[i, mask, 1]
        px, py = pred_xy[i, mask, 0], pred_xy[i, mask, 1]

        ax.plot(tx, ty, "o-", color="#2ecc71", lw=1.8, ms=3, label="Ground truth")
        ax.plot(px, py, "s--", color="#e74c3c", lw=1.8, ms=3, label="Predicted")
        ax.plot(tx[0], ty[0], "k*", ms=9, zorder=5)

        # Per-plot axis padding so stationary samples are not invisible points.
        xpad = max(np.ptp(np.concatenate([tx, px])) * 0.15, 0.5)
        ypad = max(np.ptp(np.concatenate([ty, py])) * 0.15, 0.5)
        ax.set_xlim(min(tx.min(), px.min()) - xpad, max(tx.max(), px.max()) + xpad)
        ax.set_ylim(min(ty.min(), py.min()) - ypad, max(ty.max(), py.max()) + ypad)

        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
        is_stationary = np.ptp(tx) < 0.5 and np.ptp(ty) < 0.5
        subtitle = "stationary" if is_stationary else f"ADE={ade[i]:.2f}m  FDE={fde[i]:.2f}m"
        ax.set_title(f"Sample {i}  {subtitle}", fontsize=9)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)

    for ax in list(axes_flat)[n_show:]:
        ax.set_visible(False)

    plt.suptitle("Trajectory Transformer — Predicted vs Ground Truth (Ego Frame)", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"Saved {output_path}")


# ---------------------------------------------------------------------------
# Generate all plots
# ---------------------------------------------------------------------------

def generate_all(metrics_path: str, preds_path: str, output_dir: str) -> None:
    import json
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with open(metrics_path) as f:
        metrics = json.load(f)
    preds = np.load(preds_path, allow_pickle=True)

    plot_training_curves(metrics, str(out / "tt_training_curves.png"))
    plot_token_accuracy(metrics,  str(out / "tt_token_accuracy.png"))
    plot_ade_fde(preds,           str(out / "tt_ade_fde.png"))
    plot_trajectories(preds,      str(out / "tt_trajectories.png"))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regenerate TT visualizations.")
    parser.add_argument("--metrics", default="outputs/tt_gcs_metrics.json")
    parser.add_argument("--preds",   default="outputs/tt_test_predictions.npz")
    parser.add_argument("--outdir",  default="outputs")
    args = parser.parse_args()
    generate_all(args.metrics, args.preds, args.outdir)
