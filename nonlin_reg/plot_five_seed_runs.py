#!/usr/bin/env python3
"""
Plot nonlinear-regression run batches with requested epoch snapshots.

For each run directory, write a two-panel figure:
  - left: MSE vs epoch
  - right: fitted curve snapshots

Also writes a combined overview figure for quick comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_RUN_ROOT = Path(__file__).resolve().parent / "results" / "pnas_4x4_gamma_sweep_30000ep"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot nonlinear-regression runs")
    p.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    p.add_argument(
        "--snapshots",
        type=int,
        nargs="+",
        default=None,
        help="Optional explicit epoch numbers to overlay on the fitted-curve panel",
    )
    return p.parse_args()


def _load_dataset(seed_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    meta = json.loads((seed_dir / "run_meta.json").read_text())
    dataset = meta["dataset"]
    x = np.asarray(dataset["vins"], dtype=float)
    y = np.asarray(dataset["vous"], dtype=float)
    return x, y


def _validate_snapshots(preds_hist: np.ndarray, snapshots: list[int]) -> list[int]:
    max_epoch = preds_hist.shape[0] - 1
    out = []
    for epoch in snapshots:
        if epoch < 0 or epoch > max_epoch:
            raise ValueError(f"snapshot epoch {epoch} is out of range 0..{max_epoch}")
        out.append(int(epoch))
    return out


def _default_snapshots(preds_hist: np.ndarray) -> list[int]:
    max_epoch = preds_hist.shape[0] - 1
    snapshots = [0, max_epoch // 3, (2 * max_epoch) // 3, max_epoch]
    # Guard against duplicates for short runs.
    return list(dict.fromkeys(int(epoch) for epoch in snapshots))


def _plot_seed(seed_dir: Path, snapshots: list[int]) -> Path:
    mse_hist = np.load(seed_dir / "mse_history.npy")
    preds_hist = np.load(seed_dir / "preds_history.npy")
    x, y = _load_dataset(seed_dir)
    snapshots = _validate_snapshots(preds_hist, snapshots or _default_snapshots(preds_hist))
    mse_plot = np.clip(mse_hist, np.finfo(float).tiny, None)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.plot(np.arange(mse_hist.size), mse_plot, color="black", linewidth=1.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title(f"{seed_dir.name}: MSE vs Epoch")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)

    ax = axes[1]
    ax.scatter(x, y, s=70, facecolors="none", edgecolors="black", label="Train data")
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(snapshots)))
    for color, epoch in zip(colors, snapshots):
        ax.plot(x, preds_hist[epoch], color=color, linewidth=1.8, label=f"Epoch {epoch}")
    ax.set_xlabel("Input Voltage (V)")
    ax.set_ylabel("Output Voltage (V)")
    ax.set_title(f"{seed_dir.name}: Fitted Curve Snapshots")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    fig.tight_layout()
    out_path = seed_dir / "two_panel_epoch_snapshots.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def _plot_overview(seed_dirs: list[Path], snapshots: list[int], out_path: Path) -> None:
    fig, axes = plt.subplots(len(seed_dirs), 2, figsize=(12, 4.0 * len(seed_dirs)))
    if len(seed_dirs) == 1:
        axes = np.asarray([axes])

    for row, seed_dir in enumerate(seed_dirs):
        mse_hist = np.load(seed_dir / "mse_history.npy")
        preds_hist = np.load(seed_dir / "preds_history.npy")
        x, y = _load_dataset(seed_dir)
        snapshots = _validate_snapshots(preds_hist, snapshots or _default_snapshots(preds_hist))
        mse_plot = np.clip(mse_hist, np.finfo(float).tiny, None)

        ax = axes[row, 0]
        ax.plot(np.arange(mse_hist.size), mse_plot, color="black", linewidth=1.1)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE")
        ax.set_title(f"{seed_dir.name}: MSE")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.25)

        ax = axes[row, 1]
        ax.scatter(x, y, s=55, facecolors="none", edgecolors="black", label="Train data")
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(snapshots)))
        for color, epoch in zip(colors, snapshots):
            ax.plot(x, preds_hist[epoch], color=color, linewidth=1.5, label=f"Epoch {epoch}")
        ax.set_xlabel("Input Voltage (V)")
        ax.set_ylabel("Output Voltage (V)")
        ax.set_title(f"{seed_dir.name}: Curve Snapshots")
        ax.grid(True, alpha=0.25)
        if row == 0:
            ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    run_root = args.run_root.resolve()
    snapshots = None if args.snapshots is None else [int(x) for x in args.snapshots]

    seed_dirs = sorted(
        p for p in run_root.iterdir()
        if p.is_dir() and (p / "run_meta.json").exists()
    )
    if not seed_dirs:
        raise FileNotFoundError(f"No run directories with run_meta.json found under {run_root}")

    for seed_dir in seed_dirs:
        out_path = _plot_seed(seed_dir, snapshots)
        print(out_path)

    overview_path = run_root / "all_runs_two_panel_epoch_snapshots.png"
    _plot_overview(seed_dirs, snapshots, overview_path)
    print(overview_path)


if __name__ == "__main__":
    main()
