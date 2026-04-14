#!/usr/bin/env python3
"""
Plot nonlinear-regression run batches with PNAS-style mode decomposition.

For each run directory, write a two-panel figure:
  - left: log-scale MSE with the first three projected mode-MSE contributions
  - right: fitted curve snapshots at four points through training

The mode basis is constructed by Gram-Schmidt orthonormalization on the
input-power vectors [(V1)^0, (V1)^1, (V1)^2, ...], matching the description
used for nonlinear regression in the 2024 PNAS paper. For direct comparison
to total MSE, each mode trace is plotted as a projected MSE contribution
((n_m · e)^2 / N), not as the raw projection magnitude |n_m · e|.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_RUN_ROOT = Path(__file__).resolve().parent / "results" / "pnas_4x4_gamma_sweep_30000ep"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot nonlinear-regression runs with mode traces")
    p.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    return p.parse_args()


def _load_dataset(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    meta = json.loads((run_dir / "run_meta.json").read_text())
    dataset = meta["dataset"]
    return np.asarray(dataset["vins"], dtype=float), np.asarray(dataset["vous"], dtype=float)


def _default_snapshots(preds_hist: np.ndarray) -> list[int]:
    max_epoch = preds_hist.shape[0] - 1
    return [0, max_epoch // 3, (2 * max_epoch) // 3, max_epoch]


def _gram_schmidt_modes(x: np.ndarray) -> np.ndarray:
    basis = []
    for power in range(x.size):
        vec = x ** power
        for prev in basis:
            vec = vec - np.dot(prev, vec) * prev
        norm = np.linalg.norm(vec)
        if norm <= 1e-12:
            continue
        basis.append(vec / norm)
    if len(basis) < 3:
        raise RuntimeError("Failed to construct the first three orthonormal modes")
    return np.asarray(basis[:3], dtype=float)


def _mode_mse_contributions(preds_hist: np.ndarray, targets: np.ndarray, mode_basis: np.ndarray) -> np.ndarray:
    err = np.asarray(preds_hist, dtype=float) - np.asarray(targets, dtype=float).reshape(1, -1)
    coeffs = err @ mode_basis.T
    return (coeffs ** 2) / err.shape[1]


def _plot_run(run_dir: Path) -> Path:
    mse_hist = np.load(run_dir / "mse_history.npy")
    preds_hist = np.load(run_dir / "preds_history.npy")
    x, y = _load_dataset(run_dir)
    snapshots = _default_snapshots(preds_hist)
    mode_basis = _gram_schmidt_modes(x)
    mode_mse = _mode_mse_contributions(preds_hist, y, mode_basis)
    mse_plot = np.clip(mse_hist, np.finfo(float).tiny, None)
    mode_plot = np.clip(mode_mse, np.finfo(float).tiny, None)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    ax = axes[0]
    ax.plot(np.arange(mse_hist.size), mse_plot, color="black", linewidth=1.4, label="MSE")
    ax.plot(np.arange(mode_mse.shape[0]), mode_plot[:, 0], color="#2ca02c", linewidth=1.3, label="Mode 0 MSE")
    ax.plot(np.arange(mode_mse.shape[0]), mode_plot[:, 1], color="#1f77b4", linewidth=1.3, label="Mode 1 MSE")
    ax.plot(np.arange(mode_mse.shape[0]), mode_plot[:, 2], color="#d62728", linewidth=1.3, label="Mode 2 MSE")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Projected MSE")
    ax.set_title(f"{run_dir.name}: MSE + First Three Mode Contributions")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.scatter(x, y, s=70, facecolors="none", edgecolors="black", label="Train data")
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(snapshots)))
    for color, epoch in zip(colors, snapshots):
        ax.plot(x, preds_hist[epoch], color=color, linewidth=1.8, label=f"Epoch {epoch}")
    ax.set_xlabel("Input Voltage (V)")
    ax.set_ylabel("Output Voltage (V)")
    ax.set_title(f"{run_dir.name}: Fitted Curve Snapshots")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    fig.tight_layout()
    out_path = run_dir / "two_panel_epoch_snapshots_with_modes.png"
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path


def _plot_overview(run_dirs: list[Path], out_path: Path) -> None:
    fig, axes = plt.subplots(len(run_dirs), 2, figsize=(13, 4.0 * len(run_dirs)))
    if len(run_dirs) == 1:
        axes = np.asarray([axes])

    for row, run_dir in enumerate(run_dirs):
        mse_hist = np.load(run_dir / "mse_history.npy")
        preds_hist = np.load(run_dir / "preds_history.npy")
        x, y = _load_dataset(run_dir)
        snapshots = _default_snapshots(preds_hist)
        mode_basis = _gram_schmidt_modes(x)
        mode_mse = _mode_mse_contributions(preds_hist, y, mode_basis)
        mse_plot = np.clip(mse_hist, np.finfo(float).tiny, None)
        mode_plot = np.clip(mode_mse, np.finfo(float).tiny, None)

        ax = axes[row, 0]
        ax.plot(np.arange(mse_hist.size), mse_plot, color="black", linewidth=1.2, label="MSE")
        ax.plot(np.arange(mode_mse.shape[0]), mode_plot[:, 0], color="#2ca02c", linewidth=1.1, label="Mode 0 MSE")
        ax.plot(np.arange(mode_mse.shape[0]), mode_plot[:, 1], color="#1f77b4", linewidth=1.1, label="Mode 1 MSE")
        ax.plot(np.arange(mode_mse.shape[0]), mode_plot[:, 2], color="#d62728", linewidth=1.1, label="Mode 2 MSE")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Projected MSE")
        ax.set_title(f"{run_dir.name}: MSE + Mode Contributions")
        ax.set_yscale("log")
        ax.grid(True, alpha=0.25)
        if row == 0:
            ax.legend(fontsize=8)

        ax = axes[row, 1]
        ax.scatter(x, y, s=55, facecolors="none", edgecolors="black", label="Train data")
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(snapshots)))
        for color, epoch in zip(colors, snapshots):
            ax.plot(x, preds_hist[epoch], color=color, linewidth=1.5, label=f"Epoch {epoch}")
        ax.set_xlabel("Input Voltage (V)")
        ax.set_ylabel("Output Voltage (V)")
        ax.set_title(f"{run_dir.name}: Curve Snapshots")
        ax.grid(True, alpha=0.25)
        if row == 0:
            ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    run_root = args.run_root.resolve()
    run_dirs = sorted(
        p for p in run_root.iterdir()
        if p.is_dir() and (p / "run_meta.json").exists()
    )
    if not run_dirs:
        raise FileNotFoundError(f"No run directories with run_meta.json found under {run_root}")

    for run_dir in run_dirs:
        print(_plot_run(run_dir))

    overview_path = run_root / "all_runs_two_panel_epoch_snapshots_with_modes.png"
    _plot_overview(run_dirs, overview_path)
    print(overview_path)


if __name__ == "__main__":
    main()
