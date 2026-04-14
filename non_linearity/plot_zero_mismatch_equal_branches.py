#!/usr/bin/env python3
"""
Plot the zero-mismatch slice of the saved 4-transistor sweep.

This keeps only rows with:
  - g0 = g1
  - g2 = g3

The resulting 2D map uses:
  x = branch-A equal gate value = g0 = g1
  y = branch-B equal gate value = g2 = g3
  color = rel_lin_rmse
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors


RUN_DIR_DEFAULT = Path(
    "/home/ma-lab/Desktop/pyclln-final/non_linearity/results/runs/fourtrans_50pt_1to5V_live_20260412"
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot zero-mismatch equal-branch heatmap from saved sweep data")
    p.add_argument("--run-dir", type=Path, default=RUN_DIR_DEFAULT)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--tol", type=float, default=1e-9)
    return p.parse_args()


def _compute_edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 1:
        return np.array([values[0] - 0.5, values[0] + 0.5], dtype=float)
    mids = 0.5 * (values[:-1] + values[1:])
    left = values[0] - 0.5 * (values[1] - values[0])
    right = values[-1] + 0.5 * (values[-1] - values[-2])
    return np.concatenate([[left], mids, [right]])


def main() -> None:
    args = _parse_args()
    run_dir = args.run_dir.resolve()
    out_dir = (args.out_dir.resolve() if args.out_dir else run_dir / "analysis_mechanistic")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(run_dir / "sweep_results.csv", usecols=["g0", "g1", "g2", "g3", "rel_lin_rmse"])
    df = df.replace([np.inf, -np.inf], np.nan).dropna().copy()

    tol = float(args.tol)
    mask = (df["g0"].sub(df["g1"]).abs() <= tol) & (df["g2"].sub(df["g3"]).abs() <= tol)
    sub = df.loc[mask].copy()
    if sub.empty:
        raise RuntimeError("No zero-mismatch rows found in sweep_results.csv")

    sub["Aeq"] = sub["g0"]
    sub["Beq"] = sub["g2"]

    grid = sub.groupby(["Beq", "Aeq"], observed=False)["rel_lin_rmse"].mean().unstack("Aeq")
    xvals = grid.columns.to_numpy(dtype=float)
    yvals = grid.index.to_numpy(dtype=float)
    data = grid.to_numpy(dtype=float)

    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="white")
    valid = data[np.isfinite(data)]
    norm = None
    if valid.size:
        vmin = max(float(np.quantile(valid, 0.05)), 1e-6)
        vmax = float(np.quantile(valid, 0.995))
        vmax = max(vmax, vmin * 1.01)
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax, clip=True)

    plt.figure(figsize=(7.2, 5.8))
    ax = plt.gca()
    mesh = ax.pcolormesh(
        _compute_edges(xvals),
        _compute_edges(yvals),
        np.ma.masked_invalid(data),
        cmap=cmap,
        norm=norm,
        shading="flat",
        edgecolors="none",
        linewidth=0.0,
        antialiased=False,
    )
    cbar = plt.colorbar(mesh, ax=ax)
    cbar.set_label("Relative linear-fit RMSE (log scale, clipped)")

    ax.set_xlabel("g0 = g1")
    ax.set_ylabel("g2 = g3")
    ax.set_title("Zero-mismatch slice: equal branch magnitudes")
    ax.grid(True, alpha=0.18)
    ax.text(
        0.98,
        0.98,
        f"{len(sub)} zero-mismatch configurations",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )

    plt.tight_layout()
    plt.savefig(out_dir / "heatmap_zero_mismatch_equal_branches.png", dpi=220)
    plt.close()

    sub.sort_values(["Aeq", "Beq"]).to_csv(out_dir / "zero_mismatch_equal_branches.csv", index=False)


if __name__ == "__main__":
    main()
