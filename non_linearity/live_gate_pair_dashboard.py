#!/usr/bin/env python3
"""
Watch a sweep CSV and refresh a 6-panel gate-pair nonlinearity dashboard.

This is intended for the 4-transistor sweep with gates g0..g3. Each panel is a
fixed gate-vs-gate grid:
  - white cell: no solved configuration has hit that gate pair yet
  - colored cell: mean rel_lin_rmse across solved configurations for that pair
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


PAIR_ORDER = [
    ("g0", "g1"),
    ("g1", "g2"),
    ("g2", "g3"),
    ("g3", "g0"),
    ("g1", "g3"),
    ("g0", "g2"),
]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live 6-panel gate-pair dashboard")
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--png", type=Path, required=True)
    p.add_argument("--refresh-sec", type=float, default=30.0)
    p.add_argument("--title", type=str, default="4-transistor gate-pair nonlinearity dashboard")
    return p.parse_args()


def _load_gate_grid(csv_path: Path) -> np.ndarray | None:
    meta_path = csv_path.parent / "run_meta.json"
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text())
        gate_grid = np.asarray(meta.get("gate_grid", []), dtype=float)
    except Exception:
        return None
    if gate_grid.size == 0:
        return None
    return gate_grid


def _load_rows(csv_path: Path) -> list[dict[str, float]]:
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return []

    rows: list[dict[str, float]] = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                rows.append(
                    {
                        "g0": float(row["g0"]),
                        "g1": float(row["g1"]),
                        "g2": float(row["g2"]),
                        "g3": float(row["g3"]),
                        "rel_lin_rmse": float(row["rel_lin_rmse"]),
                    }
                )
            except Exception:
                continue
    return rows


def _compute_edges(values: np.ndarray) -> np.ndarray:
    if values.size == 1:
        delta = 0.5
        return np.asarray([values[0] - delta, values[0] + delta], dtype=float)
    mids = 0.5 * (values[:-1] + values[1:])
    left = values[0] - 0.5 * (values[1] - values[0])
    right = values[-1] + 0.5 * (values[-1] - values[-2])
    return np.concatenate([[left], mids, [right]])


def _pair_grid(rows: list[dict[str, float]], gate_grid: np.ndarray, gx: str, gy: str) -> np.ndarray:
    x_index = {float(v): i for i, v in enumerate(gate_grid)}
    y_index = {float(v): i for i, v in enumerate(gate_grid)}
    grid_sum = np.zeros((gate_grid.size, gate_grid.size), dtype=float)
    grid_count = np.zeros((gate_grid.size, gate_grid.size), dtype=float)

    for row in rows:
        x = float(row[gx])
        y = float(row[gy])
        ix = x_index.get(x)
        iy = y_index.get(y)
        if ix is None or iy is None:
            continue
        grid_sum[iy, ix] += float(row["rel_lin_rmse"])
        grid_count[iy, ix] += 1.0

    grid = np.full((gate_grid.size, gate_grid.size), np.nan, dtype=float)
    mask = grid_count > 0
    grid[mask] = grid_sum[mask] / grid_count[mask]
    return grid


def _write_placeholder(png_path: Path, title: str, detail: str) -> None:
    fig, ax = plt.subplots(figsize=(11.5, 7.8))
    ax.axis("off")
    ax.text(0.5, 0.58, title, ha="center", va="center", fontsize=16, weight="bold")
    ax.text(0.5, 0.44, detail, ha="center", va="center", fontsize=11)
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=220)
    plt.close(fig)


def _write_dashboard(png_path: Path, title: str, csv_path: Path, rows: list[dict[str, float]]) -> None:
    gate_grid = _load_gate_grid(csv_path)
    if gate_grid is None:
        _write_placeholder(png_path, title, "run_meta.json with gate_grid not found")
        return
    if not rows:
        _write_placeholder(png_path, title, "Waiting for sweep_results.csv data")
        return

    grids = [_pair_grid(rows, gate_grid, gx, gy) for gx, gy in PAIR_ORDER]
    valid = np.concatenate([g[np.isfinite(g)] for g in grids if np.any(np.isfinite(g))])
    if valid.size:
        vmin = float(np.quantile(valid, 0.05))
        vmax = float(np.quantile(valid, 0.995))
        vmin = max(vmin, 1e-6)
        vmax = max(vmax, vmin * 1.01)
        norm = mcolors.LogNorm(vmin=vmin, vmax=vmax, clip=True)
    else:
        norm = None

    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="white")
    x_edges = _compute_edges(gate_grid)
    y_edges = _compute_edges(gate_grid)

    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.6), constrained_layout=True)
    mesh = None

    for ax, (gx, gy), grid in zip(axes.flat, PAIR_ORDER, grids):
        mesh = ax.imshow(
            np.ma.masked_invalid(grid),
            cmap=cmap,
            norm=norm,
            origin="lower",
            interpolation="nearest",
            extent=[float(x_edges[0]), float(x_edges[-1]), float(y_edges[0]), float(y_edges[-1])],
            aspect="auto",
        )
        ax.set_xlabel(gx)
        ax.set_ylabel(gy)
        ax.set_title(f"{gx} vs {gy}")
        ax.grid(True, alpha=0.12)

    if mesh is not None:
        cbar = fig.colorbar(mesh, ax=axes.ravel().tolist(), shrink=0.92, pad=0.02)
        cbar.set_label("Relative linear-fit RMSE (log scale, clipped)")

    fig.suptitle(
        f"{title}\n{len(rows)} solved configurations | white = no solved point yet",
        fontsize=14,
    )
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    csv_path = args.csv.resolve()
    png_path = args.png.resolve()

    _write_placeholder(png_path, args.title, f"Watching\n{csv_path}")

    while True:
        rows = _load_rows(csv_path)
        _write_dashboard(png_path, args.title, csv_path, rows)
        time.sleep(float(args.refresh_sec))


if __name__ == "__main__":
    main()
