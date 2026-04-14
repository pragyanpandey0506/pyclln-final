#!/usr/bin/env python3
"""
Watch a sweep CSV and refresh a 2D nonlinearity PNG.

The plot can use either:
  - several linear 4D -> 2D PCA reductions of the solved gate vectors [g0, g1, g2, g3]
  - fixed linear projections of the four gates
  - a branch-effective reduction using harmonic means:
        x = H(g0, g1)
        y = H(g2, g3)

Each solved configuration is plotted as exactly one point. Nothing is rendered
for unsolved configurations.
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


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live nonlinearity PNG updater")
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--png", type=Path, required=True)
    p.add_argument("--refresh-sec", type=float, default=30.0)
    p.add_argument("--title", type=str, default="4-transistor nonlinearity map")
    p.add_argument(
        "--mode",
        type=str,
        choices=[
            "pca",
            "pca-zscore",
            "pca-uncentered",
            "branch-hmean",
            "branch-mean",
            "total-balance",
            "branch-mismatch",
        ],
        default="pca",
    )
    return p.parse_args()


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


def _load_gate_grid_from_run_meta(csv_path: Path) -> np.ndarray | None:
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


def _project_linear_4d_to_2d(rows: list[dict[str, float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gates = np.asarray(
        [[row["g0"], row["g1"], row["g2"], row["g3"]] for row in rows],
        dtype=float,
    )
    colors = np.asarray([row["rel_lin_rmse"] for row in rows], dtype=float)

    if gates.shape[0] == 1:
        coords = np.zeros((1, 2), dtype=float)
        basis = np.eye(4, 2, dtype=float)
        return coords[:, 0], coords[:, 1], colors, basis

    centered = gates - gates.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)

    if vh.shape[0] >= 2:
        basis = vh[:2].T
    else:
        basis = np.eye(4, 2, dtype=float)

    coords = centered @ basis
    return coords[:, 0], coords[:, 1], colors, basis


def _project_pca_variant(
    rows: list[dict[str, float]],
    mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, str, str]:
    gates, colors = _extract_gates_and_colors(rows)

    if gates.shape[0] == 1:
        coords = np.zeros((1, 2), dtype=float)
        basis = np.eye(4, 2, dtype=float)
        if mode == "pca-zscore":
            basis_text = (
                "Standardized PCA\n"
                "PC1 = +1.00z0 +0.00z1 +0.00z2 +0.00z3\n"
                "PC2 = +0.00z0 +1.00z1 +0.00z2 +0.00z3\n"
                "White = no solved point yet"
            )
        elif mode == "pca-uncentered":
            basis_text = (
                "Uncentered PCA\n"
                "PC1 = +1.00g0 +0.00g1 +0.00g2 +0.00g3\n"
                "PC2 = +0.00g0 +1.00g1 +0.00g2 +0.00g3\n"
                "White = no solved point yet"
            )
        else:
            basis_text = (
                "PC1 = +1.00g0 +0.00g1 +0.00g2 +0.00g3\n"
                "PC2 = +0.00g0 +1.00g1 +0.00g2 +0.00g3\n"
                "White = no solved point yet"
            )
        return coords[:, 0], coords[:, 1], colors, basis, "PC1", "PC2", basis_text

    if mode == "pca-zscore":
        mean = gates.mean(axis=0, keepdims=True)
        std = gates.std(axis=0, keepdims=True)
        std[std < 1e-12] = 1.0
        data = (gates - mean) / std
        _, _, vh = np.linalg.svd(data, full_matrices=False)
        basis = vh[:2].T if vh.shape[0] >= 2 else np.eye(4, 2, dtype=float)
        coords = data @ basis
        basis_text = (
            "Standardized PCA\n"
            "PC1 = "
            f"{basis[0, 0]:+.2f}z0 {basis[1, 0]:+.2f}z1 {basis[2, 0]:+.2f}z2 {basis[3, 0]:+.2f}z3\n"
            "PC2 = "
            f"{basis[0, 1]:+.2f}z0 {basis[1, 1]:+.2f}z1 {basis[2, 1]:+.2f}z2 {basis[3, 1]:+.2f}z3\n"
            "White = no solved point yet"
        )
        return coords[:, 0], coords[:, 1], colors, basis, "PC1 (z-score)", "PC2 (z-score)", basis_text

    if mode == "pca-uncentered":
        _, _, vh = np.linalg.svd(gates, full_matrices=False)
        basis = vh[:2].T if vh.shape[0] >= 2 else np.eye(4, 2, dtype=float)
        coords = gates @ basis
        basis_text = (
            "Uncentered PCA\n"
            "PC1 = "
            f"{basis[0, 0]:+.2f}g0 {basis[1, 0]:+.2f}g1 {basis[2, 0]:+.2f}g2 {basis[3, 0]:+.2f}g3\n"
            "PC2 = "
            f"{basis[0, 1]:+.2f}g0 {basis[1, 1]:+.2f}g1 {basis[2, 1]:+.2f}g2 {basis[3, 1]:+.2f}g3\n"
            "White = no solved point yet"
        )
        return coords[:, 0], coords[:, 1], colors, basis, "PC1 (uncentered)", "PC2 (uncentered)", basis_text

    xs, ys, _, basis = _project_linear_4d_to_2d(rows)
    basis_text = (
        "Centered PCA\n"
        "PC1 = "
        f"{basis[0, 0]:+.2f}g0 {basis[1, 0]:+.2f}g1 {basis[2, 0]:+.2f}g2 {basis[3, 0]:+.2f}g3\n"
        "PC2 = "
        f"{basis[0, 1]:+.2f}g0 {basis[1, 1]:+.2f}g1 {basis[2, 1]:+.2f}g2 {basis[3, 1]:+.2f}g3\n"
        "White = no solved point yet"
    )
    return xs, ys, colors, basis, "PC1", "PC2", basis_text


def _extract_gates_and_colors(rows: list[dict[str, float]]) -> tuple[np.ndarray, np.ndarray]:
    gates = np.asarray(
        [[row["g0"], row["g1"], row["g2"], row["g3"]] for row in rows],
        dtype=float,
    )
    colors = np.asarray([row["rel_lin_rmse"] for row in rows], dtype=float)
    return gates, colors


def _fixed_projection_spec(mode: str) -> tuple[np.ndarray, str, str, str]:
    if mode == "branch-mean":
        matrix = np.asarray(
            [
                [0.5, 0.0],
                [0.5, 0.0],
                [0.0, 0.5],
                [0.0, 0.5],
            ],
            dtype=float,
        )
        return (
            matrix,
            "Branch A mean gate",
            "Branch B mean gate",
            "x = 0.5g0 + 0.5g1\n"
            "y = 0.5g2 + 0.5g3\n"
            "White = no solved point yet",
        )
    if mode == "total-balance":
        matrix = np.asarray(
            [
                [0.25, 0.25],
                [0.25, 0.25],
                [0.25, -0.25],
                [0.25, -0.25],
            ],
            dtype=float,
        )
        return (
            matrix,
            "Total gate drive",
            "Branch A minus branch B",
            "x = 0.25(g0 + g1 + g2 + g3)\n"
            "y = 0.25(g0 + g1 - g2 - g3)\n"
            "White = no solved point yet",
        )
    if mode == "branch-mismatch":
        matrix = np.asarray(
            [
                [0.5, 0.0],
                [-0.5, 0.0],
                [0.0, 0.5],
                [0.0, -0.5],
            ],
            dtype=float,
        )
        return (
            matrix,
            "Branch A mismatch",
            "Branch B mismatch",
            "x = 0.5(g0 - g1)\n"
            "y = 0.5(g2 - g3)\n"
            "White = no solved point yet",
        )
    raise ValueError(f"unsupported fixed projection mode: {mode}")


def _project_fixed_linear(rows: list[dict[str, float]], mode: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gates, colors = _extract_gates_and_colors(rows)
    matrix, _, _, _ = _fixed_projection_spec(mode)
    coords = gates @ matrix
    return coords[:, 0], coords[:, 1], colors, matrix


def _harmonic_mean(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denom = a + b
    out = np.zeros_like(denom, dtype=float)
    mask = np.abs(denom) > 1e-12
    out[mask] = 2.0 * a[mask] * b[mask] / denom[mask]
    return out


def _project_branch_hmean(rows: list[dict[str, float]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    g0 = np.asarray([row["g0"] for row in rows], dtype=float)
    g1 = np.asarray([row["g1"] for row in rows], dtype=float)
    g2 = np.asarray([row["g2"] for row in rows], dtype=float)
    g3 = np.asarray([row["g3"] for row in rows], dtype=float)
    colors = np.asarray([row["rel_lin_rmse"] for row in rows], dtype=float)
    xs = _harmonic_mean(g0, g1)
    ys = _harmonic_mean(g2, g3)
    return xs, ys, colors


def _compute_branch_hmean_axis(gate_grid: np.ndarray) -> np.ndarray:
    vals = []
    for a in gate_grid:
        for b in gate_grid:
            vals.append(float(_harmonic_mean(np.asarray([a]), np.asarray([b]))[0]))
    return np.asarray(sorted(set(vals)), dtype=float)


def _compute_bin_edges(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size == 1:
        delta = 0.5
        return np.asarray([values[0] - delta, values[0] + delta], dtype=float)

    mids = 0.5 * (values[:-1] + values[1:])
    left = values[0] - 0.5 * (values[1] - values[0])
    right = values[-1] + 0.5 * (values[-1] - values[-2])
    return np.concatenate([[left], mids, [right]])


def _build_branch_hmean_grid(
    rows: list[dict[str, float]],
    gate_grid: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if gate_grid is not None and gate_grid.size > 0:
        axis_x = _compute_branch_hmean_axis(gate_grid)
        axis_y = axis_x.copy()
    else:
        xs, ys, _ = _project_branch_hmean(rows)
        axis_x = np.asarray(sorted(set(xs.tolist())), dtype=float)
        axis_y = np.asarray(sorted(set(ys.tolist())), dtype=float)

    x_index = {float(v): i for i, v in enumerate(axis_x)}
    y_index = {float(v): i for i, v in enumerate(axis_y)}
    grid_sum = np.zeros((axis_y.size, axis_x.size), dtype=float)
    grid_count = np.zeros((axis_y.size, axis_x.size), dtype=float)

    xs, ys, cs = _project_branch_hmean(rows)
    for x, y, c in zip(xs, ys, cs):
        ix = x_index.get(float(x))
        iy = y_index.get(float(y))
        if ix is None or iy is None:
            continue
        grid_sum[iy, ix] += float(c)
        grid_count[iy, ix] += 1.0

    grid = np.full((axis_y.size, axis_x.size), np.nan, dtype=float)
    mask = grid_count > 0
    grid[mask] = grid_sum[mask] / grid_count[mask]
    return axis_x, axis_y, grid


def _build_pca_grid(
    rows: list[dict[str, float]],
    mode: str = "pca",
    bins_per_axis: int = 110,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, cs, basis, _, _, _ = _project_pca_variant(rows, mode)
    x_edges, y_edges, grid, basis = _build_projected_grid(xs, ys, cs, basis, bins_per_axis=bins_per_axis)
    grid = _smooth_fill_grid(grid, rounds=2, min_neighbors=4)
    return x_edges, y_edges, grid, basis


def _build_fixed_linear_grid(
    rows: list[dict[str, float]],
    mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, cs, matrix = _project_fixed_linear(rows, mode)
    bins_per_axis = 35 if mode == "branch-mismatch" else 70
    x_edges, y_edges, grid, matrix = _build_projected_grid(xs, ys, cs, matrix, bins_per_axis=bins_per_axis)
    grid = _smooth_fill_grid(grid, rounds=8, min_neighbors=1)
    return x_edges, y_edges, grid, matrix


def _build_projected_grid(
    xs: np.ndarray,
    ys: np.ndarray,
    cs: np.ndarray,
    basis: np.ndarray,
    bins_per_axis: int = 180,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

    x_min = float(np.min(xs))
    x_max = float(np.max(xs))
    y_min = float(np.min(ys))
    y_max = float(np.max(ys))

    if np.isclose(x_min, x_max):
        x_min -= 0.5
        x_max += 0.5
    if np.isclose(y_min, y_max):
        y_min -= 0.5
        y_max += 0.5

    x_pad = 0.02 * (x_max - x_min)
    y_pad = 0.02 * (y_max - y_min)
    x_edges = np.linspace(x_min - x_pad, x_max + x_pad, bins_per_axis + 1, dtype=float)
    y_edges = np.linspace(y_min - y_pad, y_max + y_pad, bins_per_axis + 1, dtype=float)

    x_idx = np.searchsorted(x_edges, xs, side="right") - 1
    y_idx = np.searchsorted(y_edges, ys, side="right") - 1
    x_idx = np.clip(x_idx, 0, bins_per_axis - 1)
    y_idx = np.clip(y_idx, 0, bins_per_axis - 1)

    grid_sum = np.zeros((bins_per_axis, bins_per_axis), dtype=float)
    grid_count = np.zeros((bins_per_axis, bins_per_axis), dtype=float)

    for ix, iy, c in zip(x_idx, y_idx, cs):
        grid_sum[iy, ix] += float(c)
        grid_count[iy, ix] += 1.0

    grid = np.full((bins_per_axis, bins_per_axis), np.nan, dtype=float)
    mask = grid_count > 0
    grid[mask] = grid_sum[mask] / grid_count[mask]
    return x_edges, y_edges, grid, basis


def _smooth_fill_grid(grid: np.ndarray, rounds: int = 1, min_neighbors: int = 4) -> np.ndarray:
    out = np.asarray(grid, dtype=float).copy()
    if out.size == 0:
        return out

    for _ in range(max(int(rounds), 0)):
        valid = np.isfinite(out)
        sum_acc = np.zeros_like(out, dtype=float)
        cnt_acc = np.zeros_like(out, dtype=float)

        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                src_y = slice(max(0, -dy), min(out.shape[0], out.shape[0] - dy))
                dst_y = slice(max(0, dy), min(out.shape[0], out.shape[0] + dy))
                src_x = slice(max(0, -dx), min(out.shape[1], out.shape[1] - dx))
                dst_x = slice(max(0, dx), min(out.shape[1], out.shape[1] + dx))

                vals = out[src_y, src_x]
                mask = valid[src_y, src_x]
                sum_acc[dst_y, dst_x] += np.where(mask, vals, 0.0)
                cnt_acc[dst_y, dst_x] += mask.astype(float)

        fill_mask = (~valid) & (cnt_acc >= float(min_neighbors))
        out[fill_mask] = sum_acc[fill_mask] / cnt_acc[fill_mask]

    return out


def _write_placeholder(png_path: Path, title: str, detail: str) -> None:
    plt.figure(figsize=(7.2, 5.8))
    ax = plt.gca()
    ax.axis("off")
    ax.text(0.5, 0.58, title, ha="center", va="center", fontsize=15, weight="bold")
    ax.text(0.5, 0.44, detail, ha="center", va="center", fontsize=11)
    plt.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(png_path, dpi=220)
    plt.close()


def _write_plot(png_path: Path, title: str, rows: list[dict[str, float]], mode: str, csv_path: Path) -> None:
    if not rows:
        _write_placeholder(png_path, title, "Waiting for sweep_results.csv data")
        return

    if mode == "branch-hmean":
        gate_grid = _load_gate_grid_from_run_meta(csv_path)
        axis_x, axis_y, grid = _build_branch_hmean_grid(rows, gate_grid)
        basis_text = (
            "x = H(g0, g1)\n"
            "y = H(g2, g3)\n"
            "H(a, b) = 2ab / (a + b)\n"
            "White = no solved point yet"
        )
        xlabel = "Branch A effective gate (harmonic mean)"
        ylabel = "Branch B effective gate (harmonic mean)"
    elif mode in {"pca", "pca-zscore", "pca-uncentered"}:
        x_edges, y_edges, grid, _ = _build_pca_grid(rows, mode=mode)
        _, _, _, _, xlabel, ylabel, basis_text = _project_pca_variant(rows, mode)
    else:
        x_edges, y_edges, grid, _ = _build_fixed_linear_grid(rows, mode)
        _, xlabel, ylabel, basis_text = _fixed_projection_spec(mode)

    plt.figure(figsize=(7.2, 5.8))
    ax = plt.gca()
    if mode == "branch-hmean":
        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color="white")
        x_edges = _compute_bin_edges(axis_x)
        y_edges = _compute_bin_edges(axis_y)
        valid = grid[np.isfinite(grid)]
        if valid.size:
            vmin = float(np.quantile(valid, 0.05))
            vmax = float(np.quantile(valid, 0.995))
            vmin = max(vmin, 1e-6)
            vmax = max(vmax, vmin * 1.01)
            norm = mcolors.LogNorm(vmin=vmin, vmax=vmax, clip=True)
        else:
            norm = None
        mesh = ax.pcolormesh(
            x_edges,
            y_edges,
            np.ma.masked_invalid(grid),
            cmap=cmap,
            norm=norm,
            shading="flat",
            edgecolors="none",
            linewidth=0.0,
            antialiased=False,
        )
        cbar = plt.colorbar(mesh, ax=ax)
        cbar.set_label("Relative linear-fit RMSE (log scale, clipped)")
    else:
        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color="white")
        valid = grid[np.isfinite(grid)]
        if valid.size:
            vmin = float(np.quantile(valid, 0.05))
            vmax = float(np.quantile(valid, 0.995))
            vmin = max(vmin, 1e-6)
            vmax = max(vmax, vmin * 1.01)
            norm = mcolors.LogNorm(vmin=vmin, vmax=vmax, clip=True)
        else:
            norm = None
        if mode in {"branch-mean", "total-balance", "branch-mismatch"}:
            mesh = ax.imshow(
                np.ma.masked_invalid(grid),
                cmap=cmap,
                norm=norm,
                origin="lower",
                interpolation="nearest",
                extent=[float(x_edges[0]), float(x_edges[-1]), float(y_edges[0]), float(y_edges[-1])],
                aspect="auto",
                resample=False,
            )
        else:
            mesh = ax.imshow(
                np.ma.masked_invalid(grid),
                cmap=cmap,
                norm=norm,
                origin="lower",
                interpolation="nearest",
                extent=[float(x_edges[0]), float(x_edges[-1]), float(y_edges[0]), float(y_edges[-1])],
                aspect="auto",
                resample=False,
            )
        cbar = plt.colorbar(mesh, ax=ax)
        cbar.set_label("Relative linear-fit RMSE (log scale, clipped)")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\n{len(rows)} solved configurations")
    ax.grid(True, alpha=0.2)
    ax.text(
        0.98,
        0.98,
        basis_text,
        transform=ax.transAxes,
        fontsize=8,
        ha="right",
        va="top",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )

    plt.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(png_path, dpi=220)
    plt.close()


def main() -> None:
    args = _parse_args()
    csv_path = args.csv.resolve()
    png_path = args.png.resolve()

    _write_placeholder(png_path, args.title, f"Watching\n{csv_path}")

    while True:
        rows = _load_rows(csv_path)
        _write_plot(png_path, args.title, rows, str(args.mode), csv_path)
        time.sleep(float(args.refresh_sec))


if __name__ == "__main__":
    main()
