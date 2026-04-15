#!/usr/bin/env python3
"""
Figure 3: combined scikit-digit and ionosphere classification panel.

This script writes two variants beside itself under `figures/fig3/`:
  - `fig3_classification_scaling.{png,pdf}`:
      2x3 layout with insets embedded in the accuracy panels.
  - `fig3_classification_scaling_separate.{png,pdf}`:
      2x4 layout where the decision-boundary triptych and misclassified digits
      each occupy their own panel.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

from matplotlib import colors
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIGIT_RUN = (
    ROOT
    / "scikit_digit"
    / "results"
    / "final_figures_22_jan"
    / "dense_best"
)
DEFAULT_IONO_RUN = (
    ROOT
    / "clln_ionosphere"
    / "results"
    / "sweeps"
    / "hidden5_struct_g3_ep39_20260413-173159"
    / "runs"
    / "struct_g3.0_ep39_g0.3_d0.1_m0.01_bfloating_grground_vg4.0_s0_a9sdVw"
)
DEFAULT_OUT_PREFIX = Path(__file__).resolve().parent / "fig3_classification_scaling"
DEFAULT_OUT_PREFIX_SEPARATE = Path(__file__).resolve().parent / "fig3_classification_scaling_separate"
DEFAULT_OUT_PREFIX_V2 = Path(__file__).resolve().parent / "fig3_classification_scaling_v2"
VG_NORM = colors.Normalize(vmin=0.4, vmax=8.0)
EDGE_CMAP = matplotlib.colormaps["viridis"]
IONO_CLASS_COLORS = ["#d95f02", "#1b9e77"]  # bad, good


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build Figure 3 classification panel")
    p.add_argument("--digit-run", type=Path, default=DEFAULT_DIGIT_RUN)
    p.add_argument("--iono-run", type=Path, default=DEFAULT_IONO_RUN)
    p.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT_PREFIX)
    p.add_argument("--width-in", type=float, default=11.2)
    p.add_argument("--height-in", type=float, default=6.9)
    return p.parse_args()


def _load_topology_npz(path: Path) -> dict[str, object]:
    data = np.load(path, allow_pickle=True)
    meta: dict[str, object] = {}
    if "meta" in data.files:
        meta = dict(data["meta"].item())
    elif "meta_json" in data.files:
        meta_raw = data["meta_json"]
        if getattr(meta_raw, "shape", ()) == ():
            meta = json.loads(str(meta_raw.item()))
        else:
            meta = json.loads(str(meta_raw[0]))
    return {
        "input_nodes": np.asarray(data["input_nodes"], dtype=int),
        "out_nodes": np.asarray(data["out_nodes"], dtype=int),
        "edges_D": np.asarray(data["edges_D"], dtype=int),
        "edges_S": np.asarray(data["edges_S"], dtype=int),
        "negref": int(np.asarray(data["negref"]).item()),
        "posref": int(np.asarray(data["posref"]).item()),
        "num_edges": int(np.asarray(data["edges_D"]).size if "num_edges" not in data.files else np.asarray(data["num_edges"]).item()),
        "meta": meta,
        "path": path,
    }


def _vg_epoch_num(path: Path) -> int:
    name = path.stem
    return int(name.rsplit("epoch", 1)[1])


def _load_vg_trajectory(run_dir: Path, n_edges: int, vg_init_meta: object) -> tuple[np.ndarray, np.ndarray]:
    vg_files = sorted(run_dir.glob("0_vg_unique_epoch*.npy"), key=_vg_epoch_num)
    if not vg_files:
        raise ValueError(f"No gate-voltage snapshots found under {run_dir}")

    epochs: list[int] = []
    states: list[np.ndarray] = []
    if isinstance(vg_init_meta, dict) and vg_init_meta.get("mode") == "fixed" and "fixed" in vg_init_meta:
        epochs.append(0)
        states.append(np.full(n_edges, float(vg_init_meta["fixed"]), dtype=float))

    for path in vg_files:
        epochs.append(_vg_epoch_num(path))
        states.append(np.asarray(np.load(path), dtype=float))
    return np.asarray(epochs, dtype=int), np.stack(states, axis=0)


def _style_axes(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.2, linewidth=0.5)
    ax.tick_params(labelsize=8, width=0.7, length=3.0)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)


def _annotate_panel(ax: plt.Axes, tag: str) -> None:
    ax.text(
        0.0,
        1.02,
        tag,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="bottom",
        ha="left",
        clip_on=False,
    )


def _edge_rgba(vg_values: np.ndarray) -> np.ndarray:
    frac = VG_NORM(np.clip(vg_values, VG_NORM.vmin, VG_NORM.vmax))
    rgba = EDGE_CMAP(frac)
    rgba[:, 3] = 0.08 + 0.42 * frac
    return rgba


def _edge_widths(vg_values: np.ndarray) -> np.ndarray:
    frac = VG_NORM(np.clip(vg_values, VG_NORM.vmin, VG_NORM.vmax))
    return 0.2 + 1.3 * frac


def _digit_split_indices(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    digits = load_digits()
    X = (digits.images / 16.0).astype(np.float64).reshape(len(digits.images), -1)
    y = digits.target.astype(int)
    idx = np.arange(len(X))
    _, X_test, _, y_test, _, idx_test = train_test_split(
        X, y, idx, test_size=0.2, random_state=seed, stratify=y
    )
    return X_test, y_test, idx_test


def _iono_full_dataset() -> tuple[np.ndarray, np.ndarray]:
    data_path = ROOT / "clln_ionosphere" / "data" / "ionosphere" / "ionosphere.data"
    X_list: list[list[float]] = []
    y_list: list[int] = []
    with data_path.open() as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) != 35:
                continue
            X_list.append([float(v) for v in parts[:34]])
            y_list.append(1 if parts[34] == "g" else 0)
    return np.asarray(X_list, dtype=float), np.asarray(y_list, dtype=int)


def _load_digit_run(run_dir: Path) -> dict[str, object]:
    meta = json.loads((run_dir / "run_meta.json").read_text())
    val_acc = np.load(run_dir / "0_val_acc.npy")
    peak_epoch = int(np.argmax(val_acc))
    final_epoch = int(len(val_acc) - 1)
    topology = _load_topology_npz(Path(meta["topology"]["path"]))
    test_x_saved = np.load(run_dir / "test_x.npy")
    test_y_saved = np.load(run_dir / "test_y.npy")
    X_test, y_test, idx_test = _digit_split_indices(int(meta["seed"]))
    if not np.allclose(X_test, test_x_saved):
        raise ValueError("Saved scikit-digit held-out split does not match reconstructed split")
    if not np.array_equal(y_test, test_y_saved):
        raise ValueError("Saved scikit-digit labels do not match reconstructed split")
    vout_peak = np.load(run_dir / f"0_vout_test_epoch{peak_epoch}.npy")
    pred_peak = np.argmax(vout_peak, axis=1)
    wrong_peak = np.where(pred_peak != test_y_saved)[0]
    vout_final = np.load(run_dir / f"0_vout_test_epoch{final_epoch}.npy")
    pred_final = np.argmax(vout_final, axis=1)
    wrong_final = np.where(pred_final != test_y_saved)[0]
    n_edges = int(topology["num_edges"])
    vg_epochs, vg_traj = _load_vg_trajectory(run_dir, n_edges=n_edges, vg_init_meta=meta.get("vg_init"))
    return {
        "run_dir": run_dir,
        "meta": meta,
        "val_acc": val_acc,
        "train_acc": np.load(run_dir / "0_train_acc.npy"),
        "val_hinge": np.load(run_dir / "0_val_hinge.npy"),
        "train_hinge": np.load(run_dir / "0_train_hinge.npy"),
        "peak_epoch": peak_epoch,
        "final_epoch": final_epoch,
        "peak_acc": float(val_acc[peak_epoch]),
        "final_acc": float(val_acc[final_epoch]),
        "vg_peak": np.load(run_dir / f"0_vg_unique_epoch{peak_epoch}.npy"),
        "vg_final": np.load(run_dir / f"0_vg_unique_epoch{final_epoch}.npy"),
        "vg_epochs": vg_epochs,
        "vg_traj": vg_traj,
        "topology": topology,
        "test_x": test_x_saved,
        "test_y": test_y_saved,
        "test_idx": idx_test,
        "vout_peak": vout_peak,
        "pred_peak": pred_peak,
        "wrong_idx_peak": wrong_peak,
        "vout_final": vout_final,
        "pred_final": pred_final,
        "wrong_idx_final": wrong_final,
        "digit_examples_epoch_kind": "peak",
    }


def _load_iono_run(run_dir: Path) -> dict[str, object]:
    meta = json.loads((run_dir / "run_meta.json").read_text())
    val_acc = np.load(run_dir / "0_val_acc.npy")
    topology = _load_topology_npz(Path(meta["topology"]["path"]))
    X_full, y_full = _iono_full_dataset()
    split_seed = int(meta["seed"])
    X_train, X_test, y_train, y_test = train_test_split(
        X_full,
        y_full,
        test_size=0.2,
        random_state=split_seed,
        stratify=y_full,
    )
    test_x_saved = np.load(run_dir / "test_x.npy")
    test_y_saved = np.load(run_dir / "test_y.npy")
    if not np.allclose(X_test, test_x_saved):
        raise ValueError("Saved ionosphere held-out split does not match reconstructed split")
    if not np.array_equal(y_test, test_y_saved):
        raise ValueError("Saved ionosphere labels do not match reconstructed split")
    n_edges = int(topology["num_edges"])
    vg_epochs, vg_traj = _load_vg_trajectory(run_dir, n_edges=n_edges, vg_init_meta=meta.get("vg_init"))
    return {
        "run_dir": run_dir,
        "meta": meta,
        "val_acc": val_acc,
        "train_acc": np.load(run_dir / "0_train_acc.npy"),
        "val_hinge": np.load(run_dir / "0_val_hinge.npy"),
        "train_hinge": np.load(run_dir / "0_train_hinge.npy"),
        "topology": topology,
        "test_x": test_x_saved,
        "test_y": test_y_saved,
        "vout_init": np.load(run_dir / "0_vout_test_epoch0.npy"),
        "vout_ep50": np.load(run_dir / "0_vout_test_epoch50.npy"),
        "vout_peak": np.load(run_dir / f"0_vout_test_epoch{int(np.argmax(val_acc))}.npy"),
        "vout_final": np.load(run_dir / f"0_vout_test_epoch{len(val_acc) - 1}.npy"),
        "vg_final": np.load(run_dir / f"0_vg_unique_epoch{len(val_acc) - 1}.npy"),
        "vg_epochs": vg_epochs,
        "vg_traj": vg_traj,
        "peak_epoch": int(np.argmax(val_acc)),
        "peak_acc": float(np.max(val_acc)),
    }


def _digit_positions(topo: dict[str, object]) -> dict[int, tuple[float, float]]:
    pos: dict[int, tuple[float, float]] = {}
    input_nodes = np.asarray(topo["input_nodes"], dtype=int)
    out_nodes = np.asarray(topo["out_nodes"], dtype=int)

    center = np.array([0.0, 0.0], dtype=float)
    pitch = 0.82
    offset = 3.5 * pitch
    for idx, node in enumerate(input_nodes):
        row = idx // 8
        col = idx % 8
        x = (col * pitch) - offset
        y = offset - (row * pitch)
        pos[int(node)] = (float(x), float(y))

    # Arrange the outputs around a square frame in clockwise order.
    frame = 5.1
    square_positions = [
        (0.0, frame),
        (0.68 * frame, frame),
        (frame, 0.42 * frame),
        (frame, -0.42 * frame),
        (0.68 * frame, -frame),
        (0.0, -frame),
        (-0.68 * frame, -frame),
        (-frame, -0.42 * frame),
        (-frame, 0.42 * frame),
        (-0.68 * frame, frame),
    ]
    for idx, node in enumerate(out_nodes):
        pos[int(node)] = square_positions[idx % len(square_positions)]
    return pos


def _iono_positions(topo: dict[str, object]) -> dict[int, tuple[float, float]]:
    pos: dict[int, tuple[float, float]] = {}
    input_nodes = np.asarray(topo["input_nodes"], dtype=int)
    out_nodes = np.asarray(topo["out_nodes"], dtype=int)
    hidden_nodes = sorted(
        int(n)
        for n in set(np.asarray(topo["edges_D"], dtype=int).tolist() + np.asarray(topo["edges_S"], dtype=int).tolist())
        if n not in input_nodes and n not in out_nodes and n not in {int(topo["negref"]), int(topo["posref"])}
    )

    for idx, node in enumerate(input_nodes):
        row = idx % 17
        col = idx // 17
        pos[int(node)] = (0.0 + 0.12 * col, float(16 - row))

    for idx, node in enumerate(hidden_nodes):
        pos[int(node)] = (0.62, 14.0 - 3.5 * idx)

    out_labels_y = [11.0, 5.0]
    for idx, node in enumerate(out_nodes):
        pos[int(node)] = (1.12, out_labels_y[idx])

    return pos


def _draw_topology(
    ax: plt.Axes,
    pos: dict[int, tuple[float, float]],
    edges_d: np.ndarray,
    edges_s: np.ndarray,
    vg_values: np.ndarray,
    node_groups: dict[str, tuple[np.ndarray, dict[str, object]]],
) -> None:
    segments = [
        [pos[int(d)], pos[int(s)]]
        for d, s in zip(edges_d.tolist(), edges_s.tolist())
        if int(d) in pos and int(s) in pos
    ]
    lc = LineCollection(
        segments,
        colors=_edge_rgba(vg_values),
        linewidths=_edge_widths(vg_values),
        zorder=1,
    )
    ax.add_collection(lc)

    for _, (nodes, style) in node_groups.items():
        xy = np.array([pos[int(node)] for node in nodes], dtype=float)
        ax.scatter(xy[:, 0], xy[:, 1], zorder=3, **style)


def _scikit_structure_panel(ax: plt.Axes, digit_run: dict[str, object]) -> None:
    topo = digit_run["topology"]
    pos = _digit_positions(topo)
    edges_d = np.asarray(topo["edges_D"], dtype=int)
    edges_s = np.asarray(topo["edges_S"], dtype=int)
    vg_values = np.asarray(digit_run["vg_peak"], dtype=float)

    node_groups = {
        "inputs": (
            np.asarray(topo["input_nodes"], dtype=int),
            {
                "s": 24,
                "marker": "s",
                "facecolor": "#f2f2f2",
                "edgecolor": "#9a9a9a",
                "linewidth": 0.35,
            },
        ),
        "outputs": (
            np.asarray(topo["out_nodes"], dtype=int),
            {
                "s": 105,
                "marker": "o",
                "facecolor": "#fff2c6",
                "edgecolor": "#8c6d1f",
                "linewidth": 0.8,
            },
        ),
    }
    _draw_topology(ax, pos, edges_d, edges_s, vg_values, node_groups)

    for idx, node in enumerate(np.asarray(topo["out_nodes"], dtype=int)):
        x, y = pos[int(node)]
        ax.text(x, y, str(idx), fontsize=7.5, va="center", ha="center")

    ax.set_xlim(-5.9, 5.9)
    ax.set_ylim(-5.9, 5.9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    ax.set_aspect("equal")


def _iono_boundary_inset(
    ax: plt.Axes,
    X: np.ndarray,
    y_true: np.ndarray,
    pred_labels: np.ndarray,
    title: str,
    pca: PCA,
) -> None:
    xy = pca.transform(X)
    x_min, x_max = float(xy[:, 0].min()), float(xy[:, 0].max())
    y_min, y_max = float(xy[:, 1].min()), float(xy[:, 1].max())
    dx = 0.18 * (x_max - x_min + 1e-9)
    dy = 0.18 * (y_max - y_min + 1e-9)
    xs = np.linspace(x_min - dx, x_max + dx, 140)
    ys = np.linspace(y_min - dy, y_max + dy, 140)
    xx, yy = np.meshgrid(xs, ys)

    unique_pred = np.unique(pred_labels)
    if unique_pred.size == 1:
        zz = np.full_like(xx, fill_value=float(unique_pred[0]), dtype=float)
    else:
        knn = KNeighborsClassifier(n_neighbors=min(7, len(pred_labels)), weights="distance")
        knn.fit(xy, pred_labels)
        zz = knn.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    cmap = colors.ListedColormap(["#f7d7b8", "#c9e6dc"])
    ax.contourf(xx, yy, zz, levels=[-0.5, 0.5, 1.5], cmap=cmap, alpha=0.95)
    for cls, marker in [(0, "o"), (1, "^")]:
        mask = y_true == cls
        if np.any(mask):
            ax.scatter(
                xy[mask, 0],
                xy[mask, 1],
                s=18,
                marker=marker,
                facecolor=IONO_CLASS_COLORS[cls],
                edgecolor="black",
                linewidth=0.35,
                alpha=0.9,
            )
    ax.set_title(title, fontsize=7, pad=1.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
        spine.set_edgecolor("#555555")


def _ionosphere_structure_panel(ax: plt.Axes, iono_run: dict[str, object]) -> None:
    topo = iono_run["topology"]
    pos = _iono_positions(topo)
    edges_d = np.asarray(topo["edges_D"], dtype=int)
    edges_s = np.asarray(topo["edges_S"], dtype=int)
    vg_values = np.asarray(iono_run["vg_final"], dtype=float)

    input_nodes = np.asarray(topo["input_nodes"], dtype=int)
    out_nodes = np.asarray(topo["out_nodes"], dtype=int)
    hidden_nodes = np.array(
        sorted(
            int(n)
            for n in set(edges_d.tolist() + edges_s.tolist())
            if n not in input_nodes and n not in out_nodes and n not in {int(topo["negref"]), int(topo["posref"])}
        ),
        dtype=int,
    )

    node_groups = {
        "inputs": (
            input_nodes,
            {
                "s": 24,
                "marker": "s",
                "facecolor": "#f2f2f2",
                "edgecolor": "#9a9a9a",
                "linewidth": 0.35,
            },
        ),
        "hidden": (
            hidden_nodes,
            {
                "s": 82,
                "marker": "h",
                "facecolor": "#d5ecff",
                "edgecolor": "#1f78b4",
                "linewidth": 0.7,
            },
        ),
        "outputs": (
            out_nodes,
            {
                "s": 105,
                "marker": "o",
                "facecolor": "#f4e6ff",
                "edgecolor": "#7a3db8",
                "linewidth": 0.8,
            },
        ),
    }
    _draw_topology(ax, pos, edges_d, edges_s, vg_values, node_groups)

    for idx, node in enumerate(hidden_nodes):
        x, y = pos[int(node)]
        ax.text(x, y + 0.95, f"h{idx + 1}", fontsize=7, ha="center", va="bottom")

    output_labels = ["bad", "good"]
    for idx, node in enumerate(out_nodes):
        x, y = pos[int(node)]
        ax.text(x + 0.08, y, output_labels[idx], fontsize=8, ha="left", va="center")

    ax.set_xlim(-0.08, 1.34)
    ax.set_ylim(-1.0, 17.0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

def _accuracy_panel(
    ax: plt.Axes,
    train_acc: np.ndarray,
    heldout_acc: np.ndarray,
    highlight_epoch: int,
    legend_loc: str = "lower right",
) -> None:
    ep_train = np.arange(1, min(train_acc.size, highlight_epoch) + 1)
    ep_hold = np.arange(highlight_epoch + 1)
    ax.plot(ep_train, train_acc[: ep_train.size], color="#4c78a8", linewidth=1.5, label="train")
    ax.plot(ep_hold, heldout_acc[: ep_hold.size], color="#e45756", linewidth=1.7, label="held-out")
    ax.axvline(highlight_epoch, color="#555555", linestyle="--", linewidth=0.9, alpha=0.7)
    ax.scatter([highlight_epoch], [heldout_acc[highlight_epoch]], color="#e45756", s=24, zorder=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.02)
    ax.set_xlim(0, highlight_epoch)
    ax.legend(frameon=False, loc=legend_loc, fontsize=7)
    _style_axes(ax)


def _loss_panel(
    ax: plt.Axes,
    train_loss: np.ndarray,
    heldout_loss: np.ndarray,
    highlight_epoch: int,
) -> None:
    ep_train = np.arange(1, min(train_loss.size, highlight_epoch) + 1)
    ep_hold = np.arange(highlight_epoch + 1)
    floor = np.finfo(float).tiny
    ax.plot(ep_train, np.clip(train_loss[: ep_train.size], floor, None), color="#4c78a8", linewidth=1.5, label="train")
    ax.plot(ep_hold, np.clip(heldout_loss[: ep_hold.size], floor, None), color="#72b7b2", linewidth=1.7, label="held-out")
    ax.axvline(highlight_epoch, color="#555555", linestyle="--", linewidth=0.9, alpha=0.7)
    ax.scatter([highlight_epoch], [heldout_loss[highlight_epoch]], color="#72b7b2", s=24, zorder=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Hinge Loss")
    ax.set_yscale("log")
    ax.set_xlim(0, highlight_epoch)
    ax.legend(frameon=False, loc="upper right", fontsize=7)
    _style_axes(ax)


def _vg_distribution_panel(ax: plt.Axes, vg_final: np.ndarray) -> None:
    bins = np.linspace(VG_NORM.vmin, VG_NORM.vmax, 19)
    ax.hist(
        np.asarray(vg_final, dtype=float),
        bins=bins,
        color="#4c78a8",
        alpha=0.82,
        edgecolor="white",
        linewidth=0.6,
    )
    mean_v = float(np.mean(vg_final))
    ax.axvline(mean_v, color="#e45756", linewidth=1.2, linestyle="--")
    ax.set_xlabel("$V_G$ at final epoch (V)")
    ax.set_ylabel("Edge count")
    ax.set_xlim(VG_NORM.vmin, VG_NORM.vmax)
    _style_axes(ax)


def _vg_evolution_panel(ax: plt.Axes, vg_epochs: np.ndarray, vg_traj: np.ndarray) -> None:
    vg_epochs = np.asarray(vg_epochs, dtype=int)
    vg_traj = np.asarray(vg_traj, dtype=float)
    if vg_traj.shape[0] < 2:
        ax.scatter([0.0], [0.0], s=18, color="#4c78a8")
        ax.set_xticks([])
        ax.set_yticks([])
        return

    pca = PCA(n_components=2)
    xy = pca.fit_transform(vg_traj)
    segments = [[xy[i], xy[i + 1]] for i in range(xy.shape[0] - 1)]
    epoch_frac = np.linspace(0.0, 1.0, len(segments)) if segments else np.array([])
    lc = LineCollection(
        segments,
        cmap=matplotlib.colormaps["plasma"],
        norm=colors.Normalize(vmin=0.0, vmax=1.0),
        linewidths=1.8,
        zorder=1,
    )
    if len(segments):
        lc.set_array(epoch_frac)
        ax.add_collection(lc)
    ax.scatter(xy[:, 0], xy[:, 1], c=np.linspace(0.0, 1.0, xy.shape[0]), cmap="plasma", s=15, zorder=2)
    ax.scatter([xy[0, 0]], [xy[0, 1]], color="#2f2f2f", s=30, marker="o", zorder=3)
    ax.scatter([xy[-1, 0]], [xy[-1, 1]], color="#2f2f2f", s=34, marker="X", zorder=3)
    ax.text(xy[0, 0], xy[0, 1], f"  {int(vg_epochs[0])}", fontsize=7, ha="left", va="bottom")
    ax.text(xy[-1, 0], xy[-1, 1], f"  {int(vg_epochs[-1])}", fontsize=7, ha="left", va="bottom")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    _style_axes(ax)
    ax.set_aspect("equal", adjustable="datalim")


def _ionosphere_boundary_triptych(
    host_ax: plt.Axes,
    iono_run: dict[str, object],
    boxes: list[list[float]],
) -> None:
    pca = PCA(n_components=2)
    pca.fit(iono_run["test_x"])
    init_pred = np.argmax(np.asarray(iono_run["vout_init"], dtype=float), axis=1)
    mid_pred = np.argmax(np.asarray(iono_run["vout_ep50"], dtype=float), axis=1)
    final_pred = np.argmax(np.asarray(iono_run["vout_peak"], dtype=float), axis=1)
    acc0 = float(np.mean(init_pred == np.asarray(iono_run["test_y"], dtype=int)))
    accm = float(np.mean(mid_pred == np.asarray(iono_run["test_y"], dtype=int)))
    accf = float(np.mean(final_pred == np.asarray(iono_run["test_y"], dtype=int)))

    specs = [
        (init_pred, f"epoch 0\n{100.0 * acc0:.1f}%"),
        (mid_pred, f"epoch 50\n{100.0 * accm:.1f}%"),
        (final_pred, f"epoch {int(iono_run['peak_epoch'])}\n{100.0 * accf:.1f}%"),
    ]
    for box, (pred_labels, title) in zip(boxes, specs):
        iax = host_ax.inset_axes(box)
        _iono_boundary_inset(
            iax,
            iono_run["test_x"],
            iono_run["test_y"],
            pred_labels,
            title,
            pca,
        )


def _add_iono_accuracy_insets(ax: plt.Axes, iono_run: dict[str, object]) -> None:
    _ionosphere_boundary_triptych(
        ax,
        iono_run,
        [
            [0.03, 0.12, 0.28, 0.22],
            [0.36, 0.12, 0.28, 0.22],
            [0.69, 0.12, 0.28, 0.22],
        ],
    )


def _draw_scikit_misclassified_strip(ax: plt.Axes, digit_run: dict[str, object], boxed: bool) -> None:
    epoch_kind = str(digit_run.get("digit_examples_epoch_kind", "peak"))
    if epoch_kind == "final":
        wrong = np.asarray(digit_run["wrong_idx_final"], dtype=int)
        pred = np.asarray(digit_run["pred_final"], dtype=int)
        vout = np.asarray(digit_run["vout_final"], dtype=float)
        epoch_num = int(digit_run["final_epoch"])
        epoch_label = "final epoch"
    else:
        wrong = np.asarray(digit_run["wrong_idx_peak"], dtype=int)
        pred = np.asarray(digit_run["pred_peak"], dtype=int)
        vout = np.asarray(digit_run["vout_peak"], dtype=float)
        epoch_num = int(digit_run["peak_epoch"])
        epoch_label = "peak epoch"
    test_x = np.asarray(digit_run["test_x"], dtype=float)
    test_y = np.asarray(digit_run["test_y"], dtype=int)

    holder = ax
    if boxed:
        holder = ax.inset_axes([0.07, 0.04, 0.86, 0.27], transform=ax.transAxes)
        holder.set_xticks([])
        holder.set_yticks([])
        holder.set_facecolor("white")
        for spine in holder.spines.values():
            spine.set_linewidth(0.5)
            spine.set_edgecolor("#777777")
    else:
        holder.set_xticks([])
        holder.set_yticks([])
        holder.set_frame_on(False)

    shown = wrong[:6]
    count = max(1, int(shown.size))
    margin_x = 0.02
    gap = 0.015
    inset_w = (1.0 - 2.0 * margin_x - (count - 1) * gap) / count
    img_y0 = 0.30 if boxed else 0.26
    img_h = 0.54 if boxed else 0.66
    for j, pos_idx in enumerate(shown):
        x0 = margin_x + j * (inset_w + gap)
        iax = holder.inset_axes([x0, img_y0, inset_w, img_h])
        iax.imshow(test_x[pos_idx].reshape(8, 8) * 16.0, cmap="gray_r", vmin=0.0, vmax=16.0, interpolation="nearest")
        iax.set_xticks([])
        iax.set_yticks([])
        for spine in iax.spines.values():
            spine.set_linewidth(0.45)
            spine.set_edgecolor("#555555")
        y_true = int(test_y[pos_idx])
        y_pred = int(pred[pos_idx])
        delta_mv = 1e3 * float(vout[pos_idx, y_pred] - vout[pos_idx, y_true])
        holder.text(
            x0 + 0.5 * inset_w,
            0.11 if boxed else 0.12,
            f"T={y_true}  P={y_pred}\n$\\Delta V$={delta_mv:.1f} mV",
            fontsize=6.2 if boxed else 6.8,
            ha="center",
            va="center",
        )
    if boxed:
        holder.text(0.02, 0.98, f"{epoch_label} {epoch_num} misclassified digits", fontsize=6.6, ha="left", va="top")
    else:
        holder.text(0.02, 0.98, f"{epoch_label} {epoch_num} misclassified digits", fontsize=7.4, ha="left", va="top")


def _add_scikit_accuracy_inset(ax: plt.Axes, digit_run: dict[str, object]) -> None:
    _draw_scikit_misclassified_strip(ax, digit_run, boxed=True)


def _standalone_iono_boundary_panel(ax: plt.Axes, iono_run: dict[str, object]) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    _ionosphere_boundary_triptych(
        ax,
        iono_run,
        [
            [0.02, 0.18, 0.30, 0.64],
            [0.35, 0.18, 0.30, 0.64],
            [0.68, 0.18, 0.30, 0.64],
        ],
    )


def _standalone_scikit_misclassified_panel(ax: plt.Axes, digit_run: dict[str, object]) -> None:
    _draw_scikit_misclassified_strip(ax, digit_run, boxed=False)


def _add_shared_colorbar(fig: plt.Figure) -> None:
    sm = matplotlib.cm.ScalarMappable(norm=VG_NORM, cmap=EDGE_CMAP)
    cax = fig.add_axes([0.025, 0.23, 0.015, 0.50])
    cb = fig.colorbar(sm, cax=cax, orientation="vertical")
    cb.set_label("Gate voltage $V_G$ (V)", fontsize=8, labelpad=4)
    cb.ax.tick_params(labelsize=7, width=0.6, length=2.5)


def _shrink_axis_box(ax: plt.Axes, width_frac: float, height_frac: float) -> None:
    bbox = ax.get_position()
    new_w = bbox.width * width_frac
    new_h = bbox.height * height_frac
    new_x = bbox.x0 + 0.5 * (bbox.width - new_w)
    new_y = bbox.y0 + 0.5 * (bbox.height - new_h)
    ax.set_position([new_x, new_y, new_w, new_h])


def _save_figure(fig: plt.Figure, out_prefix: Path) -> tuple[Path, Path]:
    png_path = out_prefix.with_suffix(".png")
    pdf_path = out_prefix.with_suffix(".pdf")
    fig.savefig(png_path, dpi=400, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return png_path, pdf_path


def _build_inset_layout(
    out_prefix: Path,
    digit_run: dict[str, object],
    iono_run: dict[str, object],
    width_in: float,
    height_in: float,
) -> tuple[Path, Path]:
    fig, axes = plt.subplots(2, 3, figsize=(width_in, height_in))
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.07, top=0.91, wspace=0.18, hspace=0.24)
    _add_shared_colorbar(fig)

    _ionosphere_structure_panel(axes[0, 0], iono_run)
    _annotate_panel(axes[0, 0], "A")
    _accuracy_panel(
        axes[0, 1],
        np.asarray(iono_run["train_acc"], dtype=float),
        np.asarray(iono_run["val_acc"], dtype=float),
        int(iono_run["peak_epoch"]),
        legend_loc="upper left",
    )
    _add_iono_accuracy_insets(axes[0, 1], iono_run)
    _annotate_panel(axes[0, 1], "B")
    _loss_panel(
        axes[0, 2],
        np.asarray(iono_run["train_hinge"], dtype=float),
        np.asarray(iono_run["val_hinge"], dtype=float),
        int(iono_run["peak_epoch"]),
    )
    _annotate_panel(axes[0, 2], "C")

    _scikit_structure_panel(axes[1, 0], digit_run)
    _annotate_panel(axes[1, 0], "D")
    _accuracy_panel(
        axes[1, 1],
        np.asarray(digit_run["train_acc"], dtype=float),
        np.asarray(digit_run["val_acc"], dtype=float),
        int(digit_run["peak_epoch"]),
        legend_loc="upper left",
    )
    _add_scikit_accuracy_inset(axes[1, 1], digit_run)
    _annotate_panel(axes[1, 1], "E")
    _loss_panel(
        axes[1, 2],
        np.asarray(digit_run["train_hinge"], dtype=float),
        np.asarray(digit_run["val_hinge"], dtype=float),
        int(digit_run["peak_epoch"]),
    )
    _annotate_panel(axes[1, 2], "F")
    return _save_figure(fig, out_prefix)


def _build_separate_layout(
    out_prefix: Path,
    digit_run: dict[str, object],
    iono_run: dict[str, object],
) -> tuple[Path, Path]:
    fig = plt.figure(figsize=(14.2, 6.9))
    gs = fig.add_gridspec(
        2,
        4,
        width_ratios=[3.35, 1.0, 1.0, 2.05],
        height_ratios=[1.0, 1.0],
        left=0.085,
        right=0.99,
        bottom=0.07,
        top=0.91,
        wspace=0.24,
        hspace=0.26,
    )
    axes = np.empty((2, 4), dtype=object)
    for row in range(2):
        for col in range(4):
            axes[row, col] = fig.add_subplot(gs[row, col])
    _add_shared_colorbar(fig)

    _ionosphere_structure_panel(axes[0, 0], iono_run)
    _annotate_panel(axes[0, 0], "A")
    _accuracy_panel(
        axes[0, 1],
        np.asarray(iono_run["train_acc"], dtype=float),
        np.asarray(iono_run["val_acc"], dtype=float),
        int(iono_run["peak_epoch"]),
        legend_loc="upper left",
    )
    _shrink_axis_box(axes[0, 1], width_frac=0.92, height_frac=0.60)
    _annotate_panel(axes[0, 1], "B")
    _loss_panel(
        axes[0, 2],
        np.asarray(iono_run["train_hinge"], dtype=float),
        np.asarray(iono_run["val_hinge"], dtype=float),
        int(iono_run["peak_epoch"]),
    )
    _shrink_axis_box(axes[0, 2], width_frac=0.92, height_frac=0.60)
    _annotate_panel(axes[0, 2], "C")
    _standalone_iono_boundary_panel(axes[0, 3], iono_run)
    _annotate_panel(axes[0, 3], "D")

    _scikit_structure_panel(axes[1, 0], digit_run)
    _annotate_panel(axes[1, 0], "E")
    _accuracy_panel(
        axes[1, 1],
        np.asarray(digit_run["train_acc"], dtype=float),
        np.asarray(digit_run["val_acc"], dtype=float),
        int(digit_run["peak_epoch"]),
        legend_loc="upper left",
    )
    _shrink_axis_box(axes[1, 1], width_frac=0.92, height_frac=0.60)
    _annotate_panel(axes[1, 1], "F")
    _loss_panel(
        axes[1, 2],
        np.asarray(digit_run["train_hinge"], dtype=float),
        np.asarray(digit_run["val_hinge"], dtype=float),
        int(digit_run["peak_epoch"]),
    )
    _shrink_axis_box(axes[1, 2], width_frac=0.92, height_frac=0.60)
    _annotate_panel(axes[1, 2], "G")
    _standalone_scikit_misclassified_panel(axes[1, 3], digit_run)
    _annotate_panel(axes[1, 3], "H")
    return _save_figure(fig, out_prefix)


def _build_v2_layout(
    out_prefix: Path,
    digit_run: dict[str, object],
    iono_run: dict[str, object],
) -> tuple[Path, Path]:
    fig = plt.figure(figsize=(17.8, 6.9))
    gs = fig.add_gridspec(
        2,
        5,
        width_ratios=[2.15, 1.9, 1.1, 1.15, 1.15],
        height_ratios=[1.0, 1.0],
        left=0.07,
        right=0.99,
        bottom=0.08,
        top=0.92,
        wspace=0.30,
        hspace=0.26,
    )
    axes = np.empty((2, 5), dtype=object)
    for row in range(2):
        for col in range(5):
            axes[row, col] = fig.add_subplot(gs[row, col])
    _add_shared_colorbar(fig)

    _ionosphere_structure_panel(axes[0, 0], iono_run)
    _annotate_panel(axes[0, 0], "A")
    _accuracy_panel(
        axes[0, 1],
        np.asarray(iono_run["train_acc"], dtype=float),
        np.asarray(iono_run["val_acc"], dtype=float),
        int(iono_run["peak_epoch"]),
        legend_loc="upper left",
    )
    _add_iono_accuracy_insets(axes[0, 1], iono_run)
    _annotate_panel(axes[0, 1], "B")
    _loss_panel(
        axes[0, 2],
        np.asarray(iono_run["train_hinge"], dtype=float),
        np.asarray(iono_run["val_hinge"], dtype=float),
        int(iono_run["peak_epoch"]),
    )
    _annotate_panel(axes[0, 2], "C")
    _vg_distribution_panel(axes[0, 3], np.asarray(iono_run["vg_final"], dtype=float))
    _annotate_panel(axes[0, 3], "D")
    _vg_evolution_panel(
        axes[0, 4],
        np.asarray(iono_run["vg_epochs"], dtype=int),
        np.asarray(iono_run["vg_traj"], dtype=float),
    )
    _annotate_panel(axes[0, 4], "E")

    _scikit_structure_panel(axes[1, 0], digit_run)
    _annotate_panel(axes[1, 0], "F")
    _accuracy_panel(
        axes[1, 1],
        np.asarray(digit_run["train_acc"], dtype=float),
        np.asarray(digit_run["val_acc"], dtype=float),
        int(digit_run["peak_epoch"]),
        legend_loc="upper left",
    )
    _add_scikit_accuracy_inset(axes[1, 1], digit_run)
    _annotate_panel(axes[1, 1], "G")
    _loss_panel(
        axes[1, 2],
        np.asarray(digit_run["train_hinge"], dtype=float),
        np.asarray(digit_run["val_hinge"], dtype=float),
        int(digit_run["peak_epoch"]),
    )
    _annotate_panel(axes[1, 2], "H")
    _vg_distribution_panel(axes[1, 3], np.asarray(digit_run["vg_final"], dtype=float))
    _annotate_panel(axes[1, 3], "I")
    _vg_evolution_panel(
        axes[1, 4],
        np.asarray(digit_run["vg_epochs"], dtype=int),
        np.asarray(digit_run["vg_traj"], dtype=float),
    )
    _annotate_panel(axes[1, 4], "J")
    return _save_figure(fig, out_prefix)


def main() -> None:
    args = _parse_args()
    digit_run = _load_digit_run(args.digit_run.resolve())
    iono_run = _load_iono_run(args.iono_run.resolve())

    out_prefix = args.out_prefix.resolve()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    out_prefix_separate = out_prefix.parent / DEFAULT_OUT_PREFIX_SEPARATE.name
    out_prefix_v2 = out_prefix.parent / DEFAULT_OUT_PREFIX_V2.name

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 11,
            "legend.fontsize": 7,
        }
    )

    png_path, pdf_path = _build_inset_layout(
        out_prefix,
        digit_run,
        iono_run,
        float(args.width_in),
        float(args.height_in),
    )
    png_path_separate, pdf_path_separate = _build_separate_layout(
        out_prefix_separate,
        digit_run,
        iono_run,
    )
    png_path_v2, pdf_path_v2 = _build_v2_layout(
        out_prefix_v2,
        digit_run,
        iono_run,
    )

    print(f"digit_run={args.digit_run.resolve()}")
    print(f"iono_run={args.iono_run.resolve()}")
    print(png_path)
    print(pdf_path)
    print(png_path_separate)
    print(pdf_path_separate)
    print(png_path_v2)
    print(pdf_path_v2)


if __name__ == "__main__":
    main()
