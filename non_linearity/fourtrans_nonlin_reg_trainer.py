#!/usr/bin/env python3
"""
4-transistor nonlinear-regression trainer built from saved sweep outputs.

The training target is chosen from an already-measured frozen transfer curve so
the task stays inside the behavior this motif has already exhibited. Training
starts from a measured near-linear gate configuration and uses an alter/clamp
update rule similar to the existing nonlinear-regression trainer.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from PySpice.Spice.NgSpice.Shared import NgSpiceShared
from sklearn.decomposition import PCA


REPO_ROOT = Path(__file__).resolve().parents[1]
DEVICE_LIB_PATH = str(REPO_ROOT / "device_model" / "nmos_lvl1_ald1106.lib")
DEVICE_SUBCKT = "NMOSWRAP"
VG_CLIP_LO, VG_CLIP_HI = 0.4, 8.0
DEFAULT_SWEEP_RUN = REPO_ROOT / "non_linearity" / "results" / "runs" / "fourtrans_50pt_1to5V_live_20260412"


def _exec_chunked(ng, cmds: Iterable[str], max_len: int = 900, sep: str = "; ") -> None:
    buf: List[str] = []
    length = 0
    for c in cmds:
        c = str(c)
        cl = len(c)
        if buf and (length + len(sep) + cl) > max_len:
            ng.exec_command(sep.join(buf))
            buf = [c]
            length = cl
        else:
            length = (length + len(sep) + cl) if buf else cl
            buf.append(c)
    if buf:
        ng.exec_command(sep.join(buf))


def get_voltages(ng, nodes: Iterable[int]) -> np.ndarray:
    s = ng.exec_command("print allv")
    nodemap: Dict[int, float] = {}
    for line in s.splitlines():
        line = line.strip()
        if not line.startswith("v("):
            continue
        try:
            k, v = line.split(" = ")
            nodemap[int(k[2:-1])] = float(v)
        except Exception:
            continue
    return np.array([float(nodemap.get(int(n), float("nan"))) for n in nodes], dtype=float)


def _require_finite(label: str, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)):
        raise RuntimeError(f"{label} contains non-finite voltages: {values}")
    return values


def alter_input_single(ng, vin: float) -> None:
    _exec_chunked(ng, [f"alter VVIN dc = {float(vin):.16f}"])


def alter_gate_values(ng, gate_values: Sequence[float]) -> None:
    cmds: List[str] = []
    for k, gv in enumerate(gate_values):
        gv = float(np.clip(gv, VG_CLIP_LO, VG_CLIP_HI))
        cmds.append(f"alter v.x{k}.v1 dc = {gv:.16f}")
    _exec_chunked(ng, cmds)


def mk_free(ng) -> None:
    _exec_chunked(ng, ["alter RSO 1e12"])


def alter_output_single(ng, oc: float, clamp_res: float) -> None:
    _exec_chunked(
        ng,
        [
            f"alter RSO {float(clamp_res):.16f}",
            f"alter VCLO dc = {float(oc):.16f}",
        ],
    )


def _build_motif_graph() -> Tuple[Dict[str, int], List[Tuple[int, int]]]:
    node_map = {"vin": 1, "out": 2, "h1": 3, "h2": 4}
    edge_list = [
        (node_map["vin"], node_map["h1"]),
        (node_map["h1"], node_map["out"]),
        (node_map["vin"], node_map["h2"]),
        (node_map["h2"], node_map["out"]),
    ]
    return node_map, edge_list


def mk_training_netlist(
    edge_list: List[Tuple[int, int]],
    weights: np.ndarray,
    node_map: Dict[str, int],
    load_res: float,
    ground_node_name: str,
    solver: str = "klu",
) -> str:
    weights = np.asarray(weights, dtype=float).reshape(-1)
    lines: List[str] = []
    lines.append(".title fourtrans_nonlin_reg_ald1106")
    lines.append(f'.include "{DEVICE_LIB_PATH}"')

    for edge_idx, (t_d, t_s) in enumerate(edge_list):
        gate_voltage = float(weights[edge_idx])
        lines.append(f".subckt e{edge_idx} t_D t_S")
        lines.append(f"V1 t_G 0 {gate_voltage:.16f}")
        lines.append(f"XNMOS t_D t_G t_S 0 {DEVICE_SUBCKT}")
        lines.append(f".ends e{edge_idx}")

    vin_node = int(node_map["vin"])
    out_node = int(node_map["out"])
    load_node = int(node_map[ground_node_name])
    sink_o = max(node_map.values()) + 1

    lines.append(f"RSO {out_node} {sink_o} 1e12")
    lines.append(f"VCLO {sink_o} 0 0")
    lines.append(f"VVIN {vin_node} 0 0")
    lines.append(f"RLOAD {load_node} 0 {float(load_res):.16f}")

    for edge_idx, (t_d, t_s) in enumerate(edge_list):
        lines.append(f"X{edge_idx} {t_d} {t_s} e{edge_idx}")

    if solver.lower() == "klu":
        lines.append(".options klu")

    lines.extend(
        [
            ".options TEMP = 27C",
            ".options TNOM = 27C",
            ".options itl1=40 itl2=40 itl4=6 itl5=60",
            ".options gmin=1e-8 reltol=5e-3 abstol=1e-8 vntol=1e-5",
            ".options rshunt=1e9",
            ".op",
            ".end",
        ]
    )
    return "\n".join(lines) + "\n"


def _evaluate_transfer(ng, vin_values: np.ndarray, out_node: int) -> np.ndarray:
    preds = np.full(vin_values.shape[0], np.nan, dtype=float)
    for i, vin in enumerate(vin_values):
        mk_free(ng)
        alter_input_single(ng, float(vin))
        ng.run()
        preds[i] = float(_require_finite("evaluation output", get_voltages(ng, [out_node]))[0])
        try:
            ng.exec_command("destroy all")
        except Exception:
            pass
    return preds


def _setup_logging(run_dir: Path):
    log_path = run_dir / "train_log.txt"
    log_f = open(log_path, "a", buffering=1)

    class _Tee:
        def __init__(self, *streams):
            self.streams = streams

        def write(self, s):
            for st in self.streams:
                try:
                    st.write(s)
                except Exception:
                    pass

        def flush(self):
            for st in self.streams:
                try:
                    st.flush()
                except Exception:
                    pass

    def _uniq(streams):
        out = []
        seen = set()
        for st in streams:
            if st is None:
                continue
            if id(st) in seen:
                continue
            seen.add(id(st))
            out.append(st)
        return out

    sys.stdout = _Tee(*_uniq([getattr(sys, "__stdout__", None), sys.stdout, log_f]))  # type: ignore
    sys.stderr = _Tee(*_uniq([getattr(sys, "__stderr__", None), sys.stderr, log_f]))  # type: ignore
    return log_f


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="4-transistor nonlinear regression from saved sweep outputs")
    p.add_argument("--sweep-run-dir", type=Path, default=DEFAULT_SWEEP_RUN)
    p.add_argument("--epochs", type=int, default=1200)
    p.add_argument("--gamma", type=float, default=0.2)
    p.add_argument("--eta", type=float, default=1.0)
    p.add_argument("--clamp-res", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--solver", type=str, choices=["klu", "sparse"], default="klu")
    p.add_argument("--ground-node", type=str, choices=["h1", "h2"], default=None)
    p.add_argument("--load-res", type=float, default=None)
    p.add_argument("--target-combo", type=int, default=-1)
    p.add_argument("--init-combo", type=int, default=-1)
    p.add_argument("--num-target-points", type=int, default=4)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--mse-stop", type=float, default=1e-6)
    return p.parse_args()


def _load_sweep_df(run_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(
        run_dir / "sweep_results.csv",
        usecols=["combo_idx", "g0", "g1", "g2", "g3", "rel_lin_rmse", "quad_gain", "curvature_rms", "is_nonlinear"],
    )
    df = df.replace([np.inf, -np.inf], np.nan).dropna().copy()
    return df


def _saved_transfer_indices(run_dir: Path) -> set[int]:
    out = set()
    for p in (run_dir / "transfers").glob("combo_*_transfer.npy"):
        try:
            out.add(int(p.stem.split("_")[1]))
        except Exception:
            continue
    return out


def _load_transfer(run_dir: Path, combo_idx: int) -> np.ndarray:
    return np.load(run_dir / "transfers" / f"combo_{int(combo_idx):06d}_transfer.npy")


def _choose_target_combo(df: pd.DataFrame, run_dir: Path) -> int:
    saved = _saved_transfer_indices(run_dir)
    cand = df[df["combo_idx"].isin(saved)].copy()
    best_idx = None
    best_score = -math.inf
    for row in cand.itertuples(index=False):
        curve = _load_transfer(run_dir, int(row.combo_idx))
        if np.any(~np.isfinite(curve)):
            continue
        if np.min(curve) < -1e-4:
            continue
        if np.any(np.diff(curve) < -1e-4):
            continue
        score = float(row.rel_lin_rmse)
        if score > best_score:
            best_score = score
            best_idx = int(row.combo_idx)
    if best_idx is None:
        best_idx = int(cand.nlargest(1, "rel_lin_rmse").iloc[0]["combo_idx"])
    return best_idx


def _choose_init_combo(df: pd.DataFrame, run_dir: Path, target_combo: int | None = None) -> int:
    saved = _saved_transfer_indices(run_dir)
    cand = df[df["combo_idx"].isin(saved)].copy()
    if target_combo is None or int(target_combo) not in cand["combo_idx"].values:
        return int(cand.nsmallest(1, "rel_lin_rmse").iloc[0]["combo_idx"])

    target_row = cand.loc[cand["combo_idx"] == int(target_combo)].iloc[0]
    cand["Gtot"] = 0.25 * (cand["g0"] + cand["g1"] + cand["g2"] + cand["g3"])
    cand["mismatch_l1"] = 0.5 * (cand["g0"] - cand["g1"]).abs() + 0.5 * (cand["g2"] - cand["g3"]).abs()
    cand["gtot_dist"] = (cand["Gtot"] - float(0.25 * (target_row["g0"] + target_row["g1"] + target_row["g2"] + target_row["g3"]))).abs()

    linear_pool = cand[cand["rel_lin_rmse"] <= 1e-3].copy()
    if linear_pool.empty:
        linear_pool = cand.nsmallest(min(500, len(cand)), "rel_lin_rmse").copy()

    close_pool = linear_pool[linear_pool["gtot_dist"] <= float(linear_pool["gtot_dist"].min()) + 1e-3].copy()
    if close_pool.empty:
        close_pool = linear_pool

    return int(
        close_pool.sort_values(["mismatch_l1", "rel_lin_rmse", "gtot_dist"], ascending=[True, True, True]).iloc[0]["combo_idx"]
    )


def _choose_train_indices(curve: np.ndarray, n_points: int) -> np.ndarray:
    n = int(curve.size)
    if n_points <= 1:
        return np.array([0], dtype=int)
    quantiles = [0.0, 0.35, 0.75, 1.0]
    if n_points == 3:
        quantiles = [0.0, 0.5, 1.0]
    elif n_points == 4:
        quantiles = [0.0, 0.35, 0.75, 1.0]
    else:
        quantiles = np.linspace(0.0, 1.0, n_points).tolist()
    y0 = float(curve.min())
    y1 = float(curve.max())
    idxs: List[int] = [0, n - 1]
    for q in quantiles[1:-1]:
        target = y0 + q * (y1 - y0)
        idxs.append(int(np.argmin(np.abs(curve - target))))
    idxs = sorted(set(idxs))
    if len(idxs) < n_points:
        candidates = np.linspace(0, n - 1, n_points, dtype=int).tolist()
        for idx in candidates:
            if idx not in idxs:
                idxs.append(idx)
            if len(idxs) == n_points:
                break
        idxs = sorted(set(idxs))
    return np.array(idxs[:n_points], dtype=int)


def _derive_space_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ybg"] = np.log10(np.clip(out["rel_lin_rmse"], 1e-12, None))
    out["Gtot"] = 0.25 * (out["g0"] + out["g1"] + out["g2"] + out["g3"])
    out["Amean"] = 0.5 * (out["g0"] + out["g1"])
    out["Bmean"] = 0.5 * (out["g2"] + out["g3"])
    out["DeltaA"] = 0.5 * (out["g0"] - out["g1"])
    out["DeltaB"] = 0.5 * (out["g2"] - out["g3"])
    out["Amin"] = np.minimum(out["g0"], out["g1"])
    out["Bmin"] = np.minimum(out["g2"], out["g3"])
    return out


def _trajectory_df(vg_history: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame(vg_history, columns=["g0", "g1", "g2", "g3"])
    df["epoch"] = np.arange(len(df), dtype=int)
    df["Gtot"] = 0.25 * (df["g0"] + df["g1"] + df["g2"] + df["g3"])
    df["Amean"] = 0.5 * (df["g0"] + df["g1"])
    df["Bmean"] = 0.5 * (df["g2"] + df["g3"])
    df["DeltaA"] = 0.5 * (df["g0"] - df["g1"])
    df["DeltaB"] = 0.5 * (df["g2"] - df["g3"])
    df["Amin"] = np.minimum(df["g0"], df["g1"])
    df["Bmin"] = np.minimum(df["g2"], df["g3"])
    return df


def _binned_heatmap(
    ax,
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    title: str,
    xlabel: str,
    ylabel: str,
    norm: mcolors.Normalize | None,
    bins: int = 64,
):
    x = df[xcol].to_numpy(dtype=float)
    y = df[ycol].to_numpy(dtype=float)
    z = df["ybg"].to_numpy(dtype=float)

    xedges = np.linspace(float(np.min(x)), float(np.max(x)), int(bins) + 1)
    yedges = np.linspace(float(np.min(y)), float(np.max(y)), int(bins) + 1)
    sum_grid, _, _ = np.histogram2d(y, x, bins=[yedges, xedges], weights=z)
    cnt_grid, _, _ = np.histogram2d(y, x, bins=[yedges, xedges])
    data = np.divide(sum_grid, cnt_grid, out=np.full_like(sum_grid, np.nan, dtype=float), where=cnt_grid > 0)

    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="white")
    mesh = ax.pcolormesh(xedges, yedges, np.ma.masked_invalid(data), cmap=cmap, norm=norm, shading="flat")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.15)
    return mesh


def _save_trajectory_figure(
    run_dir: Path,
    sweep_df: pd.DataFrame,
    traj_df: pd.DataFrame,
    mse_history: np.ndarray,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    vin_dense: np.ndarray,
    target_curve: np.ndarray,
    snapshot_epochs: Sequence[int],
    snapshot_curves: Dict[int, np.ndarray],
) -> None:
    sweep_df = _derive_space_columns(sweep_df)
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 9.0), constrained_layout=True)
    valid_bg = sweep_df["ybg"].to_numpy(dtype=float)
    bg_norm = mcolors.Normalize(
        vmin=max(float(np.quantile(valid_bg, 0.05)), -6.0),
        vmax=float(np.quantile(valid_bg, 0.98)),
    )

    ax = axes[0, 0]
    ax.plot(np.arange(len(mse_history)), np.clip(mse_history, 1e-12, None), color="black", linewidth=1.6)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_title("Log MSE")
    ax.grid(True, alpha=0.25)

    ax = axes[0, 1]
    ax.plot(vin_dense, target_curve, color="black", linewidth=2.0, linestyle="--", label="Measured target curve")
    ax.scatter(X_train, Y_train, s=60, facecolors="none", edgecolors="black", label="Train points")
    snapshot_labels = ["Initial prediction", "Mid prediction", "Final prediction"]
    snapshot_show = [int(snapshot_epochs[0]), int(snapshot_epochs[len(snapshot_epochs) // 2]), int(snapshot_epochs[-1])]
    snapshot_colors = ["#3b82f6", "#a855f7", "#f59e0b"]
    for color, label, ep in zip(snapshot_colors, snapshot_labels, snapshot_show):
        ax.plot(vin_dense, snapshot_curves[ep], color=color, linewidth=1.8, label=label)
    ax.set_xlabel("Vin (V)")
    ax.set_ylabel("Vout (V)")
    ax.set_title("Fit Evolution")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    traj_specs = [
        ("Gtot", "DeltaA", "Trajectory on (Gtot, DeltaA)", "Gtot", "DeltaA"),
        ("Amin", "Bmin", "Trajectory on (Amin, Bmin)", "Amin", "Bmin"),
        ("Amean", "Bmean", "Trajectory on (Amean, Bmean)", "Amean", "Bmean"),
        ("DeltaA", "DeltaB", "Trajectory on (DeltaA, DeltaB)", "DeltaA", "DeltaB"),
    ]

    for ax, (xcol, ycol, title, xlabel, ylabel) in zip([axes[0, 2], axes[1, 0], axes[1, 1], axes[1, 2]], traj_specs):
        mesh = _binned_heatmap(ax, sweep_df, xcol, ycol, title, xlabel, ylabel, norm=bg_norm)
        xs = traj_df[xcol].to_numpy(dtype=float)
        ys = traj_df[ycol].to_numpy(dtype=float)
        ax.plot(xs, ys, color="white", linewidth=3.0, alpha=0.95)
        ax.plot(xs, ys, color="black", linewidth=1.0, alpha=0.85)
        ax.scatter(xs[0], ys[0], marker="o", s=90, facecolors="none", edgecolors="cyan", linewidths=1.8, label="start")
        ax.scatter(xs[-1], ys[-1], marker="X", s=90, facecolors="yellow", edgecolors="black", linewidths=0.8, label="final")
        ax.legend(fontsize=8, loc="lower right")
        cbar = fig.colorbar(mesh, ax=ax, shrink=0.82)
        cbar.set_label("log10(rel_lin_rmse)")

    fig.savefig(run_dir / "trajectory_2plus4_panel.png", dpi=220)
    plt.close(fig)


def _save_pca_single_gate_sweeps(run_dir: Path, sweep_df: pd.DataFrame) -> None:
    gates = sweep_df[["g0", "g1", "g2", "g3"]].to_numpy(dtype=float)
    ybg = np.log10(np.clip(sweep_df["rel_lin_rmse"].to_numpy(dtype=float), 1e-12, None))
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(gates)
    sample_n = min(200_000, pcs.shape[0])
    rng = np.random.default_rng(0)
    if sample_n < pcs.shape[0]:
        idx = rng.choice(pcs.shape[0], size=sample_n, replace=False)
        pcs_bg = pcs[idx]
        ybg_bg = ybg[idx]
    else:
        pcs_bg = pcs
        ybg_bg = ybg

    valid = ybg[np.isfinite(ybg)]
    bg_norm = mcolors.Normalize(
        vmin=max(float(np.quantile(valid, 0.05)), -6.0),
        vmax=float(np.quantile(valid, 0.98)),
    )

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.0), constrained_layout=True)
    lo = float(np.min(gates))
    hi = float(np.max(gates))
    fixed = lo
    var_vals = np.linspace(lo, hi, 250, dtype=float)
    curve_colors = ["#ef4444", "#2563eb", "#059669", "#d97706"]
    bg_artist = None

    for gate_idx, ax in enumerate(axes.ravel()):
        bg_artist = ax.scatter(
            pcs_bg[:, 0],
            pcs_bg[:, 1],
            c=ybg_bg,
            cmap="viridis",
            norm=bg_norm,
            s=6,
            linewidths=0,
            alpha=0.22,
        )
        line_cfgs = np.full((var_vals.size, 4), fixed, dtype=float)
        line_cfgs[:, gate_idx] = var_vals
        line_pcs = pca.transform(line_cfgs)
        ax.plot(line_pcs[:, 0], line_pcs[:, 1], color=curve_colors[gate_idx], linewidth=2.5)
        ax.scatter(line_pcs[0, 0], line_pcs[0, 1], marker="o", s=70, facecolors="none", edgecolors="black", linewidths=1.2)
        ax.scatter(line_pcs[-1, 0], line_pcs[-1, 1], marker="X", s=80, facecolors=curve_colors[gate_idx], edgecolors="black", linewidths=0.8)
        ax.set_title(f"PCA path: vary g{gate_idx}, others={fixed:.1f} V")
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        ax.grid(True, alpha=0.2)

    if bg_artist is not None:
        cbar = fig.colorbar(bg_artist, ax=axes.ravel().tolist(), shrink=0.92)
        cbar.set_label("log10(rel_lin_rmse)")

    fig.savefig(run_dir / "pca_single_gate_sweeps.png", dpi=220)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    sweep_run_dir = args.sweep_run_dir.resolve()
    meta = json.loads((sweep_run_dir / "run_meta.json").read_text())

    ground_node = str(args.ground_node) if args.ground_node else str(meta.get("ground_node", "h1"))
    load_res = float(args.load_res) if args.load_res is not None else float(meta.get("load_res_ohm", 1e5))

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    sweep_df = _load_sweep_df(sweep_run_dir)
    target_combo = int(args.target_combo) if int(args.target_combo) >= 0 else _choose_target_combo(sweep_df, sweep_run_dir)
    init_combo = int(args.init_combo) if int(args.init_combo) >= 0 else _choose_init_combo(sweep_df, sweep_run_dir, target_combo)

    target_row = sweep_df.loc[sweep_df["combo_idx"] == target_combo].iloc[0]
    init_row = sweep_df.loc[sweep_df["combo_idx"] == init_combo].iloc[0]

    node_map, edge_list = _build_motif_graph()
    init_gates = init_row[["g0", "g1", "g2", "g3"]].to_numpy(dtype=float)

    target_curve = _load_transfer(sweep_run_dir, target_combo)
    vin_meta = meta.get("vin", {})
    vin_dense = np.linspace(float(vin_meta.get("vin_min", 0.0)), float(vin_meta.get("vin_max", 0.5)), int(vin_meta.get("num_points", 20)), dtype=float)
    train_idxs = _choose_train_indices(target_curve, int(args.num_target_points))
    X_train = vin_dense[train_idxs].copy()
    Y_train = target_curve[train_idxs].copy()

    netlist = mk_training_netlist(
        edge_list=edge_list,
        weights=init_gates,
        node_map=node_map,
        load_res=load_res,
        ground_node_name=ground_node,
        solver=str(args.solver).lower(),
    )

    this_dir = Path(__file__).resolve().parent
    results_root = this_dir / "results" / "runs"
    results_root.mkdir(parents=True, exist_ok=True)
    env_run_dir = os.environ.get("RUN_DIR")
    if env_run_dir:
        run_dir = Path(env_run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + f"_fourtrans_nonlin_reg_seed-{args.seed}"
        run_dir = results_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "netlist_initial.cir").write_text(netlist)
    np.save(run_dir / "target_curve.npy", np.asarray(target_curve, dtype=float))
    meta_out = {
        "script": str(Path(__file__).resolve()),
        "timestamp": datetime.now().isoformat(),
        "variant": "fourtrans_nonlin_reg_alter_ald1106",
        "seed": int(args.seed),
        "solver": str(args.solver).lower(),
        "ground_node": ground_node,
        "load_res_ohm": load_res,
        "epochs": int(args.epochs),
        "gamma": float(args.gamma),
        "eta": float(args.eta),
        "clamp_res": float(args.clamp_res),
        "mse_stop": float(args.mse_stop),
        "shuffle": bool(args.shuffle),
        "sweep_run_dir": str(sweep_run_dir),
        "init_combo": int(init_combo),
        "target_combo": int(target_combo),
        "init_gates": init_gates.tolist(),
        "target_gates": target_row[["g0", "g1", "g2", "g3"]].to_list(),
        "train_indices": train_idxs.tolist(),
        "dataset": {"vins": X_train.tolist(), "vous": Y_train.tolist()},
        "target_curve_dense": target_curve.tolist(),
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta_out, indent=2))

    log_f = _setup_logging(run_dir)
    print("=== RUN START (fourtrans_nonlin_reg_alter_ald1106) ===", flush=True)
    print(f"init_combo={init_combo} target_combo={target_combo}", flush=True)
    print(f"train_indices={train_idxs.tolist()}", flush=True)
    print(f"X_train={X_train.tolist()}", flush=True)
    print(f"Y_train={Y_train.tolist()}", flush=True)

    ng = NgSpiceShared(send_data=False)
    ng.load_circuit(netlist)

    nodes_list = np.asarray(sorted(node_map.values()), dtype=int)
    index_of = np.full(nodes_list.max() + 1, -1, dtype=int)
    index_of[nodes_list] = np.arange(nodes_list.size, dtype=int)
    e1 = np.asarray([a for (a, _) in edge_list], dtype=int)
    e2 = np.asarray([b for (_, b) in edge_list], dtype=int)
    out_node = int(node_map["out"])

    vg = init_gates.copy()
    preds0 = _evaluate_transfer(ng, X_train, out_node)
    mse0 = float(np.mean((preds0 - Y_train) ** 2))
    mse_history: List[float] = [mse0]
    preds_history: List[np.ndarray] = [preds0.copy()]
    vg_history: List[np.ndarray] = [vg.copy()]
    print(f"Epoch    0: mse={mse0:.8f}", flush=True)

    for ep in range(1, int(args.epochs) + 1):
        order = np.arange(X_train.shape[0], dtype=int)
        if args.shuffle:
            np.random.shuffle(order)

        for idx in order:
            vin = float(X_train[idx])
            target = float(Y_train[idx])

            mk_free(ng)
            alter_input_single(ng, vin)
            ng.run()

            free_out = float(_require_finite("free output", get_voltages(ng, [out_node]))[0])
            free_nodes = _require_finite("free node voltages", get_voltages(ng, nodes_list))

            try:
                ng.exec_command("destroy all")
            except Exception:
                pass

            clamped_out = float(args.eta) * target + (1.0 - float(args.eta)) * free_out
            alter_output_single(ng, clamped_out, float(args.clamp_res))
            ng.run()
            clamped_nodes = _require_finite("clamped node voltages", get_voltages(ng, nodes_list))

            try:
                ng.exec_command("destroy all")
            except Exception:
                pass

            free_e1 = free_nodes[index_of[e1]]
            free_e2 = free_nodes[index_of[e2]]
            clamped_e1 = clamped_nodes[index_of[e1]]
            clamped_e2 = clamped_nodes[index_of[e2]]
            update = -float(args.gamma) * ((clamped_e1 - clamped_e2) ** 2 - (free_e1 - free_e2) ** 2)

            if np.any(update != 0.0):
                cmds: List[str] = []
                for k, du in enumerate(update):
                    nv = float(np.clip(vg[k] + float(du), VG_CLIP_LO, VG_CLIP_HI))
                    vg[k] = nv
                    cmds.append(f"alter v.x{k}.v1 dc = {nv:.16f}")
                _exec_chunked(ng, cmds)

            mk_free(ng)

        preds_ep = _evaluate_transfer(ng, X_train, out_node)
        mse_ep = float(np.mean((preds_ep - Y_train) ** 2))
        mse_history.append(mse_ep)
        preds_history.append(preds_ep.copy())
        vg_history.append(vg.copy())

        if ep % int(args.log_every) == 0 or ep == 1:
            print(f"Epoch {ep:5d}: mse={mse_ep:.8f} gates={vg.tolist()}", flush=True)

        np.save(run_dir / "mse_history.npy", np.asarray(mse_history, dtype=float))
        np.save(run_dir / "preds_history.npy", np.asarray(preds_history, dtype=float))
        np.save(run_dir / "vg_history.npy", np.asarray(vg_history, dtype=float))
        np.save(run_dir / "final_predictions.npy", np.asarray(preds_ep, dtype=float))

        if mse_ep < float(args.mse_stop):
            print(f"Stopping early at epoch {ep} with mse={mse_ep:.8f}", flush=True)
            break

    mse_arr = np.asarray(mse_history, dtype=float)
    preds_arr = np.asarray(preds_history, dtype=float)
    vg_arr = np.asarray(vg_history, dtype=float)
    traj_df = _trajectory_df(vg_arr)

    snapshot_epochs = sorted(set([0, max(1, len(vg_arr) // 3), max(1, 2 * len(vg_arr) // 3), len(vg_arr) - 1]))
    snapshot_curves: Dict[int, np.ndarray] = {}
    for ep in snapshot_epochs:
        alter_gate_values(ng, vg_arr[ep])
        snapshot_curves[int(ep)] = _evaluate_transfer(ng, vin_dense, out_node)

    np.save(run_dir / "vin_dense.npy", vin_dense)
    np.save(run_dir / "target_curve_dense.npy", target_curve)
    np.save(run_dir / "train_vins.npy", X_train)
    np.save(run_dir / "train_targets.npy", Y_train)
    for ep, curve in snapshot_curves.items():
        np.save(run_dir / f"snapshot_curve_epoch_{int(ep):04d}.npy", curve)

    _save_trajectory_figure(
        run_dir=run_dir,
        sweep_df=sweep_df,
        traj_df=traj_df,
        mse_history=mse_arr,
        X_train=X_train,
        Y_train=Y_train,
        vin_dense=vin_dense,
        target_curve=target_curve,
        snapshot_epochs=snapshot_epochs,
        snapshot_curves=snapshot_curves,
    )
    _save_pca_single_gate_sweeps(run_dir=run_dir, sweep_df=sweep_df)

    print(f"FINAL mse={mse_arr[-1]:.8f}", flush=True)
    print(f"FINAL gates={vg_arr[-1].tolist()}", flush=True)
    print("=== RUN END (fourtrans_nonlin_reg_alter_ald1106) ===", flush=True)

    try:
        ng.remove_circuit()
    except Exception:
        pass
    try:
        ng.destroy()
    except Exception:
        pass
    try:
        log_f.flush()
        log_f.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
