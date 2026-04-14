#!/usr/bin/env python3
"""
7-edge transistor CLLN transfer-sweep script.

Topology:
    - direct edge: vin -> out
    - branch A:    vin -> h1 -> h2 -> out
    - branch B:    vin -> h3 -> h4 -> out

Task:
    - no training
    - no V+ / V- rails
    - apply Vin over a dense grid and read Vout
    - sweep every gate-voltage assignment over a user-specified grid
    - score each frozen transfer curve for linearity / nonlinearity

Notes:
    - A real finite load is needed to avoid a trivial floating solution. Here we tie
      one hidden node to ground through RLOAD, default 100 kOhm.
    - The loaded hidden node is CLI-controllable with --ground-node and defaults to h2.
    - Using a GOhm/TOhm load would effectively open the network and make the transfer
      nearly trivial.
    - The full 5^7 sweep is 78,125 gate combinations. With 20 Vin points, that is
      1,562,500 operating-point solves. Use --max-combos for smoke tests.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from PySpice.Spice.NgSpice.Shared import NgSpiceShared


REPO_ROOT = Path(__file__).resolve().parents[1]
DEVICE_LIB_PATH = str(REPO_ROOT / "device_model" / "nmos_lvl1_ald1106.lib")
DEVICE_SUBCKT = "NMOSWRAP"
VG_CLIP_LO, VG_CLIP_HI = 0.4, 8.0


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
            node_idx = int(k[2:-1])
            nodemap[node_idx] = float(v)
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


def _build_motif_graph() -> Tuple[nx.Graph, Dict[str, int], List[Tuple[int, int]]]:
    node_map = {
        "vin": 1,
        "out": 2,
        "h1": 3,
        "h2": 4,
        "h3": 5,
        "h4": 6,
    }

    edge_list = [
        (node_map["vin"], node_map["out"]),
        (node_map["vin"], node_map["h1"]),
        (node_map["h1"], node_map["h2"]),
        (node_map["h2"], node_map["out"]),
        (node_map["vin"], node_map["h3"]),
        (node_map["h3"], node_map["h4"]),
        (node_map["h4"], node_map["out"]),
    ]

    G = nx.Graph()
    G.add_nodes_from(node_map.values())
    G.add_edges_from(edge_list)
    return G, node_map, edge_list


def mk_switch_netlist(
    edge_list: List[Tuple[int, int]],
    weights: np.ndarray,
    max_node: int,
    node_map: Dict[str, int],
    load_res: float,
    ground_node_name: str,
    solver: str = "klu",
) -> str:
    weights = np.asarray(weights, dtype=float).reshape(-1)
    lines: List[str] = []
    lines.append(".title seven_edge_transfer_sweep_ald1106")
    lines.append(f'.include "{DEVICE_LIB_PATH}"')

    for edge_idx, (t_D, t_S) in enumerate(edge_list):
        gate_voltage = float(weights[edge_idx])
        lines.append(f".subckt e{edge_idx} t_D t_S")
        lines.append(f"V1 t_G 0 {gate_voltage:.16f}")
        lines.append(f"XNMOS t_D t_G t_S 0 {DEVICE_SUBCKT}")
        lines.append(f".ends e{edge_idx}")

    vin_node = int(node_map["vin"])
    load_node = int(node_map[ground_node_name])

    lines.append(f"VVIN  {vin_node} 0 0")
    lines.append(f"RLOAD {load_node} 0 {float(load_res):.16f}")

    for edge_idx, (t_D, t_S) in enumerate(edge_list):
        lines.append(f"X{edge_idx} {t_D} {t_S} e{edge_idx}")

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


def _evaluate_transfer(
    ng,
    vin_values: np.ndarray,
    out_node: int,
) -> np.ndarray:
    preds = np.full(vin_values.shape[0], np.nan, dtype=float)
    for i, vin in enumerate(vin_values):
        alter_input_single(ng, float(vin))
        ng.run()
        preds[i] = float(_require_finite("evaluation output", get_voltages(ng, [out_node]))[0])
        try:
            ng.exec_command("destroy all")
        except Exception:
            pass
    return preds


def _fit_transfer_metrics(
    x: np.ndarray,
    y: np.ndarray,
    rel_lin_rmse_thresh: float,
    quad_gain_thresh: float,
) -> Dict[str, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    lin_coeffs = np.polyfit(x, y, 1)
    y_lin = np.polyval(lin_coeffs, x)

    quad_coeffs = np.polyfit(x, y, 2)
    y_quad = np.polyval(quad_coeffs, x)

    lin_rmse = float(np.sqrt(np.mean((y - y_lin) ** 2)))
    quad_rmse = float(np.sqrt(np.mean((y - y_quad) ** 2)))
    span = float(np.ptp(y))
    rel_lin_rmse = float(lin_rmse / max(span, 1e-12))
    quad_gain = float(max(0.0, lin_rmse - quad_rmse) / max(lin_rmse, 1e-12))

    dx = float(x[1] - x[0]) if x.size >= 2 else 1.0
    d2 = np.diff(y, n=2) / (dx * dx) if x.size >= 3 else np.array([0.0])
    curvature_rms = float(np.sqrt(np.mean(d2**2))) if d2.size else 0.0
    curvature_max = float(np.max(np.abs(d2))) if d2.size else 0.0

    nonlinear_flag = float(rel_lin_rmse > rel_lin_rmse_thresh and quad_gain > quad_gain_thresh)

    return {
        "lin_slope": float(lin_coeffs[0]),
        "lin_intercept": float(lin_coeffs[1]),
        "quad_a": float(quad_coeffs[0]),
        "quad_b": float(quad_coeffs[1]),
        "quad_c": float(quad_coeffs[2]),
        "span": span,
        "lin_rmse": lin_rmse,
        "rel_lin_rmse": rel_lin_rmse,
        "quad_rmse": quad_rmse,
        "quad_gain": quad_gain,
        "curvature_rms": curvature_rms,
        "curvature_max": curvature_max,
        "is_nonlinear": nonlinear_flag,
    }


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

    def uniq(streams):
        out = []
        seen = set()
        for st in streams:
            if st is None:
                continue
            k = id(st)
            if k in seen:
                continue
            seen.add(k)
            out.append(st)
        return out

    out_streams = []
    err_streams = []

    try:
        if getattr(sys, "__stdout__", None):
            out_streams.append(sys.__stdout__)
    except Exception:
        pass
    out_streams.append(sys.stdout)
    out_streams.append(log_f)

    try:
        if getattr(sys, "__stderr__", None):
            err_streams.append(sys.__stderr__)
    except Exception:
        pass
    err_streams.append(sys.stderr)
    err_streams.append(log_f)

    sys.stdout = _Tee(*uniq(out_streams))  # type: ignore
    sys.stderr = _Tee(*uniq(err_streams))  # type: ignore
    return log_f


def _save_summary_plots(
    run_dir: Path,
    rel_lin_rmse_arr: np.ndarray,
    quad_gain_arr: np.ndarray,
    curvature_arr: np.ndarray,
    nonlinear_arr: np.ndarray,
    vin_values: np.ndarray,
    best_linear_curve: np.ndarray,
    best_linear_gates: Sequence[float],
    best_nonlinear_curve: np.ndarray,
    best_nonlinear_gates: Sequence[float],
) -> None:
    plt.figure()
    plt.hist(rel_lin_rmse_arr, bins=50)
    plt.xlabel("Relative linear-fit RMSE")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(run_dir / "hist_rel_lin_rmse.png", dpi=300)
    plt.close()

    plt.figure()
    plt.hist(curvature_arr, bins=50)
    plt.xlabel("Curvature RMS (V / V^2)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(run_dir / "hist_curvature_rms.png", dpi=300)
    plt.close()

    plt.figure()
    mask_nl = nonlinear_arr > 0.5
    if np.any(~mask_nl):
        plt.scatter(rel_lin_rmse_arr[~mask_nl], quad_gain_arr[~mask_nl], s=8, alpha=0.6, label="Linear")
    if np.any(mask_nl):
        plt.scatter(rel_lin_rmse_arr[mask_nl], quad_gain_arr[mask_nl], s=8, alpha=0.6, label="Nonlinear")
    plt.xlabel("Relative linear-fit RMSE")
    plt.ylabel("Quadratic gain")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "scatter_rel_rmse_vs_quad_gain.png", dpi=300)
    plt.close()

    plt.figure()
    plt.plot(vin_values, best_linear_curve, label="Most linear")
    plt.plot(vin_values, best_nonlinear_curve, label="Most nonlinear")
    plt.xlabel("Input Voltage (V)")
    plt.ylabel("Output Voltage (V)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "example_transfer_curves.png", dpi=300)
    plt.close()

    summary = {
        "best_linear_gates": [float(v) for v in best_linear_gates],
        "best_nonlinear_gates": [float(v) for v in best_nonlinear_gates],
    }
    (run_dir / "example_gate_configs.json").write_text(json.dumps(summary, indent=2))


def _parse_gate_values(s: str) -> List[float]:
    vals = [float(x.strip()) for x in s.split(",") if x.strip()]
    if not vals:
        raise ValueError("--vg-values produced an empty gate-value list")
    return vals


def _parse_args():
    p = argparse.ArgumentParser(description="7-edge transistor CLLN transfer sweep")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--solver", type=str, choices=["klu", "sparse"], default="klu")
    p.add_argument("--load-res", type=float, default=1e5, help="Finite resistor from one hidden node to ground, in ohms")
    p.add_argument("--ground-node", type=str, choices=["h1", "h2", "h3", "h4"], default="h2")
    p.add_argument("--vin-min", type=float, default=0.0)
    p.add_argument("--vin-max", type=float, default=0.5)
    p.add_argument("--num-points", type=int, default=20)
    p.add_argument("--vg-values", type=str, default="1,2,3,4,5")
    p.add_argument("--rel-lin-rmse-thresh", type=float, default=0.02)
    p.add_argument("--quad-gain-thresh", type=float, default=0.20)
    p.add_argument("--max-combos", type=int, default=-1, help="Limit number of gate combinations for testing")
    p.add_argument("--log-every", type=int, default=100)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    gate_grid = _parse_gate_values(str(args.vg_values))
    vin_values = np.linspace(float(args.vin_min), float(args.vin_max), int(args.num_points), dtype=float)

    G, node_map, edge_list = _build_motif_graph()
    n_edges = len(edge_list)
    vg0 = np.full(n_edges, float(gate_grid[0]), dtype=float)

    max_node = int(max(G.nodes()))
    netlist = mk_switch_netlist(
        edge_list=edge_list,
        weights=vg0,
        max_node=max_node,
        node_map=node_map,
        load_res=float(args.load_res),
        ground_node_name=str(args.ground_node),
        solver=str(args.solver).lower(),
    )

    this_dir = Path(__file__).resolve().parent
    results_root = this_dir / "results"
    results_root.mkdir(parents=True, exist_ok=True)

    env_run_dir = os.environ.get("RUN_DIR")
    if env_run_dir:
        run_dir = Path(env_run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        runs_dir = results_root / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + f"_seven_edge_transfer_seed-{args.seed}"
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "netlist_initial.cir").write_text(netlist)
    meta = {
        "script": str(Path(__file__).resolve()),
        "script_name": Path(__file__).name,
        "argv": list(os.sys.argv),
        "timestamp": datetime.now().isoformat(),
        "variant": "seven_edge_transfer_sweep_ald1106",
        "backend": "shared",
        "solver": str(args.solver).lower(),
        "seed": int(args.seed),
        "topology": {
            "node_map": node_map,
            "edges": edge_list,
        },
        "vin": {
            "vin_min": float(args.vin_min),
            "vin_max": float(args.vin_max),
            "num_points": int(args.num_points),
        },
        "gate_grid": gate_grid,
        "load_res_ohm": float(args.load_res),
        "ground_node": str(args.ground_node),
        "classification": {
            "rel_lin_rmse_thresh": float(args.rel_lin_rmse_thresh),
            "quad_gain_thresh": float(args.quad_gain_thresh),
        },
        "max_combos": int(args.max_combos),
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

    log_f = _setup_logging(run_dir)
    print("=== RUN START (seven_edge_transfer_sweep_ald1106) ===", flush=True)
    print(f"Gate grid: {gate_grid}", flush=True)
    print(f"Hidden-node load resistor: {float(args.load_res):.6g} ohm", flush=True)
    print(f"Loaded hidden node: {str(args.ground_node)}", flush=True)

    ng = NgSpiceShared(send_data=False)
    ng.load_circuit(netlist)

    out_node = int(node_map["out"])
    total_possible = int(len(gate_grid) ** n_edges)
    if int(args.max_combos) > 0:
        total_to_run = min(total_possible, int(args.max_combos))
    else:
        total_to_run = total_possible

    csv_path = run_dir / "sweep_results.csv"
    transfer_dir = run_dir / "transfers"
    transfer_dir.mkdir(parents=True, exist_ok=True)

    rel_lin_rmse_vals: List[float] = []
    quad_gain_vals: List[float] = []
    curvature_vals: List[float] = []
    nonlinear_flags: List[float] = []

    best_linear_score = math.inf
    best_linear_curve = np.full_like(vin_values, np.nan)
    best_linear_gates: Sequence[float] = []

    best_nonlinear_score = -math.inf
    best_nonlinear_curve = np.full_like(vin_values, np.nan)
    best_nonlinear_gates: Sequence[float] = []

    combo_iter = itertools.product(gate_grid, repeat=n_edges)

    with open(csv_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(
            [
                "combo_idx",
                *[f"g{i}" for i in range(n_edges)],
                "lin_slope",
                "lin_intercept",
                "quad_a",
                "quad_b",
                "quad_c",
                "span",
                "lin_rmse",
                "rel_lin_rmse",
                "quad_rmse",
                "quad_gain",
                "curvature_rms",
                "curvature_max",
                "is_nonlinear",
            ]
        )

        for combo_idx, gates in enumerate(combo_iter):
            if combo_idx >= total_to_run:
                break

            alter_gate_values(ng, gates)
            preds = _evaluate_transfer(ng, vin_values, out_node)
            metrics = _fit_transfer_metrics(
                vin_values,
                preds,
                rel_lin_rmse_thresh=float(args.rel_lin_rmse_thresh),
                quad_gain_thresh=float(args.quad_gain_thresh),
            )

            writer.writerow(
                [
                    combo_idx,
                    *[float(v) for v in gates],
                    metrics["lin_slope"],
                    metrics["lin_intercept"],
                    metrics["quad_a"],
                    metrics["quad_b"],
                    metrics["quad_c"],
                    metrics["span"],
                    metrics["lin_rmse"],
                    metrics["rel_lin_rmse"],
                    metrics["quad_rmse"],
                    metrics["quad_gain"],
                    metrics["curvature_rms"],
                    metrics["curvature_max"],
                    metrics["is_nonlinear"],
                ]
            )

            if combo_idx < 25 or combo_idx % max(1, int(args.log_every)) == 0:
                np.save(transfer_dir / f"combo_{combo_idx:06d}_gates.npy", np.asarray(gates, dtype=float))
                np.save(transfer_dir / f"combo_{combo_idx:06d}_transfer.npy", preds)

            rel_lin_rmse_vals.append(metrics["rel_lin_rmse"])
            quad_gain_vals.append(metrics["quad_gain"])
            curvature_vals.append(metrics["curvature_rms"])
            nonlinear_flags.append(metrics["is_nonlinear"])

            linear_score = metrics["rel_lin_rmse"]
            if linear_score < best_linear_score:
                best_linear_score = linear_score
                best_linear_curve = preds.copy()
                best_linear_gates = tuple(float(v) for v in gates)

            nonlinear_score = metrics["rel_lin_rmse"] * (1.0 + metrics["quad_gain"])
            if nonlinear_score > best_nonlinear_score:
                best_nonlinear_score = nonlinear_score
                best_nonlinear_curve = preds.copy()
                best_nonlinear_gates = tuple(float(v) for v in gates)

            if combo_idx % int(args.log_every) == 0 or combo_idx == total_to_run - 1:
                done = combo_idx + 1
                frac = 100.0 * done / max(total_to_run, 1)
                nonlinear_count = int(sum(1 for x in nonlinear_flags if x > 0.5))
                print(
                    f"Combo {done:7d}/{total_to_run:7d} ({frac:6.2f}%)  "
                    f"nonlinear_count={nonlinear_count:7d}",
                    flush=True,
                )

    rel_lin_rmse_arr = np.asarray(rel_lin_rmse_vals, dtype=float)
    quad_gain_arr = np.asarray(quad_gain_vals, dtype=float)
    curvature_arr = np.asarray(curvature_vals, dtype=float)
    nonlinear_arr = np.asarray(nonlinear_flags, dtype=float)

    np.save(run_dir / "vin_values.npy", vin_values)
    np.save(run_dir / "rel_lin_rmse.npy", rel_lin_rmse_arr)
    np.save(run_dir / "quad_gain.npy", quad_gain_arr)
    np.save(run_dir / "curvature_rms.npy", curvature_arr)
    np.save(run_dir / "is_nonlinear.npy", nonlinear_arr)
    np.save(run_dir / "best_linear_curve.npy", best_linear_curve)
    np.save(run_dir / "best_nonlinear_curve.npy", best_nonlinear_curve)

    _save_summary_plots(
        run_dir=run_dir,
        rel_lin_rmse_arr=rel_lin_rmse_arr,
        quad_gain_arr=quad_gain_arr,
        curvature_arr=curvature_arr,
        nonlinear_arr=nonlinear_arr,
        vin_values=vin_values,
        best_linear_curve=best_linear_curve,
        best_linear_gates=best_linear_gates,
        best_nonlinear_curve=best_nonlinear_curve,
        best_nonlinear_gates=best_nonlinear_gates,
    )

    total_run = rel_lin_rmse_arr.size
    nonlinear_count = int(np.sum(nonlinear_arr > 0.5))
    print(f"TOTAL combos run = {total_run}", flush=True)
    print(f"NONLINEAR combos = {nonlinear_count}", flush=True)
    print(f"FRACTION nonlinear = {nonlinear_count / max(total_run, 1):.6f}", flush=True)
    print(f"BEST linear gates = {best_linear_gates}", flush=True)
    print(f"BEST nonlinear gates = {best_nonlinear_gates}", flush=True)
    print("=== RUN END (seven_edge_transfer_sweep_ald1106) ===", flush=True)

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
