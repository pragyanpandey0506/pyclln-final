
#!/usr/bin/env python3
"""
Alter-based 4x4 nonlinear-regression trainer using the PyCLLN-style shared
NgSpice backend and the same external device include pattern as the XOR script.

Task:
  - Single-ended nonlinear regression on the 8-point PNAS-style curve.
  - 4x4 periodic grid.
  - One variable input V1, one measured output O.
  - Fixed rails V- = 0.0 V and V+ = 0.45 V.
  - Single-output nudge/clamp:
        O_C = eta * L + (1 - eta) * O_F

Notes:
  - The PNAS 2024 nonlinear-regression setup uses 8 datapoints, ordered cycling,
    and eta = 1 for regression tasks.
  - This script defaults to that regression clamp convention, but keeps CLI
    switches so you can also run older/baseline-style settings if desired.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from PySpice.Spice.NgSpice.Shared import NgSpiceShared


REPO_ROOT = Path(__file__).resolve().parents[1]
DEVICE_LIB_PATH = str(REPO_ROOT / "device_model" / "nmos_lvl1_ald1106.lib")
DEVICE_SUBCKT = "NMOSWRAP"
VG_CLIP_LO, VG_CLIP_HI = 0.4, 8.0

VINS = np.array([0.0005, 0.0628, 0.1251, 0.1873, 0.2479, 0.3118, 0.3724, 0.4347], dtype=float)
VOUS = np.array([0.1088, 0.1472, 0.1872, 0.2256, 0.2649, 0.3041, 0.3041, 0.3041], dtype=float)


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


def mk_free(ng) -> None:
    _exec_chunked(ng, ["alter RSO 1e12"])


def alter_output_single(ng, oc: float, clamp_res: float) -> None:
    cmds = [
        f"alter RSO {float(clamp_res):.16f}",
        f"alter VCLO dc = {float(oc):.16f}",
    ]
    _exec_chunked(ng, cmds)


def _build_training_graph(side: int) -> Tuple[nx.Graph, Dict[str, int], List[Tuple[int, int]]]:
    if side < 3:
        raise ValueError("_build_training_graph requires side >= 3")

    def gidx(i: int, j: int) -> int:
        return 1 + i * side + j

    G = nx.Graph()
    for i in range(side):
        for j in range(side):
            G.add_node(gidx(i, j))

    for i in range(side):
        for j in range(side):
            u = gidx(i, j)
            v_right = gidx(i, (j + 1) % side)
            v_down = gidx((i + 1) % side, j)
            G.add_edge(u, v_right)
            G.add_edge(u, v_down)

    edge_list = list(G.edges())

    node_map: Dict[str, int] = {}
    if side >= 4:
        node_map = {
            "out": gidx(side // 2, side // 2),  # center-like node; for side=4 => node 11
            "vminus": gidx(1, 1),               # node 6
            "vplus": gidx(1, side - 1),         # node 8
            "vin": gidx(side - 1, side - 1),    # node 16
        }

    return G, node_map, edge_list


def mk_switch_netlist(
    edge_list: List[Tuple[int, int]],
    weights: np.ndarray,
    max_node: int,
    node_map: Dict[str, int],
    vminus_val: float,
    vplus_val: float,
    solver: str = "klu",
) -> str:
    weights = np.asarray(weights, dtype=float).reshape(-1)
    lines: List[str] = []
    lines.append(".title nonlin_reg_4x4_alter_ald1106_single_output")
    lines.append(f'.include "{DEVICE_LIB_PATH}"')

    for edge_idx, (t_D, t_S) in enumerate(edge_list):
        gate_voltage = float(weights[edge_idx])
        lines.append(f".subckt e{edge_idx} t_D t_S")
        lines.append(f"V1 t_G 0 {gate_voltage:.16f}")
        lines.append(f"XNMOS t_D t_G t_S 0 {DEVICE_SUBCKT}")
        lines.append(f".ends e{edge_idx}")

    out_node = int(node_map["out"])
    vminus = int(node_map["vminus"])
    vplus = int(node_map["vplus"])
    vin_node = int(node_map["vin"])

    sink_o = max_node + 1

    lines.append(f"RSO {out_node} {sink_o} 1e12")
    lines.append(f"VCLO {sink_o} 0 0")

    lines.append(f"VMINUS {vminus} 0 {float(vminus_val):.16f}")
    lines.append(f"VPLUS  {vplus} 0 {float(vplus_val):.16f}")
    lines.append(f"VVIN   {vin_node} 0 0")

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


def _evaluate_regression(
    ng,
    X: np.ndarray,
    Y: np.ndarray,
    out_node: int,
) -> Tuple[np.ndarray, float]:
    preds = np.full(X.shape[0], np.nan, dtype=float)
    for i in range(X.shape[0]):
        mk_free(ng)
        alter_input_single(ng, float(X[i]))
        ng.run()
        preds[i] = float(_require_finite("evaluation output", get_voltages(ng, [out_node]))[0])
    mse = float(np.mean((preds - Y.reshape(-1)) ** 2))
    return preds, mse


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


def _save_plots(
    run_dir: Path,
    X: np.ndarray,
    Y: np.ndarray,
    mse_hist: np.ndarray,
    preds_hist: np.ndarray,
    vg_history: np.ndarray,
) -> None:
    plt.figure()
    plt.plot(np.arange(mse_hist.size), mse_hist)
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.tight_layout()
    plt.savefig(run_dir / "mse_curve.png", dpi=300)
    plt.close()

    plt.figure()
    plt.scatter(X, Y, s=60, facecolors="none", edgecolors="black", label="Train data")
    if preds_hist.shape[0] >= 1:
        plt.plot(X, preds_hist[0], label="Epoch 0")
    if preds_hist.shape[0] >= 2:
        plt.plot(X, preds_hist[max(1, preds_hist.shape[0] // 3)], label="~1/3")
        plt.plot(X, preds_hist[max(1, 2 * preds_hist.shape[0] // 3)], label="~2/3")
    plt.plot(X, preds_hist[-1], label="Final")
    plt.xlabel("Input Voltage (V)")
    plt.ylabel("Output Voltage (V)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "vout_vs_vin.png", dpi=300)
    plt.close()

    plt.figure()
    for k in range(vg_history.shape[1]):
        plt.plot(vg_history[:, k], linewidth=0.8)
    plt.xlabel("Epoch")
    plt.ylabel("Gate Voltage (V)")
    plt.tight_layout()
    plt.savefig(run_dir / "vg_history.png", dpi=300)
    plt.close()


def _parse_args():
    p = argparse.ArgumentParser(
        description="4x4 nonlinear-regression alter-based trainer (PyCLLN-style shared backend)"
    )
    p.add_argument("--side", type=int, default=4)
    p.add_argument("--epochs", type=int, default=30000)
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--eta", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--solver", type=str, choices=["klu", "sparse"], default="klu")

    p.add_argument("--vminus", type=float, default=0.0)
    p.add_argument("--vplus", type=float, default=0.45)
    p.add_argument("--clamp-res", type=float, default=1.0)
    p.add_argument("--mse-stop", type=float, default=1e-5)

    p.add_argument("--shuffle", action="store_true", help="Shuffle the 8 regression points each epoch")
    p.add_argument("--log-every", type=int, default=100)

    p.add_argument("--init", type=str, choices=["uniform", "constant"], default="constant")
    p.add_argument("--vg-init-lo", type=float, default=0.5)
    p.add_argument("--vg-init-hi", type=float, default=3.0)
    p.add_argument("--vg-init-value", type=float, default=1.5)

    p.add_argument("--out-node", type=int, default=-1)
    p.add_argument("--vminus-node", type=int, default=-1)
    p.add_argument("--vplus-node", type=int, default=-1)
    p.add_argument("--vin-node", type=int, default=-1)

    return p.parse_args()


def main() -> None:
    args = _parse_args()

    side = int(args.side)
    if side < 4:
        raise ValueError("The 4x4-style regression validation requires --side >= 4")

    epochs = int(args.epochs)
    eta = float(args.eta)
    gamma = float(args.gamma) if args.gamma is not None else 0.1 * (1.0 / eta)
    seed = int(args.seed)
    solver = str(args.solver).lower()

    random.seed(seed)
    np.random.seed(seed)

    X = VINS.copy()
    Y = VOUS.copy()

    G, node_map_default, edge_list = _build_training_graph(side)
    node_map = dict(node_map_default)

    if args.out_node > 0:
        node_map["out"] = int(args.out_node)
    if args.vminus_node > 0:
        node_map["vminus"] = int(args.vminus_node)
    if args.vplus_node > 0:
        node_map["vplus"] = int(args.vplus_node)
    if args.vin_node > 0:
        node_map["vin"] = int(args.vin_node)

    for k, n in node_map.items():
        if int(n) not in G.nodes():
            raise ValueError(f"node_map[{k}]={n} is not a valid grid node for side={side}")

    n_edges = len(edge_list)
    if args.init == "uniform":
        vg = np.random.uniform(float(args.vg_init_lo), float(args.vg_init_hi), size=n_edges).astype(float)
    else:
        vg = np.full(n_edges, float(args.vg_init_value), dtype=float)

    vg = np.clip(vg, VG_CLIP_LO, VG_CLIP_HI)
    for (u, v), w in zip(edge_list, vg):
        G[u][v]["weight"] = float(w)

    max_node = int(max(G.nodes()))
    netlist = mk_switch_netlist(
        edge_list=edge_list,
        weights=vg,
        max_node=max_node,
        node_map=node_map,
        vminus_val=float(args.vminus),
        vplus_val=float(args.vplus),
        solver=solver,
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
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + f"_nonlin_reg_side-{side}_seed-{seed}"
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "netlist_initial.cir").write_text(netlist)
    meta = {
        "script": str(Path(__file__).resolve()),
        "script_name": Path(__file__).name,
        "argv": list(os.sys.argv),
        "timestamp": datetime.now().isoformat(),
        "variant": "nonlin_reg_4x4_alter_ald1106_single_output",
        "backend": "shared",
        "solver": solver,
        "seed": seed,
        "grid_side": side,
        "epochs": epochs,
        "gamma": gamma,
        "eta": eta,
        "dataset": {"vins": X.tolist(), "vous": Y.tolist()},
        "rails": {"vminus": float(args.vminus), "vplus": float(args.vplus)},
        "node_map": node_map,
        "init": {
            "kind": args.init,
            "vg_init_lo": float(args.vg_init_lo),
            "vg_init_hi": float(args.vg_init_hi),
            "vg_init_value": float(args.vg_init_value),
        },
        "shuffle": bool(args.shuffle),
        "clamp_res": float(args.clamp_res),
        "mse_stop": float(args.mse_stop),
    }
    try:
        (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    except Exception:
        pass

    log_f = _setup_logging(run_dir)
    print("=== RUN START (nonlin_reg_4x4_alter_ald1106_single_output) ===", flush=True)

    ng = NgSpiceShared(send_data=False)
    ng.load_circuit(netlist)

    nodes_list = np.asarray(sorted(G.nodes()), dtype=int)
    index_of = np.full(nodes_list.max() + 1, -1, dtype=int)
    index_of[nodes_list] = np.arange(nodes_list.size, dtype=int)
    e1 = np.asarray([a for (a, _) in edge_list], dtype=int)
    e2 = np.asarray([b for (_, b) in edge_list], dtype=int)

    out_node = int(node_map["out"])

    preds0, mse0 = _evaluate_regression(ng, X, Y, out_node)
    mse_hist: List[float] = [mse0]
    preds_hist: List[np.ndarray] = [preds0.copy()]
    vg_history: List[np.ndarray] = [vg.copy()]

    print(f"Epoch    0: mse={mse0:.8f}", flush=True)

    for ep in range(1, epochs + 1):
        order = np.arange(X.shape[0], dtype=int)
        if args.shuffle:
            np.random.shuffle(order)

        for idx in order:
            vin = float(X[idx])
            target = float(Y[idx])

            mk_free(ng)
            alter_input_single(ng, vin)
            ng.run()

            free_out = float(_require_finite("free output", get_voltages(ng, [out_node]))[0])
            free_nodes = _require_finite("free node voltages", get_voltages(ng, nodes_list))

            try:
                ng.exec_command("destroy all")
            except Exception:
                pass

            clamped_out = eta * target + (1.0 - eta) * free_out
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

            free_diffs = free_e1 - free_e2
            clamped_diffs = clamped_e1 - clamped_e2
            update = -gamma * (clamped_diffs ** 2 - free_diffs ** 2)

            if np.any(update != 0.0):
                cmds: List[str] = []
                for k, du in enumerate(update):
                    nv = vg[k] + float(du)
                    if nv < VG_CLIP_LO:
                        nv = VG_CLIP_LO
                    elif nv > VG_CLIP_HI:
                        nv = VG_CLIP_HI
                    vg[k] = nv
                    u, v = edge_list[k]
                    G[u][v]["weight"] = float(nv)
                    cmds.append(f"alter v.x{k}.v1 dc = {nv:.16f}")
                _exec_chunked(ng, cmds)

            mk_free(ng)

        preds_ep, mse_ep = _evaluate_regression(ng, X, Y, out_node)
        mse_hist.append(float(mse_ep))
        preds_hist.append(preds_ep.copy())
        vg_history.append(vg.copy())

        if ep % int(args.log_every) == 0 or ep == 1:
            print(f"Epoch {ep:5d}: mse={mse_ep:.8f}", flush=True)

        try:
            np.save(run_dir / "mse_history.npy", np.asarray(mse_hist, dtype=float))
            np.save(run_dir / "preds_history.npy", np.asarray(preds_hist, dtype=float))
            np.save(run_dir / "vg_history.npy", np.asarray(vg_history, dtype=float))
            np.save(run_dir / "final_predictions.npy", np.asarray(preds_ep, dtype=float))
        except Exception:
            pass

        if mse_ep < float(args.mse_stop):
            print(f"Stopping early at epoch {ep} with mse={mse_ep:.8f}", flush=True)
            break

    mse_arr = np.asarray(mse_hist, dtype=float)
    preds_arr = np.asarray(preds_hist, dtype=float)
    vg_arr = np.asarray(vg_history, dtype=float)

    try:
        np.save(run_dir / "final_predictions.npy", preds_arr[-1])
    except Exception:
        pass

    _save_plots(run_dir, X, Y, mse_arr, preds_arr, vg_arr)

    try:
        nx.write_graphml(G, run_dir / "final_graph.graphml")
    except Exception as e:
        print(f"[WARN] failed to save final_graph.graphml: {e}", flush=True)

    print(f"FINAL mse={mse_arr[-1]:.8f}", flush=True)
    print("=== RUN END (nonlin_reg_4x4_alter_ald1106_single_output) ===", flush=True)

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
