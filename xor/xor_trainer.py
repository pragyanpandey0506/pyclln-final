#!/usr/bin/env python3
"""
Alter-based XOR N×N trainer using an NgSpice shared backend,
using the same device include as scikit_digit dense trainer.

PNAS-aligned (Figure 3A) boundary placement + differential output scheme:
  - Differential output: O = O+ - O-
  - Rails: V- = 0.11 V, V+ = 0.33 V (on TWO grid nodes)
  - Inputs: V1, V2 are two variable sources with values in {0, Vmax}
  - Differential nudge/clamp (Eq. 7):
        O_C^± = O^± ± (eta/2)*(L - O)

Figure 3A node placement (for side=4 exactly; generalized for side>=4):
  - O-   at (0,0)
  - V-   at (1,1)
  - V+   at (1,side-1)
  - O+   at (side//2, side//2)   (=> (2,2) when side=4)
  - V2   at (side-1,1)
  - V1   at (side-1,side-1)

Device:
  - Uses external include file defining:
      .model ncg nmos (...)
      .subckt NMOSWRAP D G S B
  - Body/bulk is tied to global ground (0), NOT source.

Run (from repo root):
  python xor/xor_trainer.py --side 4 --epochs 10 --solver klu
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

import networkx as nx
import numpy as np
from PySpice.Spice.NgSpice.Shared import NgSpiceShared


# ---------------------------------------------------------------------------
# Device include + VG limits (match scikit_digit dense trainer)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DEVICE_LIB_PATH = str(REPO_ROOT / "device_model" / "nmos_lvl1_ald1106.lib")
DEVICE_SUBCKT = "NMOSWRAP"
VG_CLIP_LO, VG_CLIP_HI = 0.4, 8.0


# ---------------------------------------------------------------------------
# PNAS XOR dataset (two variable inputs V1,V2; label in {0, L0})
# ---------------------------------------------------------------------------

def xor_dataset_pnas(Vmax: float = 0.45, L0: float = -0.087) -> Tuple[np.ndarray, np.ndarray]:
    """
    (V1, V2) in {0, Vmax}^2 with XOR labels {0, L0}.
      00 -> 0
      01 -> L0
      10 -> L0
      11 -> 0
    """
    X = np.array(
        [
            [0.0, 0.0],
            [0.0, Vmax],
            [Vmax, 0.0],
            [Vmax, Vmax],
        ],
        dtype=float,
    )
    Y = np.array([0.0, L0, L0, 0.0], dtype=float).reshape(-1, 1)
    return X, Y


def accuracy_nearest_target(preds: np.ndarray, targets: np.ndarray, L0: float) -> float:
    p = np.asarray(preds, dtype=float).reshape(-1)
    t = np.asarray(targets, dtype=float).reshape(-1)
    d0 = (p - 0.0) ** 2
    d1 = (p - L0) ** 2
    pred_bits = np.where(d1 < d0, 1, 0)
    tol = 1e-6
    true_bits = np.where(np.abs(t - L0) <= tol, 1, 0)
    if true_bits.size == 0:
        return float("nan")
    return float(np.mean(pred_bits == true_bits))


# ---------------------------------------------------------------------------
# Ngspice helper utilities (alter-based)
# ---------------------------------------------------------------------------

def _exec_chunked(ng, cmds: Iterable[str], max_len: int = 900, sep: str = "; ") -> None:
    """
    Execute a list of SPICE commands in chunks so each exec_command call stays short.
    Use '; ' separator (NOT newlines) to keep ngspice happy.
    """
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
    """
    Query Ngspice for voltages using 'print allv' and return V(node)
    for the requested numeric nodes.
    """
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


def alter_inputs_two(ng, v1: float, v2: float) -> None:
    """
    Alter the two input sources VV1 and VV2.
    """
    cmds = [
        f"alter VV1 dc = {float(v1):.16f}",
        f"alter VV2 dc = {float(v2):.16f}",
    ]
    _exec_chunked(ng, cmds)


def mk_free(ng) -> None:
    """
    Free both outputs by setting RS1/RS2 huge.
    """
    _exec_chunked(ng, ["alter RS1 1e12", "alter RS2 1e12"])


def alter_outputs_diff(ng, oc_plus: float, oc_minus: float) -> None:
    """
    Clamp outputs by lowering RS1/RS2 to ~1Ω and setting VCLP/VCLM.
    """
    cmds = [
        "alter RS1 1.0",
        "alter RS2 1.0",
        f"alter VCLP dc = {float(oc_plus):.16f}",
        f"alter VCLM dc = {float(oc_minus):.16f}",
    ]
    _exec_chunked(ng, cmds)


# ---------------------------------------------------------------------------
# Training graph + netlist builder
# ---------------------------------------------------------------------------

def _build_training_graph(side: int) -> Tuple[nx.Graph, Dict[str, int], List[Tuple[int, int]]]:
    """
    Build an N×N periodic grid on nodes 1..N^2 (no disconnected extra nodes).

    Figure 3A placement (generalized for side>=4):
      O-   : (0,0)
      V-   : (1,1)
      V+   : (1,side-1)
      O+   : (side//2, side//2)
      V2   : (side-1,1)
      V1   : (side-1,side-1)
    """
    if side < 3:
        raise ValueError("_build_training_graph requires side >= 3")

    def gidx(i: int, j: int) -> int:
        return 1 + i * side + j

    G = nx.Graph()
    for i in range(side):
        for j in range(side):
            G.add_node(gidx(i, j))

    # periodic grid edges (torus)
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
            "ominus": gidx(0, 0),                  # O-
            "vminus": gidx(1, 1),                  # V-
            "vplus":  gidx(1, side - 1),           # V+
            "oplus":  gidx(side // 2, side // 2),  # O+
            "v2":     gidx(side - 1, 1),           # V2
            "v1":     gidx(side - 1, side - 1),    # V1
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
    """
    Build a transistor-level SPICE netlist using external NMOSWRAP include.

    - Body/bulk tied to global 0
    - V- and V+ are fixed sources on grid nodes
    - VV1, VV2 are the two variable input sources on grid nodes
    - Differential output clamp via RS1/RS2 to sink nodes + VCLP/VCLM sources
    """
    weights = np.asarray(weights, dtype=float).reshape(-1)
    lines: List[str] = []
    lines.append(".title xor_nxn_alter_ald1106_diff_output_pnas_fig3A")

    lines.append(f'.include "{DEVICE_LIB_PATH}"')

    # Per-edge wrapper: exposes (t_D, t_S); body tied to 0; gate set by internal V1
    for edge_idx, (t_D, t_S) in enumerate(edge_list):
        gate_voltage = float(weights[edge_idx])
        lines.append(f".subckt e{edge_idx} t_D t_S")
        lines.append(f"V1 t_G 0 {gate_voltage:.16f}")
        lines.append(f"XNMOS t_D t_G t_S 0 {DEVICE_SUBCKT}")
        lines.append(f".ends e{edge_idx}")

    # Node map (Figure 3A)
    oplus = int(node_map["oplus"])
    ominus = int(node_map["ominus"])
    vminus = int(node_map["vminus"])
    vplus = int(node_map["vplus"])
    v1n = int(node_map["v1"])
    v2n = int(node_map["v2"])

    # Output clamp sink nodes (unique and > max_node)
    sink_p = max_node + 1
    sink_m = max_node + 2

    # Output clamp resistors (alter RS1/RS2)
    lines.append(f"RS1 {oplus} {sink_p} 1e12")
    lines.append(f"RS2 {ominus} {sink_m} 1e12")

    # Clamp voltage sources (alter VCLP/VCLM)
    lines.append(f"VCLP {sink_p} 0 0")
    lines.append(f"VCLM {sink_m} 0 0")

    # Fixed rails on grid nodes
    lines.append(f"VMINUS {vminus} 0 {float(vminus_val):.16f}")
    lines.append(f"VPLUS  {vplus}  0 {float(vplus_val):.16f}")

    # Variable inputs on grid nodes
    lines.append(f"VV1 {v1n} 0 0")
    lines.append(f"VV2 {v2n} 0 0")

    # Instantiate edge wrappers
    for edge_idx, (t_D, t_S) in enumerate(edge_list):
        lines.append(f"X{edge_idx} {t_D} {t_S} e{edge_idx}")

    # Options
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


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def _evaluate_xor(ng, X: np.ndarray, Y: np.ndarray, node_map: Dict[str, int], L0: float) -> float:
    oplus = int(node_map["oplus"])
    ominus = int(node_map["ominus"])
    preds: List[float] = []

    for i in range(X.shape[0]):
        mk_free(ng)
        alter_inputs_two(ng, float(X[i, 0]), float(X[i, 1]))
        ng.run()
        v = get_voltages(ng, [oplus, ominus])
        O = float(v[0] - v[1])
        preds.append(O)

    return float(accuracy_nearest_target(np.array(preds).reshape(-1, 1), Y, L0))


# ---------------------------------------------------------------------------
# Logging + CLI + main
# ---------------------------------------------------------------------------

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


def _parse_args():
    p = argparse.ArgumentParser(
        description="XOR N×N alter-based trainer (NMOSWRAP include, PNAS diff-output, Fig 3A placement)"
    )
    p.add_argument("--side", type=int, default=4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--gamma", type=float, default=0.3)
    p.add_argument("--eta", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--solver", type=str, choices=["klu", "sparse"], default="klu")

    # PNAS defaults
    p.add_argument("--vminus", type=float, default=0.11)
    p.add_argument("--vplus", type=float, default=0.33)
    p.add_argument("--vmax", type=float, default=0.45)
    p.add_argument("--L0", type=float, default=-0.087)

    # Optional overrides (use if you want to force exact node indices)
    p.add_argument("--oplus", type=int, default=-1)
    p.add_argument("--ominus", type=int, default=-1)
    p.add_argument("--vminus_node", type=int, default=-1)
    p.add_argument("--vplus_node", type=int, default=-1)
    p.add_argument("--v1_node", type=int, default=-1)
    p.add_argument("--v2_node", type=int, default=-1)

    return p.parse_args()


def main() -> None:
    args = _parse_args()
    side = int(args.side)
    epochs = int(args.epochs)
    gamma = float(args.gamma)
    eta = float(args.eta)
    seed = int(args.seed)
    solver = str(args.solver).lower()

    if side < 4:
        raise ValueError("Figure 3A XOR boundary placement requires --side >= 4")

    random.seed(seed)
    np.random.seed(seed)

    # Dataset
    X, Y = xor_dataset_pnas(Vmax=float(args.vmax), L0=float(args.L0))
    L0 = float(args.L0)

    # Graph + default Fig 3A node map
    G, node_map_default, edge_list = _build_training_graph(side)
    node_map = dict(node_map_default)

    # Apply overrides if provided
    if args.oplus > 0:
        node_map["oplus"] = int(args.oplus)
    if args.ominus > 0:
        node_map["ominus"] = int(args.ominus)
    if args.vminus_node > 0:
        node_map["vminus"] = int(args.vminus_node)
    if args.vplus_node > 0:
        node_map["vplus"] = int(args.vplus_node)
    if args.v1_node > 0:
        node_map["v1"] = int(args.v1_node)
    if args.v2_node > 0:
        node_map["v2"] = int(args.v2_node)

    # Validate nodes exist
    for k, n in node_map.items():
        if int(n) not in G.nodes():
            raise ValueError(f"node_map[{k}]={n} is not a valid grid node for side={side}")

    # Initial weights (gate biases)
    n_edges = len(edge_list)
    vg = np.random.uniform(0.5, 3.0, size=n_edges).astype(float)
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

    # Results directory
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
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + f"_N-{side}_seed-{seed}"
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / "netlist_initial.cir").write_text(netlist)
    meta = {
        "script": str(Path(__file__).resolve()),
        "script_name": Path(__file__).name,
        "argv": list(os.sys.argv),
        "seed": seed,
        "timestamp": datetime.now().isoformat(),
        "grid_side": side,
        "backend": "shared",
        "solver": solver,
        "variant": "xor_nxn_alter_ald1106_pnas_diff_output_fig3A",
        "epochs": epochs,
        "gamma": gamma,
        "eta": eta,
        "rails": {"vminus": float(args.vminus), "vplus": float(args.vplus)},
        "dataset": {"vmax": float(args.vmax), "L0": float(args.L0)},
        "node_map": node_map,
    }
    try:
        (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    except Exception:
        pass

    # Logging
    log_f = _setup_logging(run_dir)
    print("=== RUN START (xor_nxn_alter_ald1106_pnas_diff_output_fig3A) ===", flush=True)

    # Load ngspice
    ng = NgSpiceShared(send_data=False)
    ng.load_circuit(netlist)

    # Helpers for edge updates
    nodes_list = np.asarray(sorted(G.nodes()), dtype=int)
    index_of = np.full(nodes_list.max() + 1, -1, dtype=int)
    index_of[nodes_list] = np.arange(nodes_list.size, dtype=int)
    e1 = np.asarray([a for (a, _) in edge_list], dtype=int)
    e2 = np.asarray([b for (_, b) in edge_list], dtype=int)

    oplus = int(node_map["oplus"])
    ominus = int(node_map["ominus"])

    # Baseline accuracy
    acc_hist: List[float] = []
    acc0 = _evaluate_xor(ng, X, Y, node_map, L0)
    acc_hist.append(acc0)
    print(f"Epoch    0: acc={acc0:.3f}", flush=True)

    outputs_list: List[np.ndarray] = []

    # Track gate-voltage evolution (copy initial vg as epoch 0)
    vg_history: List[np.ndarray] = [vg.copy()]

    # Training loop: free -> destroy -> clamp -> destroy -> update
    for ep in range(1, epochs + 1):
        order = np.arange(X.shape[0])
        np.random.shuffle(order)

        epoch_preds = np.full(X.shape[0], np.nan, dtype=float)

        for idx in order:
            v1 = float(X[idx, 0])
            v2 = float(X[idx, 1])
            L = float(Y[idx, 0])

            # Free phase
            mk_free(ng)
            alter_inputs_two(ng, v1, v2)
            ng.run()

            out = get_voltages(ng, [oplus, ominus])
            free_op = float(out[0])
            free_om = float(out[1])
            free_O = free_op - free_om
            epoch_preds[idx] = free_O

            free_nodes = get_voltages(ng, nodes_list)

            try:
                ng.exec_command("destroy all")
            except Exception:
                pass

            # Clamped phase (differential nudge)
            delta = eta * (L - free_O)
            oc_plus = free_op + 0.5 * delta
            oc_minus = free_om - 0.5 * delta

            alter_outputs_diff(ng, oc_plus, oc_minus)
            ng.run()

            clamped_nodes = get_voltages(ng, nodes_list)

            try:
                ng.exec_command("destroy all")
            except Exception:
                pass

            # Edge updates (same rule form you were using)
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
                    # gate source V1 inside subckt instance Xk
                    cmds.append(f"alter v.x{k}.v1 dc = {nv:.16f}")
                _exec_chunked(ng, cmds)

            mk_free(ng)

        acc_ep = accuracy_nearest_target(epoch_preds.reshape(-1, 1), Y, L0)
        acc_hist.append(float(acc_ep))
        outputs_list.append(np.array(epoch_preds, dtype=float))

        # Snapshot gate voltages for this epoch
        vg_history.append(vg.copy())

        print(f"Epoch {ep:4d}: acc={acc_ep:.3f}", flush=True)

        # Save incrementally
        try:
            np.save(run_dir / "0_outputs.npy", np.array(outputs_list, dtype=object))
            acc_arr = np.asarray(acc_hist, dtype=float)
            np.save(run_dir / "0_acc.npy", acc_arr)
            np.save(run_dir / "0_val_acc.npy", acc_arr)
        except Exception:
            pass

    print(f"FINAL acc={acc_hist[-1]:.4f}", flush=True)
    # Save full gate-voltage history (epoch 0 .. epoch N)
    try:
        np.save(run_dir / "vg_history.npy", np.asarray(vg_history, dtype=float))
    except Exception as e:
        print(f"[WARN] failed to save vg_history.npy: {e}", flush=True)
    print("=== RUN END (xor_nxn_alter_ald1106_pnas_diff_output_fig3A) ===", flush=True)

    try:
        log_f.flush()
        log_f.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
