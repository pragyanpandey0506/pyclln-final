#!/usr/bin/env python3
"""
Scikit Digits — dense input→output NMOS network (no hidden nodes)
Hinge loss training (SGD, no batching) with two-phase (free / clamp) ngspice runs.

Key behavior:
  - Dataset: sklearn digits 8x8 (default; no crop options).
  - Topology: fully-connected input→output (Nin*K edges), unique gate per edge.
  - Per-sample training:
      1) Free phase:   RS all = 1e12 (outputs effectively unclamped), run .op
      2) Compute hinge. If inactive, skip clamp + update.
      3) Clamp phase: clamp only y and rival outputs via RS=1Ω and nudged VOUT sources.
      4) Update: dVG_e = -gamma * ( (dV_e^C)^2 - (dV_e^F)^2 ), clip to [0.4, 10.0]
  - Adaptive epochs:
      * Run base epochs (default 30).
      * If hinge_active_frac at epoch==base_epochs is > threshold (default 0.40),
        extend training up to max epochs (default 50).
  - Logging:
      * Each epoch prints two clean lines: TRAIN + VAL (+ timing) including full config.

Device model:
  - Uses external include file defining:
      .model ncg nmos (...)
      .subckt NMOSWRAP D G S B
         M0 D G S B ncg ...
      .ends
  - Body is ALWAYS tied to source through RB (default 10Ω; configurable via --body-res).
  - Source->V+ diode is OFF (not instantiated).

Paths:
  - Device include file:
      /Users/pragyanpandey/Desktop/pyclln/device_model/nmos_lvl1_ald1106.lib
  - Script location (expected):
      /Users/pragyanpandey/Desktop/pyclln/device_model/scikit_digit/dense_trainer.py
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import networkx as nx
from PySpice.Spice.NgSpice.Shared import NgSpiceShared
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -------------------------
# Fixed device include (external)
# -------------------------
DEVICE_LIB_PATH = "/Users/pragyanpandey/Desktop/pyclln/device_model/nmos_lvl1_ald1106.lib"
DEVICE_SUBCKT = "NMOSWRAP"

VG_CLIP_LO, VG_CLIP_HI = 0.4, 10.0
VG_INIT_LO, VG_INIT_HI = 0.5, 3.0
# Output clamp/free resistors (fixed)
RS_FREE = 1e12
RS_CLAMP = 1.0


# -------------------------
# Logging (tee)
# -------------------------
def setup_logging(run_dir: Path):
    log_path = run_dir / "train_log.txt"
    log_f = open(log_path, "a", buffering=1)

    class Tee:
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

    sys.stdout = Tee(*out_streams)  # type: ignore
    sys.stderr = Tee(*err_streams)  # type: ignore
    return log_f


# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Scikit Digits — dense input→output NMOS network, hinge loss, ngspice"
    )
    p.add_argument("seed", type=int, nargs="?", default=0)

    # Adaptive epochs
    p.add_argument("--epochs", type=int, default=30, help="Base epochs (default 30).")
    p.add_argument("--max-epochs", type=int, default=50, help="Max epochs if extended (default 50).")
    p.add_argument(
        "--extend-if-hinge-active-frac-gt",
        type=float,
        default=0.40,
        help="If hinge_active_frac after base epochs exceeds this, extend to max-epochs.",
    )

    # Learning hyperparams
    p.add_argument("--gamma", type=float, default=0.3)
    p.add_argument("--margin", type=float, default=0.05, help="hinge margin m (volts)")
    p.add_argument("--delta", type=float, default=0.05, help="clamp nudge delta (volts)")

    # Input mapping
    p.add_argument("--input-vmin", type=float, default=0.0)
    p.add_argument("--input-vmax", type=float, default=1.0)

    # Rails
    p.add_argument("--vminus", type=float, default=0.0)
    p.add_argument("--vplus", type=float, default=0.45)

    # Body resistor (body ALWAYS source-tied)
    p.add_argument(
        "--body-res",
        type=float,
        default=10.0,
        help="Body tie resistor in ohms; 0 -> direct body-source tie.",
    )

    # Solver
    p.add_argument("--solver", type=str, choices=["klu", "sparse"], default="klu")

    # Speed knob
    p.add_argument(
        "--update-eps",
        type=float,
        default=0.0,
        help="skip alter if |dVg| < eps (volts)",
    )

    # Debug dataset limits
    p.add_argument("--train-limit", type=int, default=0, help="if >0, limit train samples")
    p.add_argument("--test-limit", type=int, default=0, help="if >0, limit test samples")

    # Diagnostics
    p.add_argument(
        "--diag-subset",
        type=int,
        default=200,
        help="fixed validation subset size for margin/Vout diagnostics (default 200)",
    )

    p.add_argument("--final-test", action="store_true")
    return p.parse_args()


# -------------------------
# Ngspice helpers
# -------------------------
def exec_chunked(ng: NgSpiceShared, cmds: List[str], max_len: int = 900, sep: str = "; "):
    buf: List[str] = []
    length = 0
    for c in cmds:
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


def get_voltages_multi(ng: NgSpiceShared, read_specs: Dict[str, List[int]]) -> Dict[str, np.ndarray]:
    s = ng.exec_command("print allv")
    nodemap: Dict[int, float] = {}
    for line in s.splitlines():
        line = line.strip()
        if not line.startswith("v("):
            continue
        try:
            k, v = line.split(" = ")
            node_str = k[2:-1].strip()
            if node_str.isdigit():
                nodemap[int(node_str)] = float(v)
        except Exception:
            continue

    out: Dict[str, np.ndarray] = {}
    for key, nodes in read_specs.items():
        out[key] = np.array([float(nodemap.get(int(n), np.nan)) for n in nodes], dtype=float)
    return out


def run_and_read(
    ng: NgSpiceShared,
    read_specs: Dict[str, List[int]],
) -> Tuple[bool, float, Optional[Dict[str, np.ndarray]], Optional[str]]:
    t0 = time.time()
    try:
        ng.run()
        dt = time.time() - t0
        data = get_voltages_multi(ng, read_specs)
        try:
            ng.exec_command("destroy all")
        except Exception:
            pass
        return True, float(dt), data, None
    except Exception as e:
        return False, 0.0, None, str(e)


def alter_inputs_named(ng: NgSpiceShared, values: np.ndarray):
    cmds = [f"alter VIN{i} dc = {float(v):.16f}" for i, v in enumerate(values)]
    exec_chunked(ng, cmds)


def mk_free_all(ng: NgSpiceShared, K: int):
    exec_chunked(ng, [f"alter RS{i} {RS_FREE:.6g}" for i in range(1, K + 1)])


def alter_outputs_hinge(
    ng: NgSpiceShared,
    K: int,
    y: int,
    r: int,
    Vy: float,
    Vr: float,
    delta: float,
):
    # Set all free
    cmds: List[str] = [f"alter RS{i} {RS_FREE:.6g}" for i in range(1, K + 1)]
    # Clamp y and rival through RS=1
    cmds.append(f"alter RS{y+1} {RS_CLAMP:.6g}")
    cmds.append(f"alter RS{r+1} {RS_CLAMP:.6g}")

    Vy_c = float(Vy + 0.5 * delta)
    Vr_c = float(Vr - 0.5 * delta)
    cmds.append(f"alter VOUT{y} dc = {Vy_c:.16f}")
    cmds.append(f"alter VOUT{r} dc = {Vr_c:.16f}")
    exec_chunked(ng, cmds)


# -------------------------
# Topology + netlist
# -------------------------
@dataclass
class Topology:
    negref: int
    posref: int
    input_nodes: List[int]
    out_nodes: List[int]
    edges_D: np.ndarray  # (E,)
    edges_S: np.ndarray  # (E,)
    num_edges: int


def build_dense_topology(Nin: int, K: int = 10) -> Topology:
    negref = 1
    posref = 2

    out_nodes = [3 + k for k in range(K)]
    input_nodes = [out_nodes[-1] + 1 + i for i in range(Nin)]

    edges: List[Tuple[int, int]] = []
    for i in range(Nin):
        d = input_nodes[i]
        for k in range(K):
            s = out_nodes[k]
            edges.append((d, s))

    edges_D = np.array([d for d, _ in edges], dtype=int)
    edges_S = np.array([s for _, s in edges], dtype=int)

    return Topology(
        negref=negref,
        posref=posref,
        input_nodes=input_nodes,
        out_nodes=out_nodes,
        edges_D=edges_D,
        edges_S=edges_S,
        num_edges=int(len(edges)),
    )


def mk_netlist(
    topo: Topology,
    vg_unique: np.ndarray,
    vminus_val: float,
    vplus_val: float,
    solver: str,
    body_res: float,
) -> str:
    if vg_unique.size != topo.num_edges:
        raise ValueError("vg_unique size mismatch")

    lines: List[str] = []
    lines.append(".title scikit_digits_dense_io_hinge")
    lines.append(f'.include "{DEVICE_LIB_PATH}"')

    if solver.lower() == "klu":
        lines.append(".options klu")

    # Rails
    lines.append(f"VMINUS {topo.negref} 0 {float(vminus_val):.16f}")
    lines.append(f"VPLUS  {topo.posref} 0 {float(vplus_val):.16f}")

    # Inputs
    for i, n in enumerate(topo.input_nodes):
        lines.append(f"VIN{i} {n} 0 0")

    # Outputs: RS + sink node + VOUT sources
    max_node = max(topo.input_nodes + topo.out_nodes + [topo.negref, topo.posref])
    sink0 = max_node + 1
    K = len(topo.out_nodes)

    for i, on in enumerate(topo.out_nodes, start=1):
        lines.append(f"RS{i} {on} {sink0 + (i-1)} {RS_FREE:.6g}")
    for j in range(K):
        lines.append(f"VOUT{j} {sink0 + j} 0 0")

    # Devices (unique VG per edge). Body ALWAYS tied to source.
    for eidx, (D, S) in enumerate(zip(topo.edges_D.tolist(), topo.edges_S.tolist())):
        gate_node = f"g{eidx}"
        lines.append(f"VG{eidx} {gate_node} 0 {float(vg_unique[eidx]):.16f}")

        if body_res <= 0.0:
            body_node = str(S)
        else:
            body_node = f"b{eidx}"
            lines.append(f"RB{eidx} {body_node} {S} {float(body_res):.6g}")

        # Instantiate wrapper subckt from included file
        # NMOSWRAP pins: D G S B
        lines.append(f"X{eidx} {D} {gate_node} {S} {body_node} {DEVICE_SUBCKT}")

        # Source->V+ diode is intentionally NOT instantiated.

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


# -------------------------
# Metrics
# -------------------------
def hinge_loss_from_outputs(V: np.ndarray, y: int, margin: float) -> float:
    Vy = float(V[y])
    V2 = V.copy()
    V2[y] = -np.inf
    Vr = float(np.max(V2))
    return float(max(0.0, margin - (Vy - Vr)))


def margin_gap(V: np.ndarray, y: int) -> float:
    Vy = float(V[y])
    V2 = V.copy()
    V2[y] = -np.inf
    Vr = float(np.max(V2))
    return float(Vy - Vr)


def pred_and_rival(V: np.ndarray, y: int) -> Tuple[int, int]:
    pred = int(np.argmax(V))
    V2 = V.copy()
    V2[y] = -np.inf
    rival = int(np.argmax(V2))
    return pred, rival


def restore_gate_voltages(ng: NgSpiceShared, vg_unique: np.ndarray):
    exec_chunked(ng, [f"alter VG{i} dc = {float(vg_unique[i]):.16f}" for i in range(vg_unique.size)])


def compute_vg_saturation_stats(vg_unique: np.ndarray) -> Dict[str, float]:
    lo, hi = VG_CLIP_LO, VG_CLIP_HI
    return {
        "vg_unique_min": float(np.min(vg_unique)),
        "vg_unique_max": float(np.max(vg_unique)),
        "vg_unique_sat_lo": float(np.sum(vg_unique <= (lo + 1e-12))),
        "vg_unique_sat_hi": float(np.sum(vg_unique >= (hi - 1e-12))),
    }


def save_plots(run_dir: Path):
    def _load(name: str):
        p = run_dir / name
        return np.load(p) if p.exists() else None

    tr_acc = _load("0_train_acc.npy")
    tr_h = _load("0_train_hinge.npy")
    va_acc = _load("0_val_acc.npy")
    va_h = _load("0_val_hinge.npy")

    ep_total = _load("0_epoch_total_s.npy")
    ep_free = _load("0_epoch_free_s.npy")
    ep_clamp = _load("0_epoch_clamp_s.npy")
    ep_upd = _load("0_epoch_update_s.npy")

    hinge_frac = _load("0_hinge_active_frac.npy")

    if tr_acc is not None and va_acc is not None:
        plt.figure()
        plt.plot(np.arange(len(va_acc)), va_acc, label="val acc")
        plt.plot(np.arange(1, len(tr_acc) + 1), tr_acc, label="train acc")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")
        plt.title("Accuracy vs epoch")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(run_dir / "learning_curves_acc.png", dpi=160)
        plt.close()

    if tr_h is not None and va_h is not None:
        plt.figure()
        plt.plot(np.arange(len(va_h)), va_h, label="val hinge")
        plt.plot(np.arange(1, len(tr_h) + 1), tr_h, label="train hinge")
        plt.xlabel("epoch")
        plt.ylabel("hinge loss")
        plt.title("Hinge loss vs epoch")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(run_dir / "learning_curves_hinge.png", dpi=160)
        plt.close()

    if ep_total is not None and ep_free is not None and ep_clamp is not None and ep_upd is not None:
        e = np.arange(1, len(ep_total) + 1)
        plt.figure()
        plt.plot(e, ep_total, label="epoch total")
        plt.plot(e, ep_free, label="free")
        plt.plot(e, ep_clamp, label="clamp")
        plt.plot(e, ep_upd, label="update")
        plt.xlabel("epoch")
        plt.ylabel("seconds")
        plt.title("Timing vs epoch")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(run_dir / "timing.png", dpi=160)
        plt.close()

    if hinge_frac is not None:
        e = np.arange(1, len(hinge_frac) + 1)
        plt.figure()
        plt.plot(e, hinge_frac, label="hinge active fraction")
        plt.xlabel("epoch")
        plt.ylabel("fraction")
        plt.title("Hinge-active fraction vs epoch")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(run_dir / "hinge_active.png", dpi=160)
        plt.close()


# -------------------------
# Main
# -------------------------
def main():
    args = parse_args()
    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)

    base_epochs = int(args.epochs)
    max_epochs = int(args.max_epochs)
    extend_thresh = float(args.extend_if_hinge_active_frac_gt)
    if base_epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if max_epochs < base_epochs:
        raise ValueError("--max-epochs must be >= --epochs")
    if not (0.0 <= extend_thresh <= 1.0):
        raise ValueError("--extend-if-hinge-active-frac-gt must be in [0,1]")

    gamma = float(args.gamma)
    margin = float(args.margin)
    delta = float(args.delta)
    update_eps = float(args.update_eps)

    vmin = float(args.input_vmin)
    vmax = float(args.input_vmax)
    if vmax <= vmin:
        raise ValueError("--input-vmax must be > --input-vmin")

    vminus_val = float(args.vminus)
    vplus_val = float(args.vplus)
    solver = str(args.solver).lower()
    body_res = float(args.body_res)

    # Dataset: scikit digits 8x8 only
    digits = load_digits()
    imgs = (digits.images / 16.0).astype(np.float64)  # (N,8,8) in [0,1]
    y = digits.target.astype(int)

    X_raw = imgs.reshape(len(imgs), -1)  # Nin=64
    X = vmin + (vmax - vmin) * X_raw

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    if args.train_limit and args.train_limit > 0:
        X_train = X_train[: args.train_limit]
        y_train = y_train[: args.train_limit]
    if args.test_limit and args.test_limit > 0:
        X_test = X_test[: args.test_limit]
        y_test = y_test[: args.test_limit]

    train_x = [row.astype(float) for row in X_train]
    train_y = [int(v) for v in y_train.tolist()]
    test_x = [row.astype(float) for row in X_test]
    test_y = [int(v) for v in y_test.tolist()]

    Nin = int(train_x[0].size)  # should be 64
    K = 10

    topo = build_dense_topology(Nin=Nin, K=K)

    # Diag subset
    diag_n = int(max(0, args.diag_subset))
    diag_n = min(diag_n, len(test_x))
    rng = np.random.default_rng(seed + 12345)
    diag_idx = rng.choice(len(test_x), size=diag_n, replace=False) if diag_n > 0 else np.array([], dtype=int)
    diag_idx_set = set(int(i) for i in diag_idx.tolist())

    # Init weights (unique VG per edge)
    vg_unique = np.random.uniform(VG_INIT_LO, VG_INIT_HI, size=(topo.num_edges,)).astype(float)

    # Run directory
    results_dir = Path(__file__).resolve().parent / "results"
    env_run_dir = os.environ.get("RUN_DIR")
    if env_run_dir:
        run_dir = Path(env_run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        runs_dir = results_dir / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + f"_seed-{seed}"
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

    log_f = setup_logging(run_dir)

    cfg_str = (
        f"seed={seed} gamma={gamma} delta={delta} margin={margin} "
        f"in=[{vmin},{vmax}] Nin={Nin} K={K} rails=[{vminus_val},{vplus_val}] "
        f"solver={solver} body_source=True body_res={body_res} "
        f"epochs={base_epochs}->{max_epochs} if hinge_frac>{extend_thresh} "
        f"device_include={DEVICE_LIB_PATH} subckt={DEVICE_SUBCKT}"
    )

    print("=== RUN START (scikit_digits_dense_io_hinge) ===", flush=True)
    print(cfg_str, flush=True)
    print(f"train={len(train_x)} test={len(test_x)} edges={topo.num_edges}", flush=True)

    # Netlist
    netlist = mk_netlist(
        topo=topo,
        vg_unique=vg_unique,
        vminus_val=vminus_val,
        vplus_val=vplus_val,
        solver=solver,
        body_res=body_res,
    )
    (run_dir / "netlist_initial.cir").write_text(netlist)

    meta = {
        "script": str(Path(__file__).resolve()),
        "argv": list(os.sys.argv),
        "seed": seed,
        "timestamp": datetime.now().isoformat(),
        "dataset": "sklearn_digits",
        "input_shape": [8, 8],
        "train_count": len(train_x),
        "test_count": len(test_x),
        "gamma": gamma,
        "margin": margin,
        "delta": delta,
        "epochs_base": base_epochs,
        "epochs_max": max_epochs,
        "extend_if_hinge_active_frac_gt": extend_thresh,
        "input_vmin": vmin,
        "input_vmax": vmax,
        "rails": {"vminus": vminus_val, "vplus": vplus_val},
        "solver": solver,
        "body_source": True,
        "body_res": body_res,
        "update_eps": update_eps,
        "device": {
            "include_path": DEVICE_LIB_PATH,
            "subckt": DEVICE_SUBCKT,
        },
        "topology": {
            "type": "dense_input_to_output",
            "Nin": Nin,
            "out": K,
            "edges": topo.num_edges,
        },
        "diagnostics": {"diag_subset": diag_n},
        "diode_source_to_vplus": False,
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

    np.save(run_dir / "diag_subset_idx.npy", diag_idx.astype(int))
    np.save(run_dir / "diag_subset_y.npy", np.asarray([test_y[i] for i in diag_idx.tolist()], dtype=int))
    np.save(run_dir / "diag_subset_x.npy", np.asarray([test_x[i] for i in diag_idx.tolist()], dtype=float))

    # Ngspice
    ng = NgSpiceShared(send_data=False)
    ng.load_circuit(netlist)

    # Nodes list (numeric nodes only)
    net_nodes = [topo.negref, topo.posref] + topo.out_nodes + topo.input_nodes
    nodes_list = np.asarray(sorted(set(net_nodes)), dtype=int)

    index_of = np.full(nodes_list.max() + 1, -1, dtype=int)
    index_of[nodes_list] = np.arange(nodes_list.size, dtype=int)

    eD = topo.edges_D
    eS = topo.edges_S

    # Graph export (optional)
    try:
        G = nx.DiGraph()
        G.add_nodes_from(nodes_list.tolist())
        for d, s in zip(eD.tolist(), eS.tolist()):
            G.add_edge(d, s)
        nx.write_graphml(G, str(run_dir / "0.graphml"))
    except Exception:
        pass

    # Histories
    val_acc_hist: List[float] = []
    val_hinge_hist: List[float] = []
    tr_acc_hist: List[float] = []
    tr_hinge_hist: List[float] = []

    ep_total_s: List[float] = []
    ep_free_s: List[float] = []
    ep_clamp_s: List[float] = []
    ep_update_s: List[float] = []

    hinge_active_frac_hist: List[float] = []
    reload_free_hist: List[int] = []
    reload_clamp_hist: List[int] = []
    nonfinite_free_hist: List[int] = []
    nonfinite_clamp_hist: List[int] = []

    def eval_free_metrics(epoch: int) -> Tuple[float, float, Dict[str, float]]:
        mk_free_all(ng, K)
        correct = 0
        total = 0
        loss_sum = 0.0
        count = 0

        vout_subset = np.full((diag_n, K), np.nan, dtype=float) if diag_n > 0 else np.zeros((0, K), dtype=float)
        gap_list: List[float] = []
        sat_list: List[float] = []
        hinge_list: List[float] = []
        subset_pos = {int(idx): j for j, idx in enumerate(diag_idx.tolist())}

        reloads = 0
        nonfinite = 0

        for i, (xt, yt) in enumerate(zip(test_x, test_y)):
            alter_inputs_named(ng, xt)
            ok, _, data, _ = run_and_read(ng, {"out": topo.out_nodes})
            if not ok or data is None:
                reloads += 1
                try:
                    ng.remove_circuit()
                except Exception:
                    pass
                ng.load_circuit(netlist)
                restore_gate_voltages(ng, vg_unique)
                mk_free_all(ng, K)
                continue

            Vout = np.asarray(data["out"], dtype=float)
            if not np.all(np.isfinite(Vout)):
                nonfinite += 1
                continue

            pred = int(np.argmax(Vout))
            ytrue = int(yt)
            correct += int(pred == ytrue)
            total += 1

            hl = hinge_loss_from_outputs(Vout, ytrue, margin)
            loss_sum += float(hl)
            count += 1

            if i in diag_idx_set:
                j = subset_pos[i]
                vout_subset[j, :] = Vout
                g = margin_gap(Vout, ytrue)
                gap_list.append(float(g))
                sat_list.append(float(1.0 if g >= margin else 0.0))
                hinge_list.append(float(hl))

        if diag_n > 0:
            np.save(run_dir / f"0_vout_subset_epoch{epoch}.npy", vout_subset)

        diag: Dict[str, float] = {}
        if gap_list:
            gaps = np.asarray(gap_list, dtype=float)
            sats = np.asarray(sat_list, dtype=float)
            hinges = np.asarray(hinge_list, dtype=float)
            diag["subset_gap_mean"] = float(np.mean(gaps))
            diag["subset_gap_median"] = float(np.median(gaps))
            diag["subset_gap_std"] = float(np.std(gaps))
            diag["subset_satisfy_frac"] = float(np.mean(sats))
            diag["subset_hinge_mean"] = float(np.mean(hinges))
        else:
            diag["subset_gap_mean"] = float("nan")
            diag["subset_gap_median"] = float("nan")
            diag["subset_gap_std"] = float("nan")
            diag["subset_satisfy_frac"] = float("nan")
            diag["subset_hinge_mean"] = float("nan")

        diag["val_reloads"] = float(reloads)
        diag["val_nonfinite"] = float(nonfinite)

        acc = (correct / total) if total else float("nan")
        loss = (loss_sum / count) if count else float("nan")
        return float(acc), float(loss), diag

    # Epoch 0 validation
    v0, h0, diag0 = eval_free_metrics(epoch=0)
    val_acc_hist.append(v0)
    val_hinge_hist.append(h0)
    np.save(run_dir / "0_val_acc.npy", np.asarray(val_acc_hist, dtype=float))
    np.save(run_dir / "0_val_hinge.npy", np.asarray(val_hinge_hist, dtype=float))
    (run_dir / "0_diag_epoch0.json").write_text(json.dumps(diag0, indent=2))
    print(
        f"[epoch 0] {cfg_str} | VAL acc={v0:.4f} hinge={h0:.6f} subset_satisfy={diag0.get('subset_satisfy_frac', float('nan')):.4f}",
        flush=True,
    )

    target_epochs = base_epochs
    ep = 1
    while ep <= target_epochs:
        t_ep0 = time.time()
        order = np.arange(len(train_x))
        np.random.shuffle(order)

        train_correct = 0
        train_total = 0
        hinge_sum = 0.0
        hinge_count = 0
        hinge_active = 0
        skipped = 0

        reload_free = 0
        reload_clamp = 0
        nonfinite_free = 0
        nonfinite_clamp = 0

        t_free = 0.0
        t_clamp = 0.0
        t_update = 0.0
        n_free = 0
        n_clamp = 0

        for idx in order:
            ytrue = int(train_y[idx])
            mk_free_all(ng, K)

            # Free
            alter_inputs_named(ng, train_x[idx])
            ok, dt, data, _ = run_and_read(ng, {"out": topo.out_nodes, "nodes": nodes_list.tolist()})
            t_free += float(dt)
            n_free += 1

            if not ok or data is None:
                reload_free += 1
                try:
                    ng.remove_circuit()
                except Exception:
                    pass
                ng.load_circuit(netlist)
                restore_gate_voltages(ng, vg_unique)
                mk_free_all(ng, K)
                continue

            Vout = np.asarray(data["out"], dtype=float)
            Vnodes_free = np.asarray(data["nodes"], dtype=float)
            if (not np.all(np.isfinite(Vout))) or (not np.all(np.isfinite(Vnodes_free))):
                nonfinite_free += 1
                continue

            pred, rival = pred_and_rival(Vout, ytrue)
            train_correct += int(pred == ytrue)
            train_total += 1

            hl = hinge_loss_from_outputs(Vout, ytrue, margin)
            hinge_sum += float(hl)
            hinge_count += 1

            if hl <= 0.0:
                skipped += 1
                continue

            hinge_active += 1

            # Clamp
            Vy = float(Vout[ytrue])
            Vr = float(Vout[rival])
            alter_outputs_hinge(ng, K=K, y=ytrue, r=rival, Vy=Vy, Vr=Vr, delta=delta)

            ok, dt2, data2, _ = run_and_read(ng, {"nodes": nodes_list.tolist()})
            t_clamp += float(dt2)
            n_clamp += 1

            if not ok or data2 is None:
                reload_clamp += 1
                try:
                    ng.remove_circuit()
                except Exception:
                    pass
                ng.load_circuit(netlist)
                restore_gate_voltages(ng, vg_unique)
                mk_free_all(ng, K)
                continue

            Vnodes_clamp = np.asarray(data2["nodes"], dtype=float)
            if not np.all(np.isfinite(Vnodes_clamp)):
                nonfinite_clamp += 1
                continue

            # Update
            upd0 = time.time()

            Vd_free = Vnodes_free[index_of[eD]]
            Vs_free = Vnodes_free[index_of[eS]]
            Vd_c = Vnodes_clamp[index_of[eD]]
            Vs_c = Vnodes_clamp[index_of[eS]]

            dV_free = Vd_free - Vs_free
            dV_c = Vd_c - Vs_c

            update = -gamma * (dV_c**2 - dV_free**2)

            cmds: List[str] = []
            for uid in range(topo.num_edges):
                du = float(update[uid])
                if update_eps > 0.0 and abs(du) < update_eps:
                    continue
                nv = float(vg_unique[uid] + du)
                if nv < VG_CLIP_LO:
                    nv = VG_CLIP_LO
                elif nv > VG_CLIP_HI:
                    nv = VG_CLIP_HI
                vg_unique[uid] = nv
                cmds.append(f"alter VG{uid} dc = {nv:.16f}")

            if cmds:
                exec_chunked(ng, cmds)

            t_update += float(time.time() - upd0)

        tr_acc = (train_correct / train_total) if train_total else float("nan")
        tr_h = (hinge_sum / hinge_count) if hinge_count else float("nan")
        tr_acc_hist.append(float(tr_acc))
        tr_hinge_hist.append(float(tr_h))

        hinge_active_frac = (hinge_active / n_free) if n_free else float("nan")
        hinge_active_frac_hist.append(float(hinge_active_frac))

        reload_free_hist.append(int(reload_free))
        reload_clamp_hist.append(int(reload_clamp))
        nonfinite_free_hist.append(int(nonfinite_free))
        nonfinite_clamp_hist.append(int(nonfinite_clamp))

        v_acc, v_h, diag = eval_free_metrics(epoch=ep)
        val_acc_hist.append(float(v_acc))
        val_hinge_hist.append(float(v_h))

        ep_total = float(time.time() - t_ep0)
        ep_total_s.append(ep_total)
        ep_free_s.append(float(t_free))
        ep_clamp_s.append(float(t_clamp))
        ep_update_s.append(float(t_update))

        # Save arrays each epoch
        np.save(run_dir / "0_train_acc.npy", np.asarray(tr_acc_hist, dtype=float))
        np.save(run_dir / "0_train_hinge.npy", np.asarray(tr_hinge_hist, dtype=float))
        np.save(run_dir / "0_val_acc.npy", np.asarray(val_acc_hist, dtype=float))
        np.save(run_dir / "0_val_hinge.npy", np.asarray(val_hinge_hist, dtype=float))

        np.save(run_dir / "0_epoch_total_s.npy", np.asarray(ep_total_s, dtype=float))
        np.save(run_dir / "0_epoch_free_s.npy", np.asarray(ep_free_s, dtype=float))
        np.save(run_dir / "0_epoch_clamp_s.npy", np.asarray(ep_clamp_s, dtype=float))
        np.save(run_dir / "0_epoch_update_s.npy", np.asarray(ep_update_s, dtype=float))

        np.save(run_dir / "0_hinge_active_frac.npy", np.asarray(hinge_active_frac_hist, dtype=float))
        np.save(run_dir / "0_reload_free.npy", np.asarray(reload_free_hist, dtype=int))
        np.save(run_dir / "0_reload_clamp.npy", np.asarray(reload_clamp_hist, dtype=int))
        np.save(run_dir / "0_nonfinite_free.npy", np.asarray(nonfinite_free_hist, dtype=int))
        np.save(run_dir / "0_nonfinite_clamp.npy", np.asarray(nonfinite_clamp_hist, dtype=int))

        np.save(run_dir / f"0_vg_unique_epoch{ep}.npy", vg_unique.copy())

        vg_stats = compute_vg_saturation_stats(vg_unique)
        summary = {
            "epoch": int(ep),
            "config": {
                "seed": seed,
                "gamma": gamma,
                "margin": margin,
                "delta": delta,
                "input_vmin": vmin,
                "input_vmax": vmax,
                "rails": [vminus_val, vplus_val],
                "solver": solver,
                "body_source": True,
                "body_res": body_res,
                "epochs_base": base_epochs,
                "epochs_max": max_epochs,
                "extend_if_hinge_active_frac_gt": extend_thresh,
                "device_include_path": DEVICE_LIB_PATH,
                "device_subckt": DEVICE_SUBCKT,
            },
            "train": {
                "acc": float(tr_acc),
                "hinge": float(tr_h),
                "n_free": int(n_free),
                "n_clamp": int(n_clamp),
                "hinge_active": int(hinge_active),
                "hinge_active_frac": float(hinge_active_frac),
                "skipped": int(skipped),
                "reload_free": int(reload_free),
                "reload_clamp": int(reload_clamp),
                "nonfinite_free": int(nonfinite_free),
                "nonfinite_clamp": int(nonfinite_clamp),
            },
            "val": {"acc": float(v_acc), "hinge": float(v_h), **{k: float(v) for k, v in diag.items()}},
            "timing_s": {
                "epoch_total": float(ep_total),
                "train_free": float(t_free),
                "train_clamp": float(t_clamp),
                "train_update": float(t_update),
            },
            "vg_stats": vg_stats,
        }
        (run_dir / f"0_epoch_summary_epoch{ep}.json").write_text(json.dumps(summary, indent=2))
        (run_dir / f"0_diag_epoch{ep}.json").write_text(json.dumps(diag, indent=2))

        # Clean epoch prints (config repeated each epoch)
        print(
            f"[epoch {ep}/{target_epochs}] {cfg_str} | "
            f"TRAIN acc={tr_acc:.4f} hinge={tr_h:.6f} hinge_frac={hinge_active_frac:.3f} "
            f"free={n_free} clamp={n_clamp} skipped={skipped} "
            f"reloadF={reload_free} reloadC={reload_clamp} nonfiniteF={nonfinite_free} nonfiniteC={nonfinite_clamp}",
            flush=True,
        )
        print(
            f"[epoch {ep}/{target_epochs}] {cfg_str} | "
            f"VAL acc={v_acc:.4f} hinge={v_h:.6f} "
            f"subset_satisfy={diag.get('subset_satisfy_frac', float('nan')):.4f} "
            f"subset_gap_mean={diag.get('subset_gap_mean', float('nan')):.4f} | "
            f"timing total={ep_total:.2f}s free={t_free:.2f}s clamp={t_clamp:.2f}s upd={t_update:.2f}s",
            flush=True,
        )

        # Adaptive extension decision only right after base_epochs
        if (ep == base_epochs) and (target_epochs == base_epochs):
            if (hinge_active_frac > extend_thresh) and (max_epochs > base_epochs):
                target_epochs = max_epochs
                print(
                    f"[extend] hinge_active_frac={hinge_active_frac:.3f} > {extend_thresh:.3f} "
                    f"-> extending to {target_epochs} epochs | {cfg_str}",
                    flush=True,
                )

        try:
            save_plots(run_dir)
        except Exception:
            pass

        ep += 1

    # latest symlink
    latest = results_dir / "latest"
    try:
        if latest.is_symlink() or latest.exists():
            latest.unlink()
    except Exception:
        pass
    try:
        latest.symlink_to(run_dir.resolve())
    except Exception:
        pass

    if args.final_test:
        print("FINAL val acc=", val_acc_hist[-1] if val_acc_hist else float("nan"), flush=True)

    print("=== RUN END (scikit_digits_dense_io_hinge) ===", flush=True)
    try:
        log_f.flush()
        log_f.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
