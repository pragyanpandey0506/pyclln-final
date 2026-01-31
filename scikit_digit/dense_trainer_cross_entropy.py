#!/usr/bin/env python3
"""
Scikit Digits — dense input→output NMOS network (no hidden nodes)
Cross-entropy training (SGD, no batching) with two-phase (free / clamp) ngspice runs.

Key behavior:
  - Dataset: sklearn digits 8x8 (default; no crop options).
  - Topology: loaded from NPZ (digits_8x8_dense_io_x1), with parallel edges per input/output.
  - Init VG: random in [VG_INIT_LO, VG_INIT_HI] or fixed VG_INIT_SINGLE via --vg-init.
  - Per-sample training (cross entropy):
      1) Free phase:   RS all = RS_FREE (outputs effectively unclamped), run .op
      2) Compute softmax probabilities p = softmax(Vout / T)
      3) Clamp phase (TRUE K-class CE signal):
           - Clamp ALL outputs via RS=RS_CLAMP
           - Set VOUTk to Vk^C = Vk^F + delta * (qk - pk), where q is one-hot for y
      4) Update: dVG_e = -gamma * ( (dV_e^C)^2 - (dV_e^F)^2 ), clip to [VG_CLIP_LO, VG_CLIP_HI]
  - Epochs:
      * Run a fixed number of epochs (default 20).
  - Logging:
      * Each epoch prints two clean lines: TRAIN + VAL (+ timing) including full config.

Device model:
  - Uses external include file defining:
      .model ncg nmos (...)
      .subckt NMOSWRAP D G S B
         M0 D G S B ncg ...
      .ends
  - Body tie is selectable via --body-tie (source, ground, floating); default is ground.
    Body resistor is fixed to RS_CLAMP.
  - Source->V+ diode is OFF (not instantiated).

Paths:
  - Device include file (repo-relative):
      device_model/nmos_lvl1_ald1106.lib
  - Topology file (repo-relative):
      scikit_digit/topology/digits_8x8_dense_io_x1.npz
  - Script location (repo-relative):
      scikit_digit/dense_trainer.py
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
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

from topology.topology_io import Topology, load_topology_npz


# -------------------------
# Fixed device include (external)
# -------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DEVICE_LIB_PATH = str(REPO_ROOT / "device_model" / "nmos_lvl1_ald1106.lib")
DEVICE_SUBCKT = "NMOSWRAP"
TOPOLOGY_PATH = REPO_ROOT / "scikit_digit" / "topology" / "random_topology_0799.npz"
#TOPOLOGY_PATH = REPO_ROOT / "scikit_digit" / "topology" / "digits_8x8_dense_io_x1_pruned_vglt0p75_epoch20_run20260110-235531.npz"

VG_CLIP_LO, VG_CLIP_HI = 0.4, 8.0
VG_INIT_SINGLE = 2.0
# Output clamp/free resistors (fixed)
RS_FREE = 1e9
RS_CLAMP = 10.0


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
        description="Scikit Digits — dense input→output NMOS network, cross entropy, ngspice"
    )
    p.add_argument("seed", type=int, nargs="?", default=0)

    # Training epochs
    p.add_argument("--epochs", type=int, default=20, help="Total epochs (default 20).")

    # Learning hyperparams
    p.add_argument("--gamma", type=float, default=0.3)
    p.add_argument("--delta", type=float, default=0.3, help="clamp nudge scale delta (volts)")
    p.add_argument(
        "--softmax-temp",
        type=float,
        default=1.0,
        help="softmax temperature T (probabilities use softmax(Vout / T)); default 1.0",
    )

    # Input mapping
    p.add_argument("--input-vmin", type=float, default=0.0)
    p.add_argument("--input-vmax", type=float, default=1.0)

    # Rails
    p.add_argument("--vminus", type=float, default=0.0)
    p.add_argument("--vplus", type=float, default=0.45)

    # Body tie
    p.add_argument(
        "--body-tie",
        type=str,
        choices=["source", "ground", "floating"],
        default="ground",
        help="Body tie mode: source, ground, or floating (default ground).",
    )
    # Init VG
    p.add_argument(
        "--vg-init",
        type=str,
        choices=["random", "fixed"],
        default="random",
        help="VG init mode: random or fixed (default random).",
    )
    p.add_argument(
        "--vg-init-lo",
        type=float,
        default=1.0,
        help="Random VG init low (default from trainer).",
    )
    p.add_argument(
        "--vg-init-hi",
        type=float,
        default=3.0,
        help="Random VG init high (default from trainer).",
    )
    p.add_argument(
        "--vg-init-fixed",
        type=float,
        default=VG_INIT_SINGLE,
        help="Fixed VG init value (default from trainer).",
    )

    # Solver
    p.add_argument("--solver", type=str, choices=["klu", "sparse"], default="klu")

    # Debug dataset limits
    p.add_argument("--train-limit", type=int, default=0, help="if >0, limit train samples")
    p.add_argument("--test-limit", type=int, default=0, help="if >0, limit test samples")

    # Diagnostics
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


def softmax_logits(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    m = float(np.max(z))
    ez = np.exp(z - m)
    s = float(np.sum(ez))
    if not np.isfinite(s) or s <= 0.0:
        return np.full_like(z, 1.0 / float(z.size), dtype=float)
    return ez / s


def alter_outputs_xent(
    ng: NgSpiceShared,
    K: int,
    y: int,
    Vout_free: np.ndarray,
    delta: float,
    temp: float,
):
    """
    TRUE K-class cross-entropy clamp target:
      p = softmax(Vout_free / temp)
      q = onehot(y)
      Vout_clamp = Vout_free + delta * (q - p)

    And we clamp ALL outputs (RS_i = RS_CLAMP for i=1..K).
    """
    if temp <= 0.0:
        raise ValueError("--softmax-temp must be > 0")

    z = np.asarray(Vout_free, dtype=float) / float(temp)
    p = softmax_logits(z)

    q = np.zeros(K, dtype=float)
    q[int(y)] = 1.0

    dV = float(delta) * (q - p)
    Vc = np.asarray(Vout_free, dtype=float) + dV

    cmds: List[str] = []
    # Clamp all outputs
    for i in range(1, K + 1):
        cmds.append(f"alter RS{i} {RS_CLAMP:.6g}")
    # Set all output sources
    for k in range(K):
        cmds.append(f"alter VOUT{k} dc = {float(Vc[k]):.16f}")

    exec_chunked(ng, cmds)


def mk_netlist(
    topo: Topology,
    vg_unique: np.ndarray,
    vminus_val: float,
    vplus_val: float,
    solver: str,
    body_res: float,
    body_tie: str,
) -> str:
    if vg_unique.size != topo.num_edges:
        raise ValueError("vg_unique size mismatch")

    lines: List[str] = []
    lines.append(".title scikit_digits_dense_io_xent")
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
    node_pool = [topo.negref, topo.posref] + topo.input_nodes.tolist() + topo.out_nodes.tolist()
    max_node = max(node_pool)
    sink0 = max_node + 1
    K = len(topo.out_nodes)

    for i, on in enumerate(topo.out_nodes, start=1):
        lines.append(f"RS{i} {on} {sink0 + (i-1)} {RS_FREE:.6g}")
    for j in range(K):
        lines.append(f"VOUT{j} {sink0 + j} 0 0")

    # Devices (unique VG per edge). Body tie selectable.
    for eidx, (D, S) in enumerate(zip(topo.edges_D.tolist(), topo.edges_S.tolist())):
        gate_node = f"g{eidx}"
        lines.append(f"VG{eidx} {gate_node} 0 {float(vg_unique[eidx]):.16f}")

        if body_tie == "source":
            if body_res <= 0.0:
                body_node = str(S)
            else:
                body_node = f"b{eidx}"
                lines.append(f"RB{eidx} {body_node} {S} {float(body_res):.6g}")
        elif body_tie == "ground":
            if body_res <= 0.0:
                body_node = "0"
            else:
                body_node = f"b{eidx}"
                lines.append(f"RB{eidx} {body_node} 0 {float(body_res):.6g}")
        elif body_tie == "floating":
            body_node = f"b{eidx}"
        else:
            raise ValueError(f"Unsupported body_tie: {body_tie}")

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
def cross_entropy_from_outputs(V: np.ndarray, y: int, temp: float) -> Tuple[float, float]:
    """
    Returns (CE loss, p_true) for logits = V/temp.
    CE = -log p_y, with p from softmax.
    """
    if temp <= 0.0:
        return float("inf"), float("nan")
    z = np.asarray(V, dtype=float) / float(temp)
    p = softmax_logits(z)
    py = float(p[int(y)])
    if not np.isfinite(py):
        return float("inf"), float("nan")
    if py < 1e-12:
        py = 1e-12
    return float(-np.log(py)), float(py)


def pred_label(V: np.ndarray) -> int:
    return int(np.argmax(np.asarray(V, dtype=float)))


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
    tr_ce = _load("0_train_ce.npy")
    va_acc = _load("0_val_acc.npy")
    va_ce = _load("0_val_ce.npy")

    ep_total = _load("0_epoch_total_s.npy")
    ep_free = _load("0_epoch_free_s.npy")
    ep_clamp = _load("0_epoch_clamp_s.npy")
    ep_upd = _load("0_epoch_update_s.npy")

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

    if tr_ce is not None and va_ce is not None:
        plt.figure()
        plt.plot(np.arange(len(va_ce)), va_ce, label="val CE")
        plt.plot(np.arange(1, len(tr_ce) + 1), tr_ce, label="train CE")
        plt.xlabel("epoch")
        plt.ylabel("cross entropy")
        plt.title("Cross entropy vs epoch")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(run_dir / "learning_curves_ce.png", dpi=160)
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


# -------------------------
# Main
# -------------------------
def main():
    args = parse_args()
    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)

    epochs = int(args.epochs)
    if epochs < 1:
        raise ValueError("--epochs must be >= 1")

    gamma = float(args.gamma)
    delta = float(args.delta)
    temp = float(args.softmax_temp)
    vmin = float(args.input_vmin)
    vmax = float(args.input_vmax)
    if vmax <= vmin:
        raise ValueError("--input-vmax must be > --input-vmin")
    if temp <= 0.0:
        raise ValueError("--softmax-temp must be > 0")

    vminus_val = float(args.vminus)
    vplus_val = float(args.vplus)
    solver = str(args.solver).lower()
    body_res = float(RS_CLAMP)
    body_tie = str(args.body_tie)
    vg_init_mode = str(args.vg_init)
    vg_init_lo = float(args.vg_init_lo)
    vg_init_hi = float(args.vg_init_hi)
    vg_init_single = float(args.vg_init_fixed)
    if vg_init_mode == "random" and vg_init_hi <= vg_init_lo:
        raise ValueError("--vg-init-hi must be > --vg-init-lo for random init")

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
    if not TOPOLOGY_PATH.exists():
        raise FileNotFoundError(f"Topology file not found: {TOPOLOGY_PATH}")
    topo = load_topology_npz(TOPOLOGY_PATH)
    if topo.Nin != Nin:
        raise ValueError(f"Topology Nin={topo.Nin} does not match data Nin={Nin}")
    K = topo.K

    # Diagnostics use the full test set
    test_n = len(test_x)

    # Init weights (unique VG per edge)
    if vg_init_mode == "fixed":
        vg_unique = np.full((topo.num_edges,), vg_init_single, dtype=float)
    else:
        vg_unique = np.random.uniform(vg_init_lo, vg_init_hi, size=(topo.num_edges,)).astype(float)

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
        f"seed={seed} gamma={gamma} delta={delta} T={temp} "
        f"in=[{vmin},{vmax}] Nin={Nin} K={K} rails=[{vminus_val},{vplus_val}] "
        f"solver={solver} body_tie={body_tie} body_res={body_res} rs_clamp={RS_CLAMP} "
        f"vg_init={vg_init_mode} "
        f"epochs={epochs} "
        f"device_include={DEVICE_LIB_PATH} subckt={DEVICE_SUBCKT} "
        f"topology={TOPOLOGY_PATH.name}"
    )

    print("=== RUN START (scikit_digits_dense_io_xent) ===", flush=True)
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
        body_tie=body_tie,
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
        "delta": delta,
        "softmax_temp": temp,
        "epochs": epochs,
        "input_vmin": vmin,
        "input_vmax": vmax,
        "rails": {"vminus": vminus_val, "vplus": vplus_val},
        "solver": solver,
        "body_tie": body_tie,
        "body_res": body_res,
        "rs_clamp": RS_CLAMP,
        "vg_init": {
            "mode": vg_init_mode,
            "lo": vg_init_lo,
            "hi": vg_init_hi,
            "fixed": vg_init_single,
        },
        "device": {
            "include_path": DEVICE_LIB_PATH,
            "subckt": DEVICE_SUBCKT,
        },
        "topology": {
            "path": str(TOPOLOGY_PATH),
            "Nin": topo.Nin,
            "out": topo.K,
            "edges": topo.num_edges,
            "meta": topo.meta,
        },
        "loss": "cross_entropy",
        "diagnostics": {"vout_saved": "test"},
        "diode_source_to_vplus": False,
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

    np.save(run_dir / "test_x.npy", np.asarray(test_x, dtype=float))
    np.save(run_dir / "test_y.npy", np.asarray(test_y, dtype=int))

    # Ngspice
    ng = NgSpiceShared(send_data=False)
    ng.load_circuit(netlist)

    # Nodes list (numeric nodes only)
    net_nodes = [topo.negref, topo.posref] + topo.out_nodes.tolist() + topo.input_nodes.tolist()
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
    val_ce_hist: List[float] = []
    tr_acc_hist: List[float] = []
    tr_ce_hist: List[float] = []

    ep_total_s: List[float] = []
    ep_free_s: List[float] = []
    ep_clamp_s: List[float] = []
    ep_update_s: List[float] = []

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

        vout_test = np.full((test_n, K), np.nan, dtype=float) if test_n > 0 else np.zeros((0, K), dtype=float)

        reloads = 0
        nonfinite = 0
        confusion = np.zeros((K, K), dtype=int)

        py_list: List[float] = []

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

            ytrue = int(yt)
            pred = pred_label(Vout)
            correct += int(pred == ytrue)
            total += 1
            if 0 <= ytrue < K:
                confusion[ytrue, pred] += 1

            ce, py = cross_entropy_from_outputs(Vout, ytrue, temp=temp)
            loss_sum += float(ce)
            count += 1
            if np.isfinite(py):
                py_list.append(float(py))

            vout_test[i, :] = Vout

        if test_n > 0:
            np.save(run_dir / f"0_vout_test_epoch{epoch}.npy", vout_test)
            np.save(run_dir / f"0_val_confusion_epoch{epoch}.npy", confusion)

        diag: Dict[str, float] = {}
        diag["val_reloads"] = float(reloads)
        diag["val_nonfinite"] = float(nonfinite)
        diag["val_py_true_mean"] = float(np.mean(py_list)) if py_list else float("nan")

        acc = (correct / total) if total else float("nan")
        loss = (loss_sum / count) if count else float("nan")
        return float(acc), float(loss), diag

    # Epoch 0 validation
    v0, ce0, diag0 = eval_free_metrics(epoch=0)
    val_acc_hist.append(v0)
    val_ce_hist.append(ce0)
    np.save(run_dir / "0_val_acc.npy", np.asarray(val_acc_hist, dtype=float))
    np.save(run_dir / "0_val_ce.npy", np.asarray(val_ce_hist, dtype=float))
    (run_dir / "0_diag_epoch0.json").write_text(json.dumps(diag0, indent=2))
    print(
        f"[epoch 0] {cfg_str} | VAL acc={v0:.4f} CE={ce0:.6f} py_true_mean={diag0.get('val_py_true_mean', float('nan')):.4f}",
        flush=True,
    )

    for ep in range(1, epochs + 1):
        t_ep0 = time.time()
        order = np.arange(len(train_x))
        np.random.shuffle(order)

        train_correct = 0
        train_total = 0
        ce_sum = 0.0
        ce_count = 0

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

            pred = pred_label(Vout)
            train_correct += int(pred == ytrue)
            train_total += 1

            ce, _ = cross_entropy_from_outputs(Vout, ytrue, temp=temp)
            ce_sum += float(ce)
            ce_count += 1

            # Clamp (TRUE K-class CE)
            clamp0 = time.time()
            alter_outputs_xent(ng, K=K, y=ytrue, Vout_free=Vout, delta=delta, temp=temp)
            t_clamp += float(time.time() - clamp0)

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

            # Update (unchanged)
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
        tr_ce = (ce_sum / ce_count) if ce_count else float("nan")
        tr_acc_hist.append(float(tr_acc))
        tr_ce_hist.append(float(tr_ce))

        reload_free_hist.append(int(reload_free))
        reload_clamp_hist.append(int(reload_clamp))
        nonfinite_free_hist.append(int(nonfinite_free))
        nonfinite_clamp_hist.append(int(nonfinite_clamp))

        v_acc, v_ce, diag = eval_free_metrics(epoch=ep)
        val_acc_hist.append(float(v_acc))
        val_ce_hist.append(float(v_ce))

        ep_total = float(time.time() - t_ep0)
        ep_total_s.append(ep_total)
        ep_free_s.append(float(t_free))
        ep_clamp_s.append(float(t_clamp))
        ep_update_s.append(float(t_update))

        # Save arrays each epoch
        np.save(run_dir / "0_train_acc.npy", np.asarray(tr_acc_hist, dtype=float))
        np.save(run_dir / "0_train_ce.npy", np.asarray(tr_ce_hist, dtype=float))
        np.save(run_dir / "0_val_acc.npy", np.asarray(val_acc_hist, dtype=float))
        np.save(run_dir / "0_val_ce.npy", np.asarray(val_ce_hist, dtype=float))

        np.save(run_dir / "0_epoch_total_s.npy", np.asarray(ep_total_s, dtype=float))
        np.save(run_dir / "0_epoch_free_s.npy", np.asarray(ep_free_s, dtype=float))
        np.save(run_dir / "0_epoch_clamp_s.npy", np.asarray(ep_clamp_s, dtype=float))
        np.save(run_dir / "0_epoch_update_s.npy", np.asarray(ep_update_s, dtype=float))

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
                "delta": delta,
                "softmax_temp": temp,
                "input_vmin": vmin,
                "input_vmax": vmax,
                "rails": [vminus_val, vplus_val],
                "solver": solver,
                "body_tie": body_tie,
                "body_res": body_res,
                "rs_clamp": RS_CLAMP,
                "vg_init": {
                    "mode": vg_init_mode,
                    "lo": vg_init_lo,
                    "hi": vg_init_hi,
                    "fixed": vg_init_single,
                },
                "epochs": epochs,
                "device_include_path": DEVICE_LIB_PATH,
                "device_subckt": DEVICE_SUBCKT,
                "loss": "cross_entropy",
            },
            "train": {
                "acc": float(tr_acc),
                "ce": float(tr_ce),
                "n_free": int(n_free),
                "n_clamp": int(n_clamp),
                "reload_free": int(reload_free),
                "reload_clamp": int(reload_clamp),
                "nonfinite_free": int(nonfinite_free),
                "nonfinite_clamp": int(nonfinite_clamp),
            },
            "val": {"acc": float(v_acc), "ce": float(v_ce), **{k: float(v) for k, v in diag.items()}},
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
            f"[epoch {ep}/{epochs}] {cfg_str} | "
            f"TRAIN acc={tr_acc:.4f} CE={tr_ce:.6f} "
            f"free={n_free} clamp={n_clamp} "
            f"reloadF={reload_free} reloadC={reload_clamp} nonfiniteF={nonfinite_free} nonfiniteC={nonfinite_clamp}",
            flush=True,
        )
        print(
            f"[epoch {ep}/{epochs}] {cfg_str} | "
            f"VAL acc={v_acc:.4f} CE={v_ce:.6f} "
            f"py_true_mean={diag.get('val_py_true_mean', float('nan')):.4f} | "
            f"timing total={ep_total:.2f}s free={t_free:.2f}s clamp={t_clamp:.2f}s upd={t_update:.2f}s",
            flush=True,
        )

        try:
            save_plots(run_dir)
        except Exception:
            pass

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

    print("=== RUN END (scikit_digits_dense_io_xent) ===", flush=True)
    try:
        log_f.flush()
        log_f.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
