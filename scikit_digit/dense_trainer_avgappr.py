#!/usr/bin/env python3
"""
Scikit Digits — dense input→output NMOS network (no hidden nodes)
MSE training with two-phase (free / clamp) ngspice runs.

Behavior:
  - Dataset: sklearn digits 8x8 (Nin=64).
  - Topology: loaded from NPZ (digits_8x8_dense_io_x1 by default).
  - Init VG: random in [--vg-init-lo, --vg-init-hi] or fixed --vg-init-fixed via --vg-init.
  - Training is two-phase per sample:
      1) Free phase:   RS all = RS_FREE, run .op, read Vout + all nodes needed for updates
      2) Clamp phase:  set VOUT sources + RS to implement a "target" output, run .op, read nodes
      3) Update:       dVG_e = -gamma * ( (dV_e^C)^2 - (dV_e^F)^2 ), clip to [VG_CLIP_LO, VG_CLIP_HI]

Epoch semantics (as requested):
  - Epoch 0: NO training. Baseline validation only, using one-hot metric (argmax + MSE to one-hot).
  - Epoch 1: First training epoch uses ONE-HOT targets:
        target[y] = 0.5 V, target[others] = 0 V (hard clamp).
        Prediction: argmax(Vout_free)
  - Epoch >= 2: "Averaging" mode:
        At start of each epoch:
          - Compute proto_in[k] = mean(train inputs with label k)
          - Measure proto_out[k] = Vout_free(proto_in[k])
        During training and validation:
          - Prediction: nearest proto_out (L2)
          - Target: proto_out[y]
          - Clamp: soft nudge toward target:
                VOUT = Vout_free + delta * (target - Vout_free)
            with RS forced to RS_CLAMP on ALL outputs for the clamp phase.

Device model:
  - External include file defines:
      .model ncg nmos (...)
      .subckt NMOSWRAP D G S B
         M0 D G S B ncg ...
      .ends
  - Body tie selectable via --body-tie (source, ground, floating); default ground.
  - Body resistor fixed to RS_CLAMP.
  - Source->V+ diode is OFF (not instantiated).

Paths (repo-relative):
  - device_model/nmos_lvl1_ald1106.lib
  - scikit_digit/topology/digits_8x8_dense_io_x1.npz
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
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

from topology.topology_io import Topology, load_topology_npz, validate_topology


# -------------------------
# Fixed paths / constants
# -------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DEVICE_LIB_PATH = str(REPO_ROOT / "device_model" / "nmos_lvl1_ald1106.lib")
DEVICE_SUBCKT = "NMOSWRAP"

TOPOLOGY_PATH = REPO_ROOT / "scikit_digit" / "topology" / "digits_8x8_dense_io_x1.npz"
#TOPOLOGY_PATH = REPO_ROOT / "scikit_digit" / "topology" / "digits_8x8_dense_io_x1_pruned_vglt0p75_epoch20_run20260110-235531.npz"

VG_CLIP_LO, VG_CLIP_HI = 0.4, 8.0
VG_INIT_SINGLE = 2.0

RS_FREE = 1e9
RS_CLAMP = 10.0

ONEHOT_HI_V = 0.5  # epoch-1 one-hot clamp target (requested)


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
        description="Scikit Digits — dense input→output NMOS network, MSE, ngspice"
    )
    p.add_argument("seed", type=int, nargs="?", default=0)

    p.add_argument("--epochs", type=int, default=20, help="Total epochs (default 20).")

    p.add_argument("--gamma", type=float, default=0.3)
    p.add_argument("--delta", type=float, default=0.3, help="clamp nudge delta (unitless scale, typically 0..1)")

    # Input mapping (do not sweep, but keep options)
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
    p.add_argument("--vg-init-lo", type=float, default=1.0)
    p.add_argument("--vg-init-hi", type=float, default=3.0)
    p.add_argument("--vg-init-fixed", type=float, default=VG_INIT_SINGLE)

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


def restore_gate_voltages(ng: NgSpiceShared, vg_unique: np.ndarray):
    exec_chunked(ng, [f"alter VG{i} dc = {float(vg_unique[i]):.16f}" for i in range(vg_unique.size)])


def onehot_target(y: int, K: int, hi: float = ONEHOT_HI_V) -> np.ndarray:
    t = np.zeros((K,), dtype=float)
    t[y] = float(hi)
    return t


def alter_outputs_hard_target(ng: NgSpiceShared, target: np.ndarray, K: int):
    """
    Hard clamp ALL outputs through RS=RS_CLAMP and set VOUTj exactly to target[j].
    """
    cmds: List[str] = []
    for i in range(1, K + 1):
        cmds.append(f"alter RS{i} {RS_CLAMP:.6g}")
    for j in range(K):
        cmds.append(f"alter VOUT{j} dc = {float(target[j]):.16f}")
    exec_chunked(ng, cmds)


def alter_outputs_mse_to_target(
    ng: NgSpiceShared,
    Vout_free: np.ndarray,
    target: np.ndarray,
    delta: float,
    K: int,
):
    """
    Soft clamp ALL outputs through RS=RS_CLAMP:
      VOUT = Vfree + delta*(target - Vfree)
    """
    delta = float(delta)
    cmds: List[str] = []
    for i in range(1, K + 1):
        cmds.append(f"alter RS{i} {RS_CLAMP:.6g}")
    for j in range(K):
        Vc = float(Vout_free[j] + delta * (target[j] - Vout_free[j]))
        cmds.append(f"alter VOUT{j} dc = {Vc:.16f}")
    exec_chunked(ng, cmds)


# -------------------------
# Netlist
# -------------------------
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
    lines.append(".title scikit_digits_dense_io_mse")
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

        # NMOSWRAP pins: D G S B
        lines.append(f"X{eidx} {D} {gate_node} {S} {body_node} {DEVICE_SUBCKT}")

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
# MSE / prototypes
# -------------------------
def mse_to_target(V: np.ndarray, T: np.ndarray) -> float:
    d = (V - T).astype(float)
    return float(np.mean(d * d))


def predict_argmax(Vout: np.ndarray) -> int:
    return int(np.argmax(Vout))


def predict_nearest_proto(Vout: np.ndarray, proto_out: np.ndarray) -> int:
    # proto_out: (K,K) in this setting, but keep generic (K,dim)
    # ignore NaN prototypes
    dif = proto_out - Vout[None, :]
    # distances where any nan in row -> inf
    bad = ~np.all(np.isfinite(proto_out), axis=1)
    dist = np.sum(dif * dif, axis=1)
    dist[bad] = np.inf
    if not np.any(np.isfinite(dist)):
        return int(np.argmax(Vout))
    return int(np.argmin(dist))


def compute_input_prototypes(train_x: List[np.ndarray], train_y: List[int], K: int) -> np.ndarray:
    Nin = int(train_x[0].size)
    proto_in = np.full((K, Nin), np.nan, dtype=float)
    counts = np.zeros((K,), dtype=int)
    acc = np.zeros((K, Nin), dtype=float)
    for x, y in zip(train_x, train_y):
        yy = int(y)
        acc[yy] += np.asarray(x, dtype=float)
        counts[yy] += 1
    for k in range(K):
        if counts[k] > 0:
            proto_in[k] = acc[k] / float(counts[k])
    return proto_in


def measure_output_prototypes(
    ng: NgSpiceShared,
    topo: Topology,
    netlist: str,
    vg_unique: np.ndarray,
    proto_in: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, float]]:
    K = int(topo.K)
    proto_out = np.full((K, K), np.nan, dtype=float)
    reloads = 0
    nonfinite = 0
    ok_count = 0

    mk_free_all(ng, K)

    for k in range(K):
        xk = proto_in[k]
        if not np.all(np.isfinite(xk)):
            continue
        alter_inputs_named(ng, xk)
        ok, _, data, _ = run_and_read(ng, {"out": topo.out_nodes})
        if (not ok) or (data is None):
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

        proto_out[k, :] = Vout
        ok_count += 1

    diag = {
        "proto_ok": float(ok_count),
        "proto_reloads": float(reloads),
        "proto_nonfinite": float(nonfinite),
    }
    return proto_out, diag


def save_plots(run_dir: Path):
    def _load(name: str):
        p = run_dir / name
        return np.load(p) if p.exists() else None

    tr_acc = _load("0_train_acc.npy")
    tr_mse = _load("0_train_mse.npy")
    va_acc = _load("0_val_acc.npy")
    va_mse = _load("0_val_mse.npy")

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

    if tr_mse is not None and va_mse is not None:
        plt.figure()
        plt.plot(np.arange(len(va_mse)), va_mse, label="val mse")
        plt.plot(np.arange(1, len(tr_mse) + 1), tr_mse, label="train mse")
        plt.xlabel("epoch")
        plt.ylabel("mse")
        plt.title("MSE vs epoch")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(run_dir / "learning_curves_mse.png", dpi=160)
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
# Validation
# -------------------------
def eval_free_metrics(
    ng: NgSpiceShared,
    topo: Topology,
    netlist: str,
    vg_unique: np.ndarray,
    test_x: List[np.ndarray],
    test_y: List[int],
    K: int,
    run_dir: Path,
    epoch: int,
    mode: str,
    proto_out: Optional[np.ndarray] = None,
) -> Tuple[float, float, Dict[str, float]]:
    """
    mode="onehot": prediction=argmax, target=onehot(0.5V), mse computed to onehot
    mode="proto":  prediction=nearest proto_out, target=proto_out[y], mse computed to proto target
    """
    mk_free_all(ng, K)

    correct = 0
    total = 0
    mse_sum = 0.0
    mse_count = 0
    reloads = 0
    nonfinite = 0

    test_n = len(test_x)
    vout_test = np.full((test_n, K), np.nan, dtype=float) if test_n > 0 else np.zeros((0, K), dtype=float)

    for i, (xt, yt) in enumerate(zip(test_x, test_y)):
        alter_inputs_named(ng, xt)
        ok, _, data, _ = run_and_read(ng, {"out": topo.out_nodes})
        if (not ok) or (data is None):
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
        if mode == "onehot":
            yhat = predict_argmax(Vout)
            target = onehot_target(ytrue, K)
        elif mode == "proto":
            if proto_out is None:
                raise ValueError("proto_out must be provided for mode='proto'")
            yhat = predict_nearest_proto(Vout, proto_out)
            target = np.asarray(proto_out[ytrue, :], dtype=float)
            if not np.all(np.isfinite(target)):
                # cannot score MSE; still count acc
                target = None
        else:
            raise ValueError(f"unknown mode: {mode}")

        correct += int(yhat == ytrue)
        total += 1

        if target is not None:
            l = mse_to_target(Vout, target)
            if np.isfinite(l):
                mse_sum += float(l)
                mse_count += 1

        vout_test[i, :] = Vout

    if test_n > 0:
        np.save(run_dir / f"0_vout_test_epoch{epoch}.npy", vout_test)

    acc = (correct / total) if total else float("nan")
    mse = (mse_sum / mse_count) if mse_count else float("nan")

    diag = {
        "val_mode": mode,
        "val_reloads": float(reloads),
        "val_nonfinite": float(nonfinite),
        "val_mse_count": float(mse_count),
    }
    return float(acc), float(mse), diag


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

    vmin = float(args.input_vmin)
    vmax = float(args.input_vmax)
    if vmax <= vmin:
        raise ValueError("--input-vmax must be > --input-vmin")

    vminus_val = float(args.vminus)
    vplus_val = float(args.vplus)

    solver = str(args.solver).lower()
    body_tie = str(args.body_tie)
    body_res = float(RS_CLAMP)

    vg_init_mode = str(args.vg_init)
    vg_init_lo = float(args.vg_init_lo)
    vg_init_hi = float(args.vg_init_hi)
    vg_init_single = float(args.vg_init_fixed)
    if vg_init_mode == "random" and vg_init_hi <= vg_init_lo:
        raise ValueError("--vg-init-hi must be > --vg-init-lo for random init")

    # Dataset
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

    Nin = int(train_x[0].size)
    if not TOPOLOGY_PATH.exists():
        raise FileNotFoundError(f"Topology file not found: {TOPOLOGY_PATH}")
    topo = load_topology_npz(TOPOLOGY_PATH)
    validate_topology(topo)

    if topo.Nin != Nin:
        raise ValueError(f"Topology Nin={topo.Nin} does not match data Nin={Nin}")
    K = int(topo.K)

    # Init VG
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
        f"seed={seed} gamma={gamma} delta={delta} "
        f"in=[{vmin},{vmax}] Nin={Nin} K={K} rails=[{vminus_val},{vplus_val}] "
        f"solver={solver} body_tie={body_tie} body_res={body_res} rs_clamp={RS_CLAMP} "
        f"vg_init={vg_init_mode} "
        f"epochs={epochs} "
        f"onehot_hi={ONEHOT_HI_V} "
        f"device_include={DEVICE_LIB_PATH} subckt={DEVICE_SUBCKT} "
        f"topology={TOPOLOGY_PATH.name}"
    )

    print("=== RUN START (scikit_digits_dense_io_mse_avgappr) ===", flush=True)
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
        "epochs": epochs,
        "epoch0": "validation_only_onehot",
        "epoch1": "train_onehot_hard_clamp",
        "epoch2plus": "train_proto_soft_clamp",
        "onehot_hi_v": ONEHOT_HI_V,
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
        "device": {"include_path": DEVICE_LIB_PATH, "subckt": DEVICE_SUBCKT},
        "topology": {
            "path": str(TOPOLOGY_PATH),
            "Nin": topo.Nin,
            "out": topo.K,
            "edges": topo.num_edges,
            "meta": topo.meta,
        },
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

    np.save(run_dir / "test_x.npy", np.asarray(test_x, dtype=float))
    np.save(run_dir / "test_y.npy", np.asarray(test_y, dtype=int))

    # Ngspice
    ng = NgSpiceShared(send_data=False)
    ng.load_circuit(netlist)

    # Nodes list for updates (must cover all D,S nodes)
    net_nodes = [topo.negref, topo.posref] + topo.out_nodes.tolist() + topo.input_nodes.tolist()
    nodes_list = np.asarray(sorted(set(net_nodes)), dtype=int)

    index_of = np.full(nodes_list.max() + 1, -1, dtype=int)
    index_of[nodes_list] = np.arange(nodes_list.size, dtype=int)

    eD = topo.edges_D
    eS = topo.edges_S

    # Graph export
    try:
        G = nx.DiGraph()
        G.add_nodes_from(nodes_list.tolist())
        for d, s in zip(eD.tolist(), eS.tolist()):
            G.add_edge(int(d), int(s))
        nx.write_graphml(G, str(run_dir / "0.graphml"))
    except Exception:
        pass

    # Histories
    val_acc_hist: List[float] = []
    val_mse_hist: List[float] = []
    tr_acc_hist: List[float] = []
    tr_mse_hist: List[float] = []

    ep_total_s: List[float] = []
    ep_free_s: List[float] = []
    ep_clamp_s: List[float] = []
    ep_update_s: List[float] = []

    reload_free_hist: List[int] = []
    reload_clamp_hist: List[int] = []
    nonfinite_free_hist: List[int] = []
    nonfinite_clamp_hist: List[int] = []

    # -------------------------
    # Epoch 0: baseline validation only (one-hot metric)
    # -------------------------
    v0, m0, diag0 = eval_free_metrics(
        ng=ng,
        topo=topo,
        netlist=netlist,
        vg_unique=vg_unique,
        test_x=test_x,
        test_y=test_y,
        K=K,
        run_dir=run_dir,
        epoch=0,
        mode="onehot",
        proto_out=None,
    )
    val_acc_hist.append(float(v0))
    val_mse_hist.append(float(m0))
    np.save(run_dir / "0_val_acc.npy", np.asarray(val_acc_hist, dtype=float))
    np.save(run_dir / "0_val_mse.npy", np.asarray(val_mse_hist, dtype=float))
    (run_dir / "0_diag_epoch0.json").write_text(json.dumps(diag0, indent=2))
    print(f"[epoch 0] {cfg_str} | VAL(onehot) acc={v0:.4f} mse={m0:.6g}", flush=True)

    # -------------------------
    # Epochs 1..epochs
    #   ep==1: one-hot hard clamp targets
    #   ep>=2: averaging/prototype mode
    # -------------------------
    for ep in range(1, epochs + 1):
        t_ep0 = time.time()
        order = np.arange(len(train_x))
        np.random.shuffle(order)

        train_correct = 0
        train_total = 0
        mse_sum = 0.0
        mse_count = 0
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

        if ep == 1:
            mode = "onehot"
            proto_out = None
        else:
            mode = "proto"
            proto_in = compute_input_prototypes(train_x, train_y, K)
            proto_out, proto_diag = measure_output_prototypes(
                ng=ng,
                topo=topo,
                netlist=netlist,
                vg_unique=vg_unique,
                proto_in=proto_in,
            )
            np.save(run_dir / f"0_proto_in_epoch{ep}.npy", proto_in)
            np.save(run_dir / f"0_proto_out_epoch{ep}.npy", proto_out)
            (run_dir / f"0_proto_diag_epoch{ep}.json").write_text(json.dumps(proto_diag, indent=2))

        for idx in order:
            ytrue = int(train_y[idx])

            mk_free_all(ng, K)

            # Free
            alter_inputs_named(ng, train_x[idx])
            ok, dt, data, _ = run_and_read(ng, {"out": topo.out_nodes, "nodes": nodes_list.tolist()})
            t_free += float(dt)
            n_free += 1

            if (not ok) or (data is None):
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

            # Target + prediction
            if mode == "onehot":
                target = onehot_target(ytrue, K)
                yhat = predict_argmax(Vout)
            else:
                assert proto_out is not None
                target = np.asarray(proto_out[ytrue, :], dtype=float)
                if not np.all(np.isfinite(target)):
                    skipped += 1
                    continue
                yhat = predict_nearest_proto(Vout, proto_out)

            train_correct += int(yhat == ytrue)
            train_total += 1

            l = mse_to_target(Vout, target)
            if np.isfinite(l):
                mse_sum += float(l)
                mse_count += 1

            # Clamp
            if mode == "onehot":
                alter_outputs_hard_target(ng, target=target, K=K)
            else:
                alter_outputs_mse_to_target(ng, Vout_free=Vout, target=target, delta=delta, K=K)

            ok2, dt2, data2, _ = run_and_read(ng, {"nodes": nodes_list.tolist()})
            t_clamp += float(dt2)
            n_clamp += 1

            if (not ok2) or (data2 is None):
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
                nv = float(vg_unique[uid] + float(update[uid]))
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
        tr_mse = (mse_sum / mse_count) if mse_count else float("nan")
        tr_acc_hist.append(float(tr_acc))
        tr_mse_hist.append(float(tr_mse))

        # Validation
        if mode == "onehot":
            v_acc, v_mse, diag = eval_free_metrics(
                ng=ng,
                topo=topo,
                netlist=netlist,
                vg_unique=vg_unique,
                test_x=test_x,
                test_y=test_y,
                K=K,
                run_dir=run_dir,
                epoch=ep,
                mode="onehot",
                proto_out=None,
            )
        else:
            v_acc, v_mse, diag = eval_free_metrics(
                ng=ng,
                topo=topo,
                netlist=netlist,
                vg_unique=vg_unique,
                test_x=test_x,
                test_y=test_y,
                K=K,
                run_dir=run_dir,
                epoch=ep,
                mode="proto",
                proto_out=proto_out,
            )

        val_acc_hist.append(float(v_acc))
        val_mse_hist.append(float(v_mse))

        ep_total = float(time.time() - t_ep0)
        ep_total_s.append(ep_total)
        ep_free_s.append(float(t_free))
        ep_clamp_s.append(float(t_clamp))
        ep_update_s.append(float(t_update))

        reload_free_hist.append(int(reload_free))
        reload_clamp_hist.append(int(reload_clamp))
        nonfinite_free_hist.append(int(nonfinite_free))
        nonfinite_clamp_hist.append(int(nonfinite_clamp))

        # Save arrays
        np.save(run_dir / "0_train_acc.npy", np.asarray(tr_acc_hist, dtype=float))
        np.save(run_dir / "0_train_mse.npy", np.asarray(tr_mse_hist, dtype=float))
        np.save(run_dir / "0_val_acc.npy", np.asarray(val_acc_hist, dtype=float))
        np.save(run_dir / "0_val_mse.npy", np.asarray(val_mse_hist, dtype=float))

        np.save(run_dir / "0_epoch_total_s.npy", np.asarray(ep_total_s, dtype=float))
        np.save(run_dir / "0_epoch_free_s.npy", np.asarray(ep_free_s, dtype=float))
        np.save(run_dir / "0_epoch_clamp_s.npy", np.asarray(ep_clamp_s, dtype=float))
        np.save(run_dir / "0_epoch_update_s.npy", np.asarray(ep_update_s, dtype=float))

        np.save(run_dir / "0_reload_free.npy", np.asarray(reload_free_hist, dtype=int))
        np.save(run_dir / "0_reload_clamp.npy", np.asarray(reload_clamp_hist, dtype=int))
        np.save(run_dir / "0_nonfinite_free.npy", np.asarray(nonfinite_free_hist, dtype=int))
        np.save(run_dir / "0_nonfinite_clamp.npy", np.asarray(nonfinite_clamp_hist, dtype=int))

        np.save(run_dir / f"0_vg_unique_epoch{ep}.npy", vg_unique.copy())

        summary = {
            "epoch": int(ep),
            "mode": mode,
            "train": {
                "acc": float(tr_acc),
                "mse": float(tr_mse),
                "n_free": int(n_free),
                "n_clamp": int(n_clamp),
                "skipped": int(skipped),
                "reload_free": int(reload_free),
                "reload_clamp": int(reload_clamp),
                "nonfinite_free": int(nonfinite_free),
                "nonfinite_clamp": int(nonfinite_clamp),
            },
            "val": {"acc": float(v_acc), "mse": float(v_mse), **diag},
            "timing_s": {
                "epoch_total": float(ep_total),
                "train_free": float(t_free),
                "train_clamp": float(t_clamp),
                "train_update": float(t_update),
            },
        }
        (run_dir / f"0_epoch_summary_epoch{ep}.json").write_text(json.dumps(summary, indent=2))
        (run_dir / f"0_diag_epoch{ep}.json").write_text(json.dumps(diag, indent=2))

        print(
            f"[epoch {ep}/{epochs}] {cfg_str} | "
            f"TRAIN({mode}) acc={tr_acc:.4f} mse={tr_mse:.6g} "
            f"free={n_free} clamp={n_clamp} skipped={skipped} "
            f"reloadF={reload_free} reloadC={reload_clamp} nonfiniteF={nonfinite_free} nonfiniteC={nonfinite_clamp}",
            flush=True,
        )
        print(
            f"[epoch {ep}/{epochs}] {cfg_str} | "
            f"VAL({mode}) acc={v_acc:.4f} mse={v_mse:.6g} | "
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

    print("=== RUN END (scikit_digits_dense_io_mse_avgappr) ===", flush=True)
    try:
        log_f.flush()
        log_f.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
