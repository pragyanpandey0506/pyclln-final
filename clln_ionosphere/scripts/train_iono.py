#!/usr/bin/env python3
"""
Scikit Digits — dense input→output NMOS network (no hidden nodes)
Hinge loss training (SGD, no batching) with two-phase (free / clamp) ngspice runs.

Key behavior:
  - Dataset: sklearn digits 8x8 (default; no crop options).
  - Topology: loaded from NPZ (digits_8x8_dense_io_x3), with parallel edges per input/output.
  - Init VG: random in [VG_INIT_LO, VG_INIT_HI] or fixed VG_INIT_SINGLE via --vg-init.
  - Per-sample training:
      1) Free phase:   RS all = 1e12 (outputs effectively unclamped), run .op
      2) Compute hinge. If inactive, skip clamp + update.
      3) Clamp phase: clamp only y and rival outputs via RS=RS_CLAMP and nudged VOUT sources.
      4) Update: dVG_e = -gamma * ( (dV_e^C)^2 - (dV_e^F)^2 ), clip to [0.4, 10.0]
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
      scikit_digit/topology/digits_8x8_dense_io_x3.npz
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

from simple_topology_io import Topology, load_topology_npz

def load_ionosphere_from_dir(data_dir: str):
    candidates = [
        os.path.join(data_dir, "ionosphere.data"),
        os.path.join(data_dir, "ionosphere.data.txt"),
        os.path.join(data_dir, "ionosphere.csv"),
    ]
    path = None
    for c in candidates:
        if os.path.exists(c):
            path = c
            break
    if path is None:
        raise FileNotFoundError(f"Could not find ionosphere.data in {data_dir}. Tried: {candidates}")

    X_list, y_list = [], []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 35:
                raise ValueError(f"Expected 35 columns (34+label), got {len(parts)}")
            X_list.append([float(v) for v in parts[:34]])
            lab = parts[34].strip()
            if lab not in ("g", "b"):
                raise ValueError(f"Unexpected label: {lab}")
            # Map to {0,1} so K=2 works with your argmax + confusion matrix
            y_list.append(1 if lab == "g" else 0)

    X = np.asarray(X_list, dtype=np.float64)  # shape (351,34) in [-1,1]
    y = np.asarray(y_list, dtype=int)         # 0/1
    return X, y, path

# -------------------------
# Fixed device include (external)
# -------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DEVICE_LIB_PATH = str(
    REPO_ROOT / "dima_experiments" / "paper_2" / "device_model" / "nmos_lvl1_ald1106.lib"
)
DEFAULT_IONOSPHERE_DIR = str(REPO_ROOT / "clln_ionosphere" / "data" / "ionosphere")
DEFAULT_DEVICE_MODE = "nmos"     # nmos / pmos / diode
DEFAULT_DEVICE_MODEL = "ncg"     # model name inside the .lib (for your ald1106 wrapper)
DEFAULT_DEVICE_SUBCKT = "NMOSWRAP"  # used only for "subckt" mode (your old file)
#TOPOLOGY_PATH = REPO_ROOT / "scikit_digit" / "topology" / "digits_8x8_dense_io_x1.npz"
TOPOLOGY_PATH = Path(__file__).resolve().parent / "ionosphere_34_dense_io_x3.npz"
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
        description="Scikit Digits — dense input→output NMOS network, hinge loss, ngspice"
    )
    p.add_argument("seed", type=int, nargs="?", default=0)
    p.add_argument("--topology", type=str, default=None,
                   help="Path to topology NPZ file (default: ionosphere_34_dense_io_x3.npz).")

    # Training epochs
    p.add_argument("--epochs", type=int, default=20, help="Total epochs (default 20).")

    # Learning hyperparams
    p.add_argument("--gamma", type=float, default=0.3)
    p.add_argument("--gamma-final", type=float, default=-1.0,
                   help="If >= 0, linearly decay gamma -> gamma-final over all epochs.")
    p.add_argument("--gamma-decay-rate", type=float, default=-1.0,
                   help="If >= 0, multiply gamma by this factor each epoch (exponential decay).")
    p.add_argument("--margin", type=float, default=0.02, help="hinge margin m (volts)")
    p.add_argument("--delta", type=float, default=0.05, help="clamp nudge delta (volts)")
    p.add_argument("--delta-final", type=float, default=-1.0,
                   help="If >= 0, linearly decay delta -> delta-final over all epochs.")

    # Input mapping
    p.add_argument("--input-vmin", type=float, default=0.0)
    p.add_argument("--input-vmax", type=float, default=1.0)
    
    # Device selection (new)
    p.add_argument("--device-mode", type=str,
                   choices=["subckt", "nmos", "pmos", "diode"],
                   default="subckt",
                   help="subckt: use X...NMOSWRAP; nmos/pmos: use M... primitive; diode: use D... primitive")

    p.add_argument("--device-lib", type=str, default=DEFAULT_DEVICE_LIB_PATH,
                   help="Path to .lib to include")

    p.add_argument("--device-model", type=str, default=DEFAULT_DEVICE_MODEL,
                   help="Model name inside .lib (e.g., TN0102, TP0602, 1N914, etc.)")

    p.add_argument("--device-subckt", type=str, default=DEFAULT_DEVICE_SUBCKT,
                   help="Subckt name (only used when --device-mode=subckt)")

    p.add_argument("--swap-ds", action="store_true",
                   help="Swap D and S when instantiating devices (useful for PMOS/diode direction experiments)")

    # Rails
    p.add_argument("--vminus", type=float, default=0.0)
    p.add_argument("--vplus", type=float, default=0.45)
    p.add_argument("--ionosphere-dir", type=str, default=DEFAULT_IONOSPHERE_DIR)
    p.add_argument("--dataset", type=str, choices=["digits", "ionosphere"], default="ionosphere")
    # Body tie
    p.add_argument(
        "--body-tie",
        type=str,
        choices=["source", "ground", "floating", "drain"],
        default="ground",
        help="Body tie mode: source, ground, floating, or drain (default ground).",
    )
    p.add_argument(
        "--gate-ref",
        type=str,
        choices=["ground", "source", "drain"],
        default="ground",
        help="Gate voltage reference node: ground (VGate absolute), source (VGS direct), drain (VGD direct).",
    )
    p.add_argument(
        "--loss",
        type=str,
        choices=["hinge", "mse", "sq_hinge"],
        default="hinge",
        help="Loss / clamp type: hinge (nudge y+rival), mse (clamp all to rails), sq_hinge (nudge scaled by loss).",
    )
    # Init VG
    p.add_argument("--vg-init-from", type=str, default=None,
                   help="Load initial VG from a .npy file (overrides --vg-init).")
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
    p.add_argument("--vg-clip-lo", type=float, default=VG_CLIP_LO,
                   help="VG clip lower bound (default 0.4 for NMOS, use -8.0 for PMOS)")
    p.add_argument("--vg-clip-hi", type=float, default=VG_CLIP_HI,
                   help="VG clip upper bound (default 8.0 for NMOS, use -0.4 for PMOS)")

    # Input scaling (ionosphere only)
    p.add_argument("--input-scale", type=float, default=5.0,
                   help="Multiply raw ionosphere features [-1,1] by this factor (default 5.0)")

    # Solver
    p.add_argument("--solver", type=str, choices=["klu", "sparse"], default="klu")

    # Debug dataset limits
    p.add_argument("--train-limit", type=int, default=0, help="if >0, limit train samples")
    p.add_argument("--test-limit", type=int, default=0, help="if >0, limit test samples")

    # Diagnostics
    p.add_argument("--final-test", action="store_true")

    # Pruning
    p.add_argument("--prune", action="store_true",
                   help="Enable edge pruning during training.")
    p.add_argument("--prune-frac", type=float, default=0.2,
                   help="Fraction of active edges to prune at each pruning event (default 0.2).")
    p.add_argument("--prune-start-epoch", type=int, default=5,
                   help="Epoch at which first pruning occurs (default 5).")
    p.add_argument("--prune-interval", type=int, default=5,
                   help="Prune every N epochs after prune-start-epoch (default 5).")
    p.add_argument("--regrow-after", type=int, default=10,
                   help="Epochs after pruning before regrowing an edge (default 10). 0 = never regrow.")
    return p.parse_args()


# -------------------------
# Pruning helpers
# -------------------------
def compute_off_distance(vg_arr: np.ndarray, clip_lo: float, clip_hi: float, device_mode: str) -> np.ndarray:
    if device_mode == "pmos":
        return clip_hi - vg_arr
    else:
        return vg_arr - clip_lo

def do_prune(ep: int, vg_unique: np.ndarray, pruned_mask: np.ndarray,
             prune_epoch_map: dict, prune_frac: float,
             clip_lo: float, clip_hi: float, device_mode: str,
             ng) -> int:
    active = np.where(~pruned_mask)[0]
    if len(active) == 0:
        return 0
    n_prune = max(1, int(len(active) * prune_frac))
    off_dist = compute_off_distance(vg_unique[active], clip_lo, clip_hi, device_mode)
    weakest = active[np.argsort(off_dist)[:n_prune]]
    off_val = clip_lo if device_mode != "pmos" else clip_hi
    cmds = []
    for uid in weakest:
        pruned_mask[uid] = True
        prune_epoch_map[uid] = ep
        vg_unique[uid] = off_val
        cmds.append(f"alter VG{uid} dc = {off_val:.16f}")
    if cmds:
        exec_chunked(ng, cmds)
    print(f"[prune] epoch={ep} pruned {len(weakest)} edges → "
          f"total_pruned={int(pruned_mask.sum())}/{len(pruned_mask)}", flush=True)
    return len(weakest)

def do_regrow(ep: int, vg_unique: np.ndarray, pruned_mask: np.ndarray,
              prune_epoch_map: dict, regrow_after: int, vg_regrow_val: float,
              ng) -> int:
    to_regrow = [uid for uid, ep_p in list(prune_epoch_map.items())
                 if (ep - ep_p) >= regrow_after and pruned_mask[uid]]
    if not to_regrow:
        return 0
    cmds = []
    for uid in to_regrow:
        pruned_mask[uid] = False
        del prune_epoch_map[uid]
        vg_unique[uid] = vg_regrow_val
        cmds.append(f"alter VG{uid} dc = {vg_regrow_val:.16f}")
    if cmds:
        exec_chunked(ng, cmds)
    print(f"[regrow] epoch={ep} regrew {len(to_regrow)} edges → "
          f"total_pruned={int(pruned_mask.sum())}/{len(pruned_mask)}", flush=True)
    return len(to_regrow)


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


def get_branch_currents(ng: NgSpiceShared, source_names: List[str]) -> Dict[str, float]:
    """Read branch currents for named voltage sources after a .op solve.

    ngspice prints branch currents as:  vplus#branch = -1.23456e-03
    Convention: positive value = current flows OUT of the + terminal into the network
    (source is delivering power). Power delivered = V_source * current.

    Returns a dict {name_lowercase: current_A}.
    Returns NaN for any source not found in the output.
    """
    s = ng.exec_command("print alli")
    currmap: Dict[str, float] = {}
    for line in s.splitlines():
        line = line.strip()
        if "#branch" not in line:
            continue
        try:
            k, v = line.split(" = ")
            src = k.replace("#branch", "").strip().lower()
            currmap[src] = float(v)
        except Exception:
            continue
    return {name.lower(): currmap.get(name.lower(), float("nan")) for name in source_names}


def run_and_read(
    ng: NgSpiceShared,
    read_specs: Dict[str, List[int]],
    current_sources: Optional[List[str]] = None,
) -> Tuple[bool, float, Optional[Dict[str, np.ndarray]], Optional[str]]:
    t0 = time.time()
    try:
        ng.run()
        dt = time.time() - t0
        data = get_voltages_multi(ng, read_specs)
        if current_sources:
            # Read branch currents before destroy; store as scalar floats in data
            cmap = get_branch_currents(ng, current_sources)
            data["_currents"] = np.array([cmap.get(n.lower(), float("nan"))
                                          for n in current_sources], dtype=float)
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
    # Clamp y and rival through RS=RS_CLAMP
    cmds.append(f"alter RS{y+1} {RS_CLAMP:.6g}")
    cmds.append(f"alter RS{r+1} {RS_CLAMP:.6g}")

    Vy_c = float(Vy + 0.5 * delta)
    Vr_c = float(Vr - 0.5 * delta)
    cmds.append(f"alter VOUT{y} dc = {Vy_c:.16f}")
    cmds.append(f"alter VOUT{r} dc = {Vr_c:.16f}")
    exec_chunked(ng, cmds)


def alter_outputs_sq_hinge(
    ng: NgSpiceShared,
    K: int,
    y: int,
    r: int,
    Vy: float,
    Vr: float,
    delta: float,
    loss_val: float,
    margin: float,
):
    # Nudge scaled by loss magnitude: larger loss → larger clamp nudge.
    # ∂(sq_hinge)/∂Vy ∝ -2*loss, ∂/∂Vr ∝ +2*loss, so scale delta by loss/margin.
    scale = float(loss_val) / float(margin) if margin > 1e-12 else 1.0
    scaled_delta = float(delta) * scale
    cmds: List[str] = [f"alter RS{i} {RS_FREE:.6g}" for i in range(1, K + 1)]
    cmds.append(f"alter RS{y+1} {RS_CLAMP:.6g}")
    cmds.append(f"alter RS{r+1} {RS_CLAMP:.6g}")
    cmds.append(f"alter VOUT{y} dc = {float(Vy + 0.5 * scaled_delta):.16f}")
    cmds.append(f"alter VOUT{r} dc = {float(Vr - 0.5 * scaled_delta):.16f}")
    exec_chunked(ng, cmds)


def alter_outputs_mse(
    ng: NgSpiceShared,
    K: int,
    y: int,
    vminus_val: float,
    vplus_val: float,
):
    # Clamp ALL outputs hard to their targets: y → V+, others → V-.
    # Provides full supervision signal in the clamp phase.
    cmds: List[str] = []
    for i in range(K):
        target = vplus_val if i == y else vminus_val
        cmds.append(f"alter RS{i+1} {RS_CLAMP:.6g}")
        cmds.append(f"alter VOUT{i} dc = {float(target):.16f}")
    exec_chunked(ng, cmds)

def mk_netlist(
    topo: Topology,
    vg_unique: np.ndarray,
    vminus_val: float,
    vplus_val: float,
    solver: str,
    body_res: float,
    body_tie: str,
    device_lib_path: str,
    device_mode: str,
    device_model: str,
    device_subckt: str,
    swap_ds: bool,
    gate_ref: str = "ground",
) -> str:
    if vg_unique.size != topo.num_edges:
        raise ValueError("vg_unique size mismatch")

    lines: List[str] = []
    lines.append(".title scikit_digits_dense_io_hinge")
    lines.append(f'.include "{device_lib_path}"')

    '''
    # --- TOY NMOS MODEL (debug only; replaces external .include) ---
    lines.extend([
        "* ---- TOY NMOS MODEL (for debug / pipeline bring-up) ----",
        "* Level-1 NMOS + wrapper subckt matching your expected pin order: D G S B",
        ".model ncg nmos (LEVEL=1 VTO=0.45 KP=2e-4 GAMMA=0.0 LAMBDA=0.02 PHI=0.6)",
        ".subckt NMOSWRAP D G S B",
        "M0 D G S B ncg",
        ".ends NMOSWRAP",
        "* ---------------------------------------------------------",
    ])
    # --------------------------------------------------------------
    '''
    if solver.lower() == "klu":
        lines.append(".options klu")

    # Rails
    lines.append(f"VMINUS {topo.negref} 0 {float(vminus_val):.16f}")
    lines.append(f"VPLUS  {topo.posref} 0 {float(vplus_val):.16f}")

    # Inputs
    for i, n in enumerate(topo.input_nodes):
        lines.append(f"VIN{i} {n} 0 0")

    # Outputs: RS + sink node + VOUT sources
    node_pool = ([topo.negref, topo.posref]
                 + topo.input_nodes.tolist()
                 + topo.out_nodes.tolist()
                 + topo.edges_D.tolist()
                 + topo.edges_S.tolist())
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

        # Optionally swap D and S for experiments (must happen before gate_ref/body use)
        Dp, Sp = (S, D) if swap_ds else (D, S)

        # Gate voltage source: reference node depends on gate_ref
        if gate_ref == "ground":
            gate_ref_node = "0"
        elif gate_ref == "source":
            gate_ref_node = str(Sp)
        elif gate_ref == "drain":
            gate_ref_node = str(Dp)
        else:
            raise ValueError(f"Unsupported gate_ref: {gate_ref}")
        lines.append(f"VG{eidx} {gate_node} {gate_ref_node} {float(vg_unique[eidx]):.16f}")

        if body_tie == "source":
            if body_res <= 0.0:
                body_node = str(Sp)
            else:
                body_node = f"b{eidx}"
                lines.append(f"RB{eidx} {body_node} {Sp} {float(body_res):.6g}")
        elif body_tie == "ground":
            if body_res <= 0.0:
                body_node = "0"
            else:
                body_node = f"b{eidx}"
                lines.append(f"RB{eidx} {body_node} 0 {float(body_res):.6g}")
        elif body_tie == "floating":
            body_node = f"b{eidx}"
        elif body_tie == "drain":
            body_node = str(Dp)
        else:
            raise ValueError(f"Unsupported body_tie: {body_tie}")

        if device_mode == "subckt":
            # Old behavior: requires .subckt NMOSWRAP D G S B
            lines.append(f"X{eidx} {Dp} {gate_node} {Sp} {body_node} {device_subckt}")

        elif device_mode in ("nmos", "pmos"):
            # MicroCap libraries usually provide .MODEL <name> NMOS/PMOS (...)
            # Instantiate the primitive MOSFET directly:
            #   Mname D G S B <model> [L=.. W=..]
            # L/W are optional; if your model already includes L/W you can omit.
            lines.append(
                f"M{eidx} {Dp} {gate_node} {Sp} {body_node} {device_model} "
                f"L=2.5e-6 W=0.9e-2"
            )

        elif device_mode == "diode":
            # Diode primitive: Dname Anode Cathode <model>
            # Gate voltage sources remain in netlist (harmless) so your training code doesn’t change.
            lines.append(f"D{eidx} {Dp} {Sp} {device_model}")

        else:
            raise ValueError(f"Unsupported device_mode={device_mode}")
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


def compute_vg_saturation_stats(vg_unique: np.ndarray, lo: float = VG_CLIP_LO, hi: float = VG_CLIP_HI) -> Dict[str, float]:
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

    device_mode = args.device_mode
    device_lib_path = args.device_lib
    device_model = args.device_model
    device_subckt = args.device_subckt
    swap_ds = bool(args.swap_ds)

    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)

    epochs = int(args.epochs)
    if epochs < 1:
        raise ValueError("--epochs must be >= 1")

    gamma = float(args.gamma)
    gamma_final = float(args.gamma_final)
    gamma_decay_rate = float(args.gamma_decay_rate)
    margin = float(args.margin)
    delta = float(args.delta)
    delta_final = float(args.delta_final)
    vmin = float(args.input_vmin)
    vmax = float(args.input_vmax)
    if vmax <= vmin:
        raise ValueError("--input-vmax must be > --input-vmin")

    vminus_val = float(args.vminus)
    vplus_val = float(args.vplus)
    solver = str(args.solver).lower()
    body_res = float(RS_CLAMP)
    body_tie = str(args.body_tie)
    gate_ref = str(args.gate_ref)
    loss_type = str(args.loss)
    vg_init_mode = str(args.vg_init)
    vg_init_lo = float(args.vg_init_lo)
    vg_init_hi = float(args.vg_init_hi)
    vg_init_single = float(args.vg_init_fixed)
    vg_clip_lo = float(args.vg_clip_lo)
    vg_clip_hi = float(args.vg_clip_hi)
    if vg_init_mode == "random" and vg_init_hi <= vg_init_lo:
        raise ValueError("--vg-init-hi must be > --vg-init-lo for random init")

    if args.dataset == "digits":
        digits = load_digits()
        imgs = (digits.images / 16.0).astype(np.float64)  # (N,8,8) in [0,1]
        y = digits.target.astype(int)
        X_raw = imgs.reshape(len(imgs), -1)  # Nin=64
        X = vmin + (vmax - vmin) * X_raw

    else:
        # Dataset: UCI Ionosphere (34 features already in [-1,1])
        X_raw, y, data_path = load_ionosphere_from_dir(args.ionosphere_dir)
        print(f"[dataset] ionosphere file={data_path}", flush=True)
        print(f"[dataset] raw min/max = {float(X_raw.min()):.3f} / {float(X_raw.max()):.3f}", flush=True)

        # IMPORTANT: keep raw feature ordering; no per-feature normalization.
        # If the caller requests a different voltage window, apply one global
        # affine map from the observed dataset min/max into [vmin, vmax].
        raw_min = float(X_raw.min())
        raw_max = float(X_raw.max())
        if np.isclose(vmin, raw_min) and np.isclose(vmax, raw_max):
            X = X_raw.astype(np.float64)
        else:
            a = raw_min
            b = raw_max
            X = vmin + (vmax - vmin) * (X_raw - a) / (b - a + 1e-12)

        X = X * float(args.input_scale)

        # Threshold stats for your diode regime
        frac_vals = float((X < -0.5).mean())
        frac_samples = float((X < -0.5).any(axis=1).mean())
        print(f"[dataset] frac(values<-0.5)={frac_vals:.4f}  frac(samples any<-0.5)={frac_samples:.4f}", flush=True)

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
    test_x  = [row.astype(float) for row in X_test]
    test_y  = [int(v) for v in y_test.tolist()]
    Nin = int(train_x[0].size)  # digits=64, ionosphere=34
    topo_path = Path(args.topology) if args.topology else TOPOLOGY_PATH
    if not topo_path.exists():
        raise FileNotFoundError(f"Topology file not found: {topo_path}")
    topo = load_topology_npz(topo_path)
    if topo.Nin != Nin:
        raise ValueError(f"Topology Nin={topo.Nin} does not match data Nin={Nin}")
    K = topo.K

    # Diagnostics use the full test set
    test_n = len(test_x)

    # Init weights (unique VG per edge)
    if args.vg_init_from:
        vg_unique = np.load(args.vg_init_from).astype(float)
        if vg_unique.size != topo.num_edges:
            raise ValueError(f"--vg-init-from size {vg_unique.size} != topo.num_edges {topo.num_edges}")
    elif vg_init_mode == "fixed":
        vg_unique = np.full((topo.num_edges,), vg_init_single, dtype=float)
    else:
        vg_unique = np.random.uniform(vg_init_lo, vg_init_hi, size=(topo.num_edges,)).astype(float)

    # Pruning state
    prune_enabled = bool(args.prune)
    prune_frac = float(args.prune_frac)
    prune_start_epoch = int(args.prune_start_epoch)
    prune_interval = int(args.prune_interval)
    regrow_after = int(args.regrow_after)
    pruned_mask = np.zeros(topo.num_edges, dtype=bool)
    prune_epoch_map: dict = {}
    vg_regrow_val = vg_init_single if vg_init_mode == "fixed" else 0.5 * (vg_init_lo + vg_init_hi)
    prune_count_hist: List[int] = []
    regrow_count_hist: List[int] = []
    active_edge_hist: List[int] = []

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
        f"solver={solver} body_tie={body_tie} gate_ref={gate_ref} loss={loss_type} "
        f"body_res={body_res} rs_clamp={RS_CLAMP} "
        f"vg_init={vg_init_mode} "
        f"epochs={epochs} "
        f"device_mode={device_mode} device_include={device_lib_path} "
        f"device_model={device_model} device_subckt={device_subckt} swap_ds={swap_ds} "
        f"topology={topo_path.name}"
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
        body_tie=body_tie,
        device_lib_path=device_lib_path,
        device_mode=device_mode,
        device_model=device_model,
        device_subckt=device_subckt,
        swap_ds=swap_ds,
        gate_ref=gate_ref,
    )    
    
    (run_dir / "netlist_initial.cir").write_text(netlist)

    meta = {
        "script": str(Path(__file__).resolve()),
        "argv": list(os.sys.argv),
        "seed": seed,
        "timestamp": datetime.now().isoformat(),
        "dataset": args.dataset,
        "input_shape": [34] if args.dataset == "ionosphere" else [8, 8],
        "test_count": len(test_x),
        "gamma": gamma,
        "margin": margin,
        "delta": delta,
        "epochs": epochs,
        "input_vmin": vmin,
        "input_vmax": vmax,
        "rails": {"vminus": vminus_val, "vplus": vplus_val},
        "solver": solver,
        "body_tie": body_tie,
        "gate_ref": gate_ref,
        "loss_type": loss_type,
        "body_res": body_res,
        "rs_clamp": RS_CLAMP,
        "vg_init": {
            "mode": vg_init_mode,
            "lo": vg_init_lo,
            "hi": vg_init_hi,
            "fixed": vg_init_single,
        },
        "device": {
            "mode": device_mode,
            "include_path": device_lib_path,
            "model": device_model,
            "subckt": device_subckt,
            "swap_ds": swap_ds,
        },
        "topology": {
            "path": str(topo_path),
            "Nin": topo.Nin,
            "out": topo.K,
            "edges": topo.num_edges,
            "meta": topo.meta,
        },
        "diagnostics": {"vout_saved": "test"},
        "diode_source_to_vplus": False,
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

    np.save(run_dir / "test_x.npy", np.asarray(test_x, dtype=float))
    np.save(run_dir / "test_y.npy", np.asarray(test_y, dtype=int))

    # Ngspice
    ng = NgSpiceShared(send_data=False)
    ng.load_circuit(netlist)

    # Nodes list (numeric nodes only) — include any hidden nodes referenced by edges
    net_nodes = ([topo.negref, topo.posref]
                 + topo.out_nodes.tolist()
                 + topo.input_nodes.tolist()
                 + topo.edges_D.tolist()
                 + topo.edges_S.tolist())
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
    power_supply_uw_hist: List[float] = []
    power_total_uw_hist: List[float] = []

    ep_total_s: List[float] = []
    ep_free_s: List[float] = []
    ep_clamp_s: List[float] = []
    ep_update_s: List[float] = []

    hinge_active_frac_hist: List[float] = []
    reload_free_hist: List[int] = []
    reload_clamp_hist: List[int] = []
    nonfinite_free_hist: List[int] = []
    nonfinite_clamp_hist: List[int] = []

    # Names of voltage sources whose currents we track for power logging.
    # VPLUS and VMINUS are the supply rails. VIN sources drive the inputs.
    _power_vplus_name  = "vplus"
    _power_vminus_name = "vminus"
    _power_vin_names   = [f"vin{i}" for i in range(Nin)]

    def eval_free_metrics(epoch: int) -> Tuple[float, float, Dict[str, float]]:
        mk_free_all(ng, K)
        correct = 0
        total = 0
        loss_sum = 0.0
        count = 0

        vout_test = np.full((test_n, K), np.nan, dtype=float) if test_n > 0 else np.zeros((0, K), dtype=float)
        gap_list: List[float] = []
        sat_list: List[float] = []
        hinge_list: List[float] = []

        # Power tracking (micro-watts per sample)
        power_supply_uw_list: List[float] = []   # from VPLUS rail only
        power_total_uw_list:  List[float] = []   # supply + input sources

        reloads = 0
        nonfinite = 0
        confusion = np.zeros((K, K), dtype=int)

        # Source names to read currents from, ordered for indexing below.
        # Only sample power on the FIRST test sample per epoch to avoid
        # the overhead of `print alli` (240+ branches) on every sample.
        _src_order = [_power_vplus_name, _power_vminus_name] + _power_vin_names

        for i, (xt, yt) in enumerate(zip(test_x, test_y)):
            alter_inputs_named(ng, xt)
            read_currents = _src_order if i == 0 else None
            ok, _, data, _ = run_and_read(ng, {"out": topo.out_nodes},
                                          current_sources=read_currents)
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

            # ── Power measurement ──────────────────────────────────────────
            # Power measured on sample 0 only (representative; avoids per-sample overhead).
            if i == 0:
                try:
                    cvals = data.get("_currents")
                    if cvals is not None and len(cvals) >= 2:
                        i_vplus  = float(cvals[0])
                        i_vminus = float(cvals[1])
                        i_vins   = cvals[2:]
                        p_supply = 0.0
                        if np.isfinite(i_vplus):
                            p_supply += abs(vplus_val * i_vplus)
                        if np.isfinite(i_vminus):
                            p_supply += abs(vminus_val * i_vminus)
                        p_input = 0.0
                        for j, i_vin in enumerate(i_vins):
                            if np.isfinite(i_vin) and j < len(xt):
                                p_input += abs(float(xt[j]) * i_vin)
                        p_supply_uw = p_supply * 1e6
                        p_total_uw  = (p_supply + p_input) * 1e6
                        if np.isfinite(p_supply_uw):
                            power_supply_uw_list.append(p_supply_uw)
                        if np.isfinite(p_total_uw):
                            power_total_uw_list.append(p_total_uw)
                except Exception:
                    pass

            pred = int(np.argmax(Vout))
            ytrue = int(yt)
            correct += int(pred == ytrue)
            total += 1
            if 0 <= ytrue < K:
                confusion[ytrue, pred] += 1

            hl = hinge_loss_from_outputs(Vout, ytrue, margin)
            loss_sum += float(hl)
            count += 1

            vout_test[i, :] = Vout
            g = margin_gap(Vout, ytrue)
            gap_list.append(float(g))
            sat_list.append(float(1.0 if g >= margin else 0.0))
            hinge_list.append(float(hl))

        if test_n > 0:
            np.save(run_dir / f"0_vout_test_epoch{epoch}.npy", vout_test)
            np.save(run_dir / f"0_val_confusion_epoch{epoch}.npy", confusion)

        diag: Dict[str, float] = {}
        if gap_list:
            gaps = np.asarray(gap_list, dtype=float)
            sats = np.asarray(sat_list, dtype=float)
            hinges = np.asarray(hinge_list, dtype=float)
            diag["test_gap_mean"] = float(np.mean(gaps))
            diag["test_gap_median"] = float(np.median(gaps))
            diag["test_gap_std"] = float(np.std(gaps))
            diag["test_satisfy_frac"] = float(np.mean(sats))
            diag["test_hinge_mean"] = float(np.mean(hinges))
        else:
            diag["test_gap_mean"] = float("nan")
            diag["test_gap_median"] = float("nan")
            diag["test_gap_std"] = float("nan")
            diag["test_satisfy_frac"] = float("nan")
            diag["test_hinge_mean"] = float("nan")

        diag["val_reloads"] = float(reloads)
        diag["val_nonfinite"] = float(nonfinite)

        # Power diagnostics
        if power_supply_uw_list:
            diag["power_supply_uw_mean"] = float(np.mean(power_supply_uw_list))
            diag["power_supply_uw_std"]  = float(np.std(power_supply_uw_list))
        else:
            diag["power_supply_uw_mean"] = float("nan")
            diag["power_supply_uw_std"]  = float("nan")
        if power_total_uw_list:
            diag["power_total_uw_mean"] = float(np.mean(power_total_uw_list))
            diag["power_total_uw_std"]  = float(np.std(power_total_uw_list))
        else:
            diag["power_total_uw_mean"] = float("nan")
            diag["power_total_uw_std"]  = float("nan")

        acc = (correct / total) if total else float("nan")
        loss = (loss_sum / count) if count else float("nan")
        return float(acc), float(loss), diag

    # Epoch 0 validation
    v0, h0, diag0 = eval_free_metrics(epoch=0)
    val_acc_hist.append(v0)
    val_hinge_hist.append(h0)
    power_supply_uw_hist.append(diag0.get("power_supply_uw_mean", float("nan")))
    power_total_uw_hist.append(diag0.get("power_total_uw_mean", float("nan")))
    np.save(run_dir / "0_val_acc.npy", np.asarray(val_acc_hist, dtype=float))
    np.save(run_dir / "0_val_hinge.npy", np.asarray(val_hinge_hist, dtype=float))
    np.save(run_dir / "0_power_supply_uw.npy", np.asarray(power_supply_uw_hist, dtype=float))
    np.save(run_dir / "0_power_total_uw.npy", np.asarray(power_total_uw_hist, dtype=float))
    (run_dir / "0_diag_epoch0.json").write_text(json.dumps(diag0, indent=2))
    print(
        f"[epoch 0] {cfg_str} | VAL acc={v0:.4f} hinge={h0:.6f} test_satisfy={diag0.get('test_satisfy_frac', float('nan')):.4f}",
        flush=True,
    )

    for ep in range(1, epochs + 1):
        t_ep0 = time.time()
        # Gamma schedule
        if gamma_decay_rate >= 0.0:
            gamma_ep = gamma * (gamma_decay_rate ** (ep - 1))
        elif gamma_final >= 0.0 and epochs > 1:
            gamma_ep = gamma + (gamma_final - gamma) * (ep - 1) / (epochs - 1)
        else:
            gamma_ep = gamma
        # Delta schedule
        if delta_final >= 0.0 and epochs > 1:
            delta_ep = delta + (delta_final - delta) * (ep - 1) / (epochs - 1)
        else:
            delta_ep = delta
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
        diode_hit_count = 0
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
            if loss_type == "hinge":
                alter_outputs_hinge(ng, K=K, y=ytrue, r=rival, Vy=Vy, Vr=Vr, delta=delta_ep)
            elif loss_type == "sq_hinge":
                alter_outputs_sq_hinge(ng, K=K, y=ytrue, r=rival, Vy=Vy, Vr=Vr,
                                       delta=delta_ep, loss_val=hl, margin=margin)
            elif loss_type == "mse":
                alter_outputs_mse(ng, K=K, y=ytrue,
                                  vminus_val=vminus_val, vplus_val=vplus_val)

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
            hit = np.any((Vd_free < -0.5) | (Vs_free < -0.5))
            if hit:
                diode_hit_count += 1
            # accumulate per-epoch
            Vd_c = Vnodes_clamp[index_of[eD]]
            Vs_c = Vnodes_clamp[index_of[eS]]

            dV_free = Vd_free - Vs_free
            dV_c = Vd_c - Vs_c

            update = -gamma_ep * (dV_c**2 - dV_free**2)

            cmds: List[str] = []
            for uid in range(topo.num_edges):
                if pruned_mask[uid]:
                    continue
                du = float(update[uid])
                nv = float(vg_unique[uid] + du)
                if nv < vg_clip_lo:
                    nv = vg_clip_lo
                elif nv > vg_clip_hi:
                    nv = vg_clip_hi
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
        diode_hit_frac = (diode_hit_count / n_free) if n_free else float("nan")
        hinge_active_frac_hist.append(float(hinge_active_frac))

        reload_free_hist.append(int(reload_free))
        reload_clamp_hist.append(int(reload_clamp))
        nonfinite_free_hist.append(int(nonfinite_free))
        nonfinite_clamp_hist.append(int(nonfinite_clamp))

        v_acc, v_h, diag = eval_free_metrics(epoch=ep)
        val_acc_hist.append(float(v_acc))
        val_hinge_hist.append(float(v_h))
        power_supply_uw_hist.append(diag.get("power_supply_uw_mean", float("nan")))
        power_total_uw_hist.append(diag.get("power_total_uw_mean", float("nan")))

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
        np.save(run_dir / "0_power_supply_uw.npy", np.asarray(power_supply_uw_hist, dtype=float))
        np.save(run_dir / "0_power_total_uw.npy", np.asarray(power_total_uw_hist, dtype=float))

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

        # Pruning / regrowth
        n_pruned_this_ep = 0
        n_regrown_this_ep = 0
        if prune_enabled:
            if regrow_after > 0:
                n_regrown_this_ep = do_regrow(ep, vg_unique, pruned_mask, prune_epoch_map,
                                              regrow_after, vg_regrow_val, ng)
            if ep >= prune_start_epoch and (ep - prune_start_epoch) % prune_interval == 0:
                n_pruned_this_ep = do_prune(ep, vg_unique, pruned_mask, prune_epoch_map,
                                            prune_frac, vg_clip_lo, vg_clip_hi, device_mode, ng)
        prune_count_hist.append(n_pruned_this_ep)
        regrow_count_hist.append(n_regrown_this_ep)
        active_edge_hist.append(int((~pruned_mask).sum()))
        np.save(run_dir / "0_pruned_mask.npy", pruned_mask.copy())
        np.save(run_dir / "0_prune_count.npy", np.asarray(prune_count_hist, dtype=int))
        np.save(run_dir / "0_regrow_count.npy", np.asarray(regrow_count_hist, dtype=int))
        np.save(run_dir / "0_active_edges.npy", np.asarray(active_edge_hist, dtype=int))

        vg_stats = compute_vg_saturation_stats(vg_unique, lo=vg_clip_lo, hi=vg_clip_hi)
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
                "device_include_path": device_lib_path,
                "device_subckt": device_subckt,
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
            f"[epoch {ep}/{epochs}] {cfg_str} | "
            f"TRAIN acc={tr_acc:.4f} hinge={tr_h:.6f} hinge_frac={hinge_active_frac:.3f} "
            f"diode_hit_frac={diode_hit_frac:.3f} "
            f"free={n_free} clamp={n_clamp} skipped={skipped} "
            f"reloadF={reload_free} reloadC={reload_clamp} nonfiniteF={nonfinite_free} nonfiniteC={nonfinite_clamp}",
            flush=True,
        )
        print(
            f"[epoch {ep}/{epochs}] {cfg_str} | "
            f"VAL acc={v_acc:.4f} hinge={v_h:.6f} "
            f"test_satisfy={diag.get('test_satisfy_frac', float('nan')):.4f} "
            f"test_gap_mean={diag.get('test_gap_mean', float('nan')):.4f} | "
            f"timing total={ep_total:.2f}s free={t_free:.2f}s clamp={t_clamp:.2f}s upd={t_update:.2f}s",
            flush=True,
        )
        print(
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

    print("=== RUN END (scikit_digits_dense_io_hinge) ===", flush=True)
    try:
        log_f.flush()
        log_f.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
