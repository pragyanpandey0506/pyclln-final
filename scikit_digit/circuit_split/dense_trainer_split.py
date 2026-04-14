#!/usr/bin/env python3
"""
Split-backend variant of scikit_digit/dense_trainer.py.

This trainer exploits the no-hidden-node topology: each output node is only
connected to driven input sources, so each output branch can be simulated in an
independent ngspice session. The training loop keeps the original free ->
hinge -> clamp -> update ordering and updates every session in place after the
clamped solve.
"""

from __future__ import annotations

import ctypes.util
import json
import locale
import os
import random
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SCIKIT_ROOT = REPO_ROOT / "scikit_digit"
LOCAL_NGSPICE_LIB_DIR = Path(__file__).resolve().parent / "_ngspice_libs"
MAX_PARALLEL_LIBS = 10


def resolve_ngspice_source_library() -> Path:
    candidates = [
        Path(sys.prefix) / "lib" / "libngspice.so",
        Path(sys.prefix) / "lib" / "libngspice.so.0",
        Path(sys.prefix) / "lib" / "libngspice.so.0.0.8",
    ]
    found = ctypes.util.find_library("ngspice")
    if found:
        found_path = Path(found)
        if found_path.exists() and found_path.is_file():
            candidates.insert(0, found_path)

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("Could not locate libngspice for the split backend.")


def prepare_ngspice_library_pattern(num_instances: int) -> str:
    if num_instances < 1:
        raise ValueError("num_instances must be >= 1")

    src = resolve_ngspice_source_library()
    LOCAL_NGSPICE_LIB_DIR.mkdir(parents=True, exist_ok=True)

    for ng_id in range(num_instances):
        name = "libngspice.so" if ng_id == 0 else f"libngspice{ng_id}.so"
        dst = LOCAL_NGSPICE_LIB_DIR / name
        if not dst.exists() or dst.stat().st_size != src.stat().st_size:
            shutil.copy2(src, dst)

    return str(LOCAL_NGSPICE_LIB_DIR / "libngspice{}.so")


NGSPICE_LIBRARY_PATTERN = prepare_ngspice_library_pattern(MAX_PARALLEL_LIBS)
os.environ["NGSPICE_LIBRARY_PATH"] = NGSPICE_LIBRARY_PATTERN

sys.path.insert(0, str(SCIKIT_ROOT))

import dense_trainer as base  # noqa: E402
import PySpice.Spice.NgSpice.Shared as ngshared_module  # noqa: E402
from PySpice.Spice.NgSpice.Shared import NgSpiceShared  # noqa: E402
from topology.topology_io import Topology, load_topology_npz  # noqa: E402


RESULTS_ROOT = REPO_ROOT / "scikit_digit" / "results" / "circuit_split"
BEST_REFERENCE_RUN = (
    REPO_ROOT
    / "scikit_digit"
    / "results"
    / "dense_sweep_11jan2026"
    / "run_0489_seed1_g0.3_d0.05_m0.02_btfloating_vgfix4"
)


def patch_pyspice_shared_loader() -> None:
    if getattr(ngshared_module, "_split_loader_patched", False):
        return

    def _patched_load_library(self, verbose):
        if ngshared_module.ConfigInstall.OS.on_windows:
            if "SPICE_LIB_DIR" not in os.environ:
                os.environ["SPICE_LIB_DIR"] = str(
                    Path(self.NGSPICE_PATH).joinpath("share", "ngspice")
                )
        elif ngshared_module.ConfigInstall.OS.on_linux or ngshared_module.ConfigInstall.OS.on_osx:
            locale.setlocale(locale.LC_NUMERIC, "C")

        api_path = Path(ngshared_module.__file__).parent.joinpath("api.h")
        if not getattr(ngshared_module, "_split_cdef_loaded", False):
            with open(api_path) as fh:
                ngshared_module.ffi.cdef(fh.read())
            ngshared_module._split_cdef_loaded = True

        if verbose:
            print(f"Load library {self.library_path}")
        self._ngspice_shared = ngshared_module.ffi.dlopen(self.library_path)

    NgSpiceShared._load_library = _patched_load_library
    ngshared_module._split_loader_patched = True


patch_pyspice_shared_loader()


@dataclass(frozen=True)
class SplitSession:
    out_index: int
    out_node: int
    edge_ids: np.ndarray
    drain_input_idx: np.ndarray
    topo: Topology
    netlist: str
    ngspice_id: int
    ng: NgSpiceShared


def build_split_sessions(
    topo: Topology,
    vg_unique: np.ndarray,
    vminus_val: float,
    vplus_val: float,
    solver: str,
    body_res: float,
    body_tie: str,
) -> List[SplitSession]:
    input_index_of = {int(node): idx for idx, node in enumerate(topo.input_nodes.tolist())}
    sessions: List[SplitSession] = []

    for out_index, out_node in enumerate(topo.out_nodes.tolist()):
        edge_ids = np.flatnonzero(topo.edges_S == int(out_node))
        if edge_ids.size == 0:
            raise ValueError(f"Output node {out_node} has no attached edges.")

        local_topo = Topology(
            negref=topo.negref,
            posref=topo.posref,
            input_nodes=topo.input_nodes,
            out_nodes=np.asarray([int(out_node)], dtype=int),
            edges_D=topo.edges_D[edge_ids],
            edges_S=topo.edges_S[edge_ids],
            meta={
                "split_backend": True,
                "parent_topology_path": str(base.TOPOLOGY_PATH),
                "out_index": int(out_index),
                "out_node": int(out_node),
                "edges": int(edge_ids.size),
            },
        )
        local_vg = vg_unique[edge_ids]
        netlist = base.mk_netlist(
            topo=local_topo,
            vg_unique=local_vg,
            vminus_val=vminus_val,
            vplus_val=vplus_val,
            solver=solver,
            body_res=body_res,
            body_tie=body_tie,
        )
        ng = NgSpiceShared(ngspice_id=out_index, send_data=False)
        ng.load_circuit(netlist)
        sessions.append(
            SplitSession(
                out_index=int(out_index),
                out_node=int(out_node),
                edge_ids=edge_ids.astype(int),
                drain_input_idx=np.asarray(
                    [input_index_of[int(node)] for node in local_topo.edges_D.tolist()],
                    dtype=int,
                ),
                topo=local_topo,
                netlist=netlist,
                ngspice_id=int(out_index),
                ng=ng,
            )
        )
    return sessions


def restore_session_gate_voltages(session: SplitSession, vg_unique: np.ndarray) -> None:
    cmds = [
        f"alter VG{local_idx} dc = {float(vg_unique[global_idx]):.16f}"
        for local_idx, global_idx in enumerate(session.edge_ids.tolist())
    ]
    if cmds:
        base.exec_chunked(session.ng, cmds)


def reload_all_sessions(sessions: Sequence[SplitSession], vg_unique: np.ndarray) -> None:
    for session in sessions:
        try:
            session.ng.remove_circuit()
        except Exception:
            pass
        session.ng.load_circuit(session.netlist)
        restore_session_gate_voltages(session, vg_unique)
        base.mk_free_all(session.ng, 1)


def _run_free_session(session: SplitSession, x: np.ndarray) -> Tuple[bool, bool, float]:
    base.mk_free_all(session.ng, 1)
    base.alter_inputs_named(session.ng, x)
    ok, _, data, _ = base.run_and_read(session.ng, {"out": session.topo.out_nodes.tolist()})
    if not ok or data is None:
        return False, False, float("nan")
    vout = float(np.asarray(data["out"], dtype=float)[0])
    if not np.isfinite(vout):
        return True, False, float("nan")
    return True, True, vout


def _run_clamp_session(
    session: SplitSession,
    x: np.ndarray,
    ytrue: int,
    rival: int,
    Vout_free: np.ndarray,
    delta: float,
) -> Tuple[bool, bool, float]:
    if session.out_index == ytrue:
        base.exec_chunked(
            session.ng,
            [
                f"alter RS1 {base.RS_CLAMP:.6g}",
                f"alter VOUT0 dc = {float(Vout_free[ytrue] + 0.5 * delta):.16f}",
            ],
        )
    elif session.out_index == rival:
        base.exec_chunked(
            session.ng,
            [
                f"alter RS1 {base.RS_CLAMP:.6g}",
                f"alter VOUT0 dc = {float(Vout_free[rival] - 0.5 * delta):.16f}",
            ],
        )
    else:
        base.exec_chunked(session.ng, [f"alter RS1 {base.RS_FREE:.6g}"])

    base.alter_inputs_named(session.ng, x)
    ok, _, data, _ = base.run_and_read(session.ng, {"out": session.topo.out_nodes.tolist()})
    if not ok or data is None:
        return False, False, float("nan")
    vout = float(np.asarray(data["out"], dtype=float)[0])
    if not np.isfinite(vout):
        return True, False, float("nan")
    return True, True, vout


def _apply_updated_gates(session: SplitSession, new_gate_values: np.ndarray) -> None:
    cmds = [f"alter VG{idx} dc = {float(v):.16f}" for idx, v in enumerate(new_gate_values.tolist())]
    if cmds:
        base.exec_chunked(session.ng, cmds)


def run_sessions_parallel(
    executor: ThreadPoolExecutor,
    fn,
    sessions: Sequence[SplitSession],
) -> List[Tuple[bool, bool, float]]:
    return list(executor.map(fn, sessions))


def eval_free_metrics(
    *,
    executor: ThreadPoolExecutor,
    sessions: Sequence[SplitSession],
    vg_unique: np.ndarray,
    run_dir: Path,
    epoch: int,
    test_x: Sequence[np.ndarray],
    test_y: Sequence[int],
    K: int,
    margin: float,
) -> Tuple[float, float, Dict[str, float]]:
    correct = 0
    total = 0
    loss_sum = 0.0
    count = 0

    test_n = len(test_x)
    vout_test = np.full((test_n, K), np.nan, dtype=float) if test_n > 0 else np.zeros((0, K), dtype=float)
    confusion = np.zeros((K, K), dtype=int)
    gap_list: List[float] = []
    sat_list: List[float] = []
    hinge_list: List[float] = []

    reloads = 0
    nonfinite = 0

    for sample_idx, (xt, yt) in enumerate(zip(test_x, test_y)):
        t0 = time.time()
        results = run_sessions_parallel(
            executor,
            lambda session: _run_free_session(session, xt),
            sessions,
        )
        _ = time.time() - t0
        if not all(item[0] for item in results):
            reloads += 1
            reload_all_sessions(sessions, vg_unique)
            continue
        if not all(item[1] for item in results):
            nonfinite += 1
            continue

        Vout = np.asarray([item[2] for item in results], dtype=float)
        pred = int(np.argmax(Vout))
        ytrue = int(yt)
        correct += int(pred == ytrue)
        total += 1
        confusion[ytrue, pred] += 1

        hl = base.hinge_loss_from_outputs(Vout, ytrue, margin)
        loss_sum += float(hl)
        count += 1

        vout_test[sample_idx, :] = Vout
        gap = base.margin_gap(Vout, ytrue)
        gap_list.append(float(gap))
        sat_list.append(float(1.0 if gap >= margin else 0.0))
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

    acc = (correct / total) if total else float("nan")
    loss = (loss_sum / count) if count else float("nan")
    return float(acc), float(loss), diag


def main() -> None:
    args = base.parse_args()
    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)

    epochs = int(args.epochs)
    if epochs < 1:
        raise ValueError("--epochs must be >= 1")

    gamma = float(args.gamma)
    margin = float(args.margin)
    delta = float(args.delta)
    vmin = float(args.input_vmin)
    vmax = float(args.input_vmax)
    if vmax <= vmin:
        raise ValueError("--input-vmax must be > --input-vmin")

    vminus_val = float(args.vminus)
    vplus_val = float(args.vplus)
    solver = str(args.solver).lower()
    body_res = float(base.RS_CLAMP)
    body_tie = str(args.body_tie)
    vg_init_mode = str(args.vg_init)
    vg_init_lo = float(args.vg_init_lo)
    vg_init_hi = float(args.vg_init_hi)
    vg_init_single = float(args.vg_init_fixed)
    if vg_init_mode == "random" and vg_init_hi <= vg_init_lo:
        raise ValueError("--vg-init-hi must be > --vg-init-lo for random init")

    digits = base.load_digits()
    imgs = (digits.images / 16.0).astype(np.float64)
    y = digits.target.astype(int)

    X_raw = imgs.reshape(len(imgs), -1)
    X = vmin + (vmax - vmin) * X_raw

    X_train, X_test, y_train, y_test = base.train_test_split(
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
    if not base.TOPOLOGY_PATH.exists():
        raise FileNotFoundError(f"Topology file not found: {base.TOPOLOGY_PATH}")
    topo = load_topology_npz(base.TOPOLOGY_PATH)
    if topo.Nin != Nin:
        raise ValueError(f"Topology Nin={topo.Nin} does not match data Nin={Nin}")
    K = topo.K
    if K != 10:
        raise ValueError(f"Split backend currently expects K=10, got K={K}")

    if vg_init_mode == "fixed":
        vg_unique = np.full((topo.num_edges,), vg_init_single, dtype=float)
    else:
        vg_unique = np.random.uniform(vg_init_lo, vg_init_hi, size=(topo.num_edges,)).astype(float)

    env_run_dir = os.environ.get("RUN_DIR")
    if env_run_dir:
        run_dir = Path(env_run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        runs_dir = RESULTS_ROOT / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + f"_seed-{seed}"
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

    log_f = base.setup_logging(run_dir)

    cfg_str = (
        f"seed={seed} gamma={gamma} delta={delta} margin={margin} "
        f"in=[{vmin},{vmax}] Nin={Nin} K={K} rails=[{vminus_val},{vplus_val}] "
        f"solver={solver} body_tie={body_tie} body_res={body_res} rs_clamp={base.RS_CLAMP} "
        f"vg_init={vg_init_mode} epochs={epochs} "
        f"device_include={base.DEVICE_LIB_PATH} subckt={base.DEVICE_SUBCKT} "
        f"topology={base.TOPOLOGY_PATH.name} split_backend=per_output parallel_sessions={K}"
    )

    print("=== RUN START (scikit_digits_dense_io_hinge_split) ===", flush=True)
    print(cfg_str, flush=True)
    print(f"train={len(train_x)} test={len(test_x)} edges={topo.num_edges}", flush=True)

    full_netlist = base.mk_netlist(
        topo=topo,
        vg_unique=vg_unique,
        vminus_val=vminus_val,
        vplus_val=vplus_val,
        solver=solver,
        body_res=body_res,
        body_tie=body_tie,
    )
    (run_dir / "netlist_initial_full_reference.cir").write_text(full_netlist)

    sessions = build_split_sessions(
        topo=topo,
        vg_unique=vg_unique,
        vminus_val=vminus_val,
        vplus_val=vplus_val,
        solver=solver,
        body_res=body_res,
        body_tie=body_tie,
    )
    for session in sessions:
        (run_dir / f"netlist_split_out{session.out_index}.cir").write_text(session.netlist)

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
        "epochs": epochs,
        "input_vmin": vmin,
        "input_vmax": vmax,
        "rails": {"vminus": vminus_val, "vplus": vplus_val},
        "solver": solver,
        "body_tie": body_tie,
        "body_res": body_res,
        "rs_clamp": base.RS_CLAMP,
        "vg_init": {
            "mode": vg_init_mode,
            "lo": vg_init_lo,
            "hi": vg_init_hi,
            "fixed": vg_init_single,
        },
        "device": {
            "include_path": base.DEVICE_LIB_PATH,
            "subckt": base.DEVICE_SUBCKT,
        },
        "topology": {
            "path": str(base.TOPOLOGY_PATH),
            "Nin": topo.Nin,
            "out": topo.K,
            "edges": topo.num_edges,
            "meta": topo.meta,
        },
        "split_backend": {
            "enabled": True,
            "sessions": int(K),
            "library_pattern": NGSPICE_LIBRARY_PATTERN,
            "library_copy_dir": str(LOCAL_NGSPICE_LIB_DIR),
            "reference_best_run": str(BEST_REFERENCE_RUN),
            "per_output_edge_counts": [int(session.edge_ids.size) for session in sessions],
        },
        "diagnostics": {"vout_saved": "test"},
        "diode_source_to_vplus": False,
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

    np.save(run_dir / "test_x.npy", np.asarray(test_x, dtype=float))
    np.save(run_dir / "test_y.npy", np.asarray(test_y, dtype=int))

    try:
        G = base.nx.DiGraph()
        net_nodes = [topo.negref, topo.posref] + topo.out_nodes.tolist() + topo.input_nodes.tolist()
        G.add_nodes_from(sorted(set(net_nodes)))
        for d, s in zip(topo.edges_D.tolist(), topo.edges_S.tolist()):
            G.add_edge(d, s)
        base.nx.write_graphml(G, str(run_dir / "0.graphml"))
    except Exception:
        pass

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

    with ThreadPoolExecutor(max_workers=K) as executor:
        v0, h0, diag0 = eval_free_metrics(
            executor=executor,
            sessions=sessions,
            vg_unique=vg_unique,
            run_dir=run_dir,
            epoch=0,
            test_x=test_x,
            test_y=test_y,
            K=K,
            margin=margin,
        )
        val_acc_hist.append(v0)
        val_hinge_hist.append(h0)
        np.save(run_dir / "0_val_acc.npy", np.asarray(val_acc_hist, dtype=float))
        np.save(run_dir / "0_val_hinge.npy", np.asarray(val_hinge_hist, dtype=float))
        (run_dir / "0_diag_epoch0.json").write_text(json.dumps(diag0, indent=2))
        print(
            f"[epoch 0] {cfg_str} | VAL acc={v0:.4f} hinge={h0:.6f} "
            f"test_satisfy={diag0.get('test_satisfy_frac', float('nan')):.4f}",
            flush=True,
        )

        for ep in range(1, epochs + 1):
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
                xt = train_x[idx]

                free_t0 = time.time()
                free_results = run_sessions_parallel(
                    executor,
                    lambda session: _run_free_session(session, xt),
                    sessions,
                )
                t_free += float(time.time() - free_t0)
                n_free += 1

                if not all(item[0] for item in free_results):
                    reload_free += 1
                    reload_all_sessions(sessions, vg_unique)
                    continue
                if not all(item[1] for item in free_results):
                    nonfinite_free += 1
                    continue

                Vout = np.asarray([item[2] for item in free_results], dtype=float)
                pred, rival = base.pred_and_rival(Vout, ytrue)
                train_correct += int(pred == ytrue)
                train_total += 1

                hl = base.hinge_loss_from_outputs(Vout, ytrue, margin)
                hinge_sum += float(hl)
                hinge_count += 1

                if hl <= 0.0:
                    skipped += 1
                    continue

                hinge_active += 1

                clamp_t0 = time.time()
                clamp_results = run_sessions_parallel(
                    executor,
                    lambda session: _run_clamp_session(session, xt, ytrue, rival, Vout, delta),
                    sessions,
                )
                t_clamp += float(time.time() - clamp_t0)
                n_clamp += 1

                if not all(item[0] for item in clamp_results):
                    reload_clamp += 1
                    reload_all_sessions(sessions, vg_unique)
                    continue
                if not all(item[1] for item in clamp_results):
                    nonfinite_clamp += 1
                    continue

                Vout_clamp = np.asarray([item[2] for item in clamp_results], dtype=float)

                upd0 = time.time()
                updated_local_gate_values: List[np.ndarray] = []
                for session in sessions:
                    x_local = xt[session.drain_input_idx]
                    v_free = float(Vout[session.out_index])
                    v_clamp = float(Vout_clamp[session.out_index])
                    update = -gamma * ((x_local - v_clamp) ** 2 - (x_local - v_free) ** 2)
                    new_vals = np.clip(
                        vg_unique[session.edge_ids] + update,
                        base.VG_CLIP_LO,
                        base.VG_CLIP_HI,
                    )
                    vg_unique[session.edge_ids] = new_vals
                    updated_local_gate_values.append(new_vals.astype(float))

                list(
                    executor.map(
                        _apply_updated_gates,
                        sessions,
                        updated_local_gate_values,
                    )
                )
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

            v_acc, v_h, diag = eval_free_metrics(
                executor=executor,
                sessions=sessions,
                vg_unique=vg_unique,
                run_dir=run_dir,
                epoch=ep,
                test_x=test_x,
                test_y=test_y,
                K=K,
                margin=margin,
            )
            val_acc_hist.append(float(v_acc))
            val_hinge_hist.append(float(v_h))

            ep_total = float(time.time() - t_ep0)
            ep_total_s.append(ep_total)
            ep_free_s.append(float(t_free))
            ep_clamp_s.append(float(t_clamp))
            ep_update_s.append(float(t_update))

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

            vg_stats = base.compute_vg_saturation_stats(vg_unique)
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
                    "rs_clamp": base.RS_CLAMP,
                    "vg_init": {
                        "mode": vg_init_mode,
                        "lo": vg_init_lo,
                        "hi": vg_init_hi,
                        "fixed": vg_init_single,
                    },
                    "epochs": epochs,
                    "device_include_path": base.DEVICE_LIB_PATH,
                    "device_subckt": base.DEVICE_SUBCKT,
                    "split_backend": True,
                    "parallel_sessions": int(K),
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

            print(
                f"[epoch {ep}/{epochs}] {cfg_str} | "
                f"TRAIN acc={tr_acc:.4f} hinge={tr_h:.6f} hinge_frac={hinge_active_frac:.3f} "
                f"free={n_free} clamp={n_clamp} skipped={skipped} "
                f"reloadF={reload_free} reloadC={reload_clamp} "
                f"nonfiniteF={nonfinite_free} nonfiniteC={nonfinite_clamp}",
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

            try:
                base.save_plots(run_dir)
            except Exception:
                pass

    latest = RESULTS_ROOT / "latest"
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

    print("=== RUN END (scikit_digits_dense_io_hinge_split) ===", flush=True)
    try:
        log_f.flush()
        log_f.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
