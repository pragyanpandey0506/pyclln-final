#!/usr/bin/env python3
"""
Linear-resistor hidden-path KCL variant of the 32-token embed4 one-hot CE trainer.

This keeps the same task, CLI shape, and fresh-process epoch execution model as
`clln_lang_trainer_embed4_onehot_ce.py`, but replaces ngspice with an analytic
nodal solve for a linear resistor network that contains:

  - 24 fixed-voltage input nodes (6 context tokens x 4D embedding)
  - 32 output nodes
  - one direct trainable resistor per input/output pair
  - one additional hidden path per input/output pair:
      input[i] -> hidden[i,k] -> output[k]

Each hidden node is unique to one input/output pair, so it can be eliminated
analytically. The output solve therefore still reduces to 32 independent KCL
equations with an effective input/output conductance:

  g_eff[i,k] = g_dir[i,k] + (g_in_h[i,k] * g_h_out[i,k]) / (g_in_h[i,k] + g_h_out[i,k])

The stored trainable array keeps the existing `vg_unique` naming for run-file
compatibility, but here it stores three clipped resistor-parameter blocks:
direct, input->hidden, and hidden->output.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split

from clln_lang_trainer_embed4_onehot_ce import (
    CONTEXT_LEN,
    DEFAULT_DEVICE_LIB_PATH,
    DEVICE_SUBCKT,
    DenseIOTopology,
    INPUT_DIM,
    INPUT_ID_TO_WORD,
    INPUT_VOCAB,
    OUTPUT_DIM,
    OUTPUT_ID_TO_WORD,
    OUTPUT_VOCAB,
    RS_CLAMP,
    RS_FREE,
    TERMINAL_PUNCT,
    TOKEN_EMBED_4D,
    TOKEN_EMBED_DIM,
    VG_CLIP_HI,
    VG_CLIP_LO,
    VG_INIT_SINGLE,
    build_sentence_corpus,
    build_windows_from_sentences,
    decode_context_ids_to_words,
    encode_context_tokens,
    make_dense_io_topology,
    one_hot,
    pred_label,
    save_plots,
    setup_logging,
    softmax_logits,
    top_words_from_q,
)


EDGE_CONDUCTANCE_SCALE = 1.0e-2
PATH_PARAM_BLOCKS = 3


@dataclass
class EvalOut:
    exact_acc: float
    support_acc: float
    soft_ce: float
    qmass_mean: float
    unseen_contexts: float
    nonfinite_rows: float


@dataclass
class RemodelOut:
    active_edges: int
    pruned_edges: int
    born_edges: int
    utility_threshold: float
    backbone_threshold: float
    active_edge_fraction: float
    utility_active_mean: float
    backbone_active_mean: float


def build_context_target_distributions(
    ctx_keys: Sequence[Tuple[int, ...]],
    ys: Sequence[int],
    K: int,
) -> Dict[Tuple[int, ...], np.ndarray]:
    counts: Dict[Tuple[int, ...], np.ndarray] = {}
    for ctx, y in zip(ctx_keys, ys):
        if ctx not in counts:
            counts[ctx] = np.zeros(K, dtype=float)
        counts[ctx][int(y)] += 1.0

    q_map: Dict[Tuple[int, ...], np.ndarray] = {}
    for ctx, c in counts.items():
        s = float(np.sum(c))
        q_map[ctx] = c / s if s > 0.0 else np.full(K, 1.0 / K, dtype=float)
    return q_map


def build_unigram_target_distribution(ys: Sequence[int], K: int) -> np.ndarray:
    c = np.zeros(K, dtype=float)
    for y in ys:
        c[int(y)] += 1.0
    s = float(np.sum(c))
    return c / s if s > 0.0 else np.full(K, 1.0 / K, dtype=float)


def build_q_matrix(
    ctx_list: Sequence[Tuple[int, ...]],
    q_map: Dict[Tuple[int, ...], np.ndarray],
    fallback_q: np.ndarray,
) -> Tuple[np.ndarray, int]:
    out = np.zeros((len(ctx_list), OUTPUT_DIM), dtype=float)
    unseen = 0
    for i, c in enumerate(ctx_list):
        if c in q_map:
            out[i, :] = q_map[c]
        else:
            out[i, :] = fallback_q
            unseen += 1
    return out, unseen


def parse_args():
    p = argparse.ArgumentParser(
        description="CLLN dense 24->32 language trainer, fixed 4D token embeddings, one-hot CE, linear resistor hidden-path KCL backend"
    )
    p.add_argument("seed", type=int, nargs="?", default=0)
    p.add_argument("--epochs", type=int, default=20)

    p.add_argument("--gamma", type=float, default=0.30)
    p.add_argument("--delta", type=float, default=0.30)
    p.add_argument("--softmax-temp", type=float, default=1.0)

    p.add_argument("--vminus", type=float, default=0.0)
    p.add_argument("--vplus", type=float, default=0.45)

    p.add_argument("--num-sentences", type=int, default=12000)
    p.add_argument("--min-target-count", type=int, default=80)
    p.add_argument("--max-sentence-words", type=int, default=9)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--max-train", type=int, default=0)
    p.add_argument("--max-val", type=int, default=0)
    p.add_argument(
        "--template-mode",
        type=str,
        choices=["balanced", "broad"],
        default="broad",
        help="balanced = roughly uniform template sampling; broad = slightly reweighted to emphasize punctuation, negation, questions, and comma clauses",
    )

    # Kept for CLI compatibility with the analog trainer. They are recorded in
    # metadata but do not affect the linear-resistor backend.
    p.add_argument("--device-lib", type=str, default=DEFAULT_DEVICE_LIB_PATH)
    p.add_argument("--body-tie", type=str, choices=["source", "ground", "floating"], default="ground")
    p.add_argument("--solver", type=str, choices=["klu", "sparse"], default="klu")

    p.add_argument("--vg-init", type=str, choices=["random", "fixed"], default="random")
    p.add_argument("--vg-init-lo", type=float, default=1.0)
    p.add_argument("--vg-init-hi", type=float, default=3.0)
    p.add_argument("--vg-init-fixed", type=float, default=VG_INIT_SINGLE)

    p.add_argument("--sample-prompts", type=int, default=8)
    p.add_argument("--sample-max-len", type=int, default=12)
    p.add_argument("--final-val", action="store_true")
    p.add_argument("--process-mode", type=str, choices=["fresh_process", "in_process"], default="fresh_process")
    p.add_argument("--eval-every", type=int, default=1, help="Evaluate train/val metrics every N epochs. Final epoch is always evaluated.")
    p.add_argument("--sample-every", type=int, default=1, help="Write sample generations every N evaluated epochs. Use 0 to disable.")
    p.add_argument("--plot-every", type=int, default=1, help="Refresh plots every N evaluated epochs. Use 0 to disable.")
    p.add_argument("--remodel-every-epochs", type=int, default=0)
    p.add_argument("--utility-beta", type=float, default=0.999)
    p.add_argument("--prune-utility-quantile", type=float, default=0.10)
    p.add_argument("--prune-backbone-quantile", type=float, default=0.10)
    p.add_argument("--prune-rand-prob", type=float, default=0.0)
    p.add_argument("--max-prune-frac", type=float, default=0.03)
    p.add_argument("--min-edge-age", type=int, default=3)
    p.add_argument("--birth-vg-lo", type=float, default=VG_CLIP_LO)
    p.add_argument("--birth-vg-hi", type=float, default=VG_CLIP_LO)
    p.add_argument("--worker-run-dir", type=str, default="", help=argparse.SUPPRESS)
    p.add_argument("--worker-epoch", type=int, default=-1, help=argparse.SUPPRESS)
    return p.parse_args()


def softmax_rows(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    z = z - np.max(z, axis=1, keepdims=True)
    ez = np.exp(z)
    s = np.sum(ez, axis=1, keepdims=True)
    s = np.where((~np.isfinite(s)) | (s <= 0.0), 1.0, s)
    return ez / s


def view_param_blocks(vg_unique: np.ndarray, topo: DenseIOTopology) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat = np.asarray(vg_unique, dtype=float).reshape(-1)
    n = topo.K * topo.Nin
    expected = PATH_PARAM_BLOCKS * n
    if flat.size != expected:
        raise ValueError(f"Expected {expected} stored parameters, got {flat.size}")
    return (
        flat[0:n].reshape(topo.K, topo.Nin),
        flat[n : 2 * n].reshape(topo.K, topo.Nin),
        flat[2 * n : 3 * n].reshape(topo.K, topo.Nin),
    )


def view_mask_blocks(edge_active: np.ndarray, topo: DenseIOTopology) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    flat = np.asarray(edge_active, dtype=bool).reshape(-1)
    n = topo.K * topo.Nin
    expected = PATH_PARAM_BLOCKS * n
    if flat.size != expected:
        raise ValueError(f"Expected {expected} edge-activity entries, got {flat.size}")
    return (
        flat[0:n].reshape(topo.K, topo.Nin),
        flat[n : 2 * n].reshape(topo.K, topo.Nin),
        flat[2 * n : 3 * n].reshape(topo.K, topo.Nin),
    )


def params_to_conductances(
    vg_unique: np.ndarray,
    topo: DenseIOTopology,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    direct_params, in_hidden_params, hidden_out_params = view_param_blocks(vg_unique, topo)
    g_direct = EDGE_CONDUCTANCE_SCALE * np.asarray(direct_params, dtype=float)
    g_in_hidden = EDGE_CONDUCTANCE_SCALE * np.asarray(in_hidden_params, dtype=float)
    g_hidden_out = EDGE_CONDUCTANCE_SCALE * np.asarray(hidden_out_params, dtype=float)
    den = g_in_hidden + g_hidden_out
    g_series = np.where(den > 1e-12, (g_in_hidden * g_hidden_out) / den, 0.0)
    g_total = g_direct + g_series
    return g_direct, g_in_hidden, g_hidden_out, g_total


def params_to_masked_conductances(
    vg_unique: np.ndarray,
    edge_active: np.ndarray,
    topo: DenseIOTopology,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    direct_params, in_hidden_params, hidden_out_params = view_param_blocks(vg_unique, topo)
    active_direct, active_in_hidden, active_hidden_out = view_mask_blocks(edge_active, topo)
    g_direct = EDGE_CONDUCTANCE_SCALE * np.asarray(direct_params, dtype=float) * np.asarray(active_direct, dtype=float)
    g_in_hidden = EDGE_CONDUCTANCE_SCALE * np.asarray(in_hidden_params, dtype=float) * np.asarray(active_in_hidden, dtype=float)
    g_hidden_out = EDGE_CONDUCTANCE_SCALE * np.asarray(hidden_out_params, dtype=float) * np.asarray(active_hidden_out, dtype=float)
    den = g_in_hidden + g_hidden_out
    g_series = np.where(den > 1e-12, (g_in_hidden * g_hidden_out) / den, 0.0)
    g_total = g_direct + g_series
    return g_direct, g_in_hidden, g_hidden_out, g_total


def compute_hidden_voltages(
    *,
    g_in_hidden: np.ndarray,
    g_hidden_out: np.ndarray,
    xin: np.ndarray,
    vout: np.ndarray,
) -> np.ndarray:
    xin = np.asarray(xin, dtype=float).reshape(1, -1)
    vout = np.asarray(vout, dtype=float).reshape(-1, 1)
    den = g_in_hidden + g_hidden_out
    fallback = 0.5 * (xin + vout)
    return np.where(den > 1e-12, (g_in_hidden * xin + g_hidden_out * vout) / den, fallback)


def solve_outputs(
    *,
    gmat: np.ndarray,
    xin: np.ndarray,
    clamp_res: float,
    clamp_target: np.ndarray | None,
    sum_g: np.ndarray | None = None,
) -> np.ndarray:
    xin = np.asarray(xin, dtype=float).reshape(-1)
    if sum_g is None:
        sum_g = np.sum(gmat, axis=1)
    g_clamp = (1.0 / float(clamp_res)) if float(clamp_res) > 0.0 else 0.0

    num = gmat @ xin
    if clamp_target is not None and g_clamp > 0.0:
        num = num + g_clamp * np.asarray(clamp_target, dtype=float).reshape(-1)

    den = sum_g + g_clamp
    den = np.where(den <= 0.0, 1e-12, den)
    return num / den


def solve_outputs_batch_free(gmat: np.ndarray, X: np.ndarray, sum_g: np.ndarray | None = None) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if sum_g is None:
        sum_g = np.sum(gmat, axis=1)
    g_free = 1.0 / RS_FREE
    den = sum_g + g_free
    den = np.where(den <= 0.0, 1e-12, den)
    return (X @ gmat.T) / den[None, :]


def clamp_targets_from_free(Vout_free: np.ndarray, q_target: np.ndarray, delta: float, temp: float) -> np.ndarray:
    if temp <= 0.0:
        raise ValueError("--softmax-temp must be > 0")
    p = softmax_logits(np.asarray(Vout_free, dtype=float) / float(temp))
    q = np.asarray(q_target, dtype=float)
    q_sum = float(np.sum(q))
    if q_sum <= 0.0:
        raise ValueError("q_target must have positive mass")
    q = q / q_sum
    return np.asarray(Vout_free, dtype=float) + float(delta) * (q - p)


def edge_diffs_from_outputs(Vout: np.ndarray, xin: np.ndarray) -> np.ndarray:
    return (np.asarray(Vout, dtype=float)[:, None] - np.asarray(xin, dtype=float)[None, :]).reshape(-1)


def compute_vg_saturation_stats(vg_unique: np.ndarray, edge_active: np.ndarray | None = None) -> Dict[str, float]:
    g_unique = EDGE_CONDUCTANCE_SCALE * np.asarray(vg_unique, dtype=float)
    stats = {
        "vg_unique_min": float(np.min(vg_unique)),
        "vg_unique_max": float(np.max(vg_unique)),
        "vg_unique_sat_lo": float(np.sum(vg_unique <= (VG_CLIP_LO + 1e-12))),
        "vg_unique_sat_hi": float(np.sum(vg_unique >= (VG_CLIP_HI - 1e-12))),
        "conductance_min": float(np.min(g_unique)),
        "conductance_max": float(np.max(g_unique)),
        "conductance_scale": float(EDGE_CONDUCTANCE_SCALE),
        "param_blocks": float(PATH_PARAM_BLOCKS),
    }
    if edge_active is not None:
        active = np.asarray(edge_active, dtype=bool)
        active_g = g_unique[active] if np.any(active) else np.asarray([], dtype=float)
        stats["active_edges"] = float(np.sum(active))
        stats["inactive_edges"] = float(active.size - np.sum(active))
        stats["active_conductance_mean"] = float(np.mean(active_g)) if active_g.size else 0.0
    return stats


def build_cfg_str(
    *,
    seed: int,
    gamma: float,
    delta: float,
    temp: float,
    topo: DenseIOTopology,
    num_sentences_actual: int,
    min_target_count: int,
    max_sentence_words: int,
    template_mode: str,
    vminus_val: float,
    vplus_val: float,
    solver: str,
    body_tie: str,
    vg_init_mode: str,
    epochs: int,
    device_lib: str,
    process_mode: str,
    eval_every: int,
    remodel_every_epochs: int,
) -> str:
    return (
        f"seed={seed} gamma={gamma} delta={delta} T={temp} "
        f"context={CONTEXT_LEN} embed_dim={TOKEN_EMBED_DIM} Nin={topo.Nin} K={topo.K} "
        f"sentences={num_sentences_actual} min_target_count={min_target_count} max_words={max_sentence_words} template={template_mode} "
        f"rails=[{vminus_val},{vplus_val}] solver={solver} body_tie={body_tie} rs_clamp={RS_CLAMP} "
        f"vg_init={vg_init_mode} epochs={epochs} device_include={device_lib} subckt={DEVICE_SUBCKT} "
        f"backend=linear_resistor_hidden_path_kcl g_scale={EDGE_CONDUCTANCE_SCALE} "
        f"param_blocks={PATH_PARAM_BLOCKS} process_mode={process_mode} "
        f"eval_every={eval_every} remodel_every={remodel_every_epochs}"
    )


def load_hist_list(run_dir: Path, name: str, dtype=float) -> List[float]:
    path = run_dir / name
    if not path.exists():
        return []
    arr = np.asarray(np.load(path), dtype=dtype)
    return arr.tolist()


def should_run_interval(epoch: int, interval: int, is_final_epoch: bool) -> bool:
    if is_final_epoch:
        return True
    if interval <= 0:
        return False
    return (epoch % interval) == 0


def make_nan_evalout() -> EvalOut:
    return EvalOut(
        exact_acc=float("nan"),
        support_acc=float("nan"),
        soft_ce=float("nan"),
        qmass_mean=float("nan"),
        unseen_contexts=float("nan"),
        nonfinite_rows=float("nan"),
    )


def make_nan_diag(prefix: str) -> Dict[str, float]:
    return {
        f"{prefix}_reloads": float("nan"),
        f"{prefix}_nonfinite": float("nan"),
        f"{prefix}_qmass_mean": float("nan"),
        f"{prefix}_support_acc": float("nan"),
        f"{prefix}_unseen_contexts": float("nan"),
    }


def save_history_arrays(
    run_dir: Path,
    *,
    tr_acc_hist: Sequence[float],
    tr_support_hist: Sequence[float],
    tr_ce_hist: Sequence[float],
    tr_qmass_hist: Sequence[float],
    val_acc_hist: Sequence[float],
    val_support_hist: Sequence[float],
    val_ce_hist: Sequence[float],
    val_qmass_hist: Sequence[float],
    val_unseen_hist: Sequence[float],
    ep_total_s: Sequence[float],
    ep_free_s: Sequence[float],
    ep_clamp_s: Sequence[float],
    ep_update_s: Sequence[float],
    reload_free_hist: Sequence[int],
    reload_clamp_hist: Sequence[int],
    nonfinite_free_hist: Sequence[int],
    nonfinite_clamp_hist: Sequence[int],
    active_edges_hist: Sequence[int],
    pruned_edges_hist: Sequence[int],
    born_edges_hist: Sequence[int],
    utility_thresh_hist: Sequence[float],
    backbone_thresh_hist: Sequence[float],
):
    np.save(run_dir / "0_train_acc.npy", np.asarray(tr_acc_hist, dtype=float))
    np.save(run_dir / "0_train_support_acc.npy", np.asarray(tr_support_hist, dtype=float))
    np.save(run_dir / "0_train_ce.npy", np.asarray(tr_ce_hist, dtype=float))
    np.save(run_dir / "0_train_qmass.npy", np.asarray(tr_qmass_hist, dtype=float))
    np.save(run_dir / "0_val_acc.npy", np.asarray(val_acc_hist, dtype=float))
    np.save(run_dir / "0_val_support_acc.npy", np.asarray(val_support_hist, dtype=float))
    np.save(run_dir / "0_val_ce.npy", np.asarray(val_ce_hist, dtype=float))
    np.save(run_dir / "0_val_qmass.npy", np.asarray(val_qmass_hist, dtype=float))
    np.save(run_dir / "0_val_unseen_contexts.npy", np.asarray(val_unseen_hist, dtype=float))
    np.save(run_dir / "0_epoch_total_s.npy", np.asarray(ep_total_s, dtype=float))
    np.save(run_dir / "0_epoch_free_s.npy", np.asarray(ep_free_s, dtype=float))
    np.save(run_dir / "0_epoch_clamp_s.npy", np.asarray(ep_clamp_s, dtype=float))
    np.save(run_dir / "0_epoch_update_s.npy", np.asarray(ep_update_s, dtype=float))
    np.save(run_dir / "0_reload_free.npy", np.asarray(reload_free_hist, dtype=int))
    np.save(run_dir / "0_reload_clamp.npy", np.asarray(reload_clamp_hist, dtype=int))
    np.save(run_dir / "0_nonfinite_free.npy", np.asarray(nonfinite_free_hist, dtype=int))
    np.save(run_dir / "0_nonfinite_clamp.npy", np.asarray(nonfinite_clamp_hist, dtype=int))
    np.save(run_dir / "0_active_edges.npy", np.asarray(active_edges_hist, dtype=int))
    np.save(run_dir / "0_pruned_edges.npy", np.asarray(pruned_edges_hist, dtype=int))
    np.save(run_dir / "0_born_edges.npy", np.asarray(born_edges_hist, dtype=int))
    np.save(run_dir / "0_utility_threshold.npy", np.asarray(utility_thresh_hist, dtype=float))
    np.save(run_dir / "0_backbone_threshold.npy", np.asarray(backbone_thresh_hist, dtype=float))


def infer_context_ids_from_x(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != INPUT_DIM:
        raise ValueError(f"Expected X with shape (N, {INPUT_DIM}), got {X.shape}")

    embed_to_idx: Dict[Tuple[float, ...], int] = {}
    for idx, tok in enumerate(INPUT_VOCAB):
        embed = tuple(np.round(np.asarray(TOKEN_EMBED_4D[tok], dtype=float), 8).tolist())
        embed_to_idx[embed] = idx

    ctx = np.zeros((X.shape[0], CONTEXT_LEN), dtype=int)
    for row_idx in range(X.shape[0]):
        row = X[row_idx]
        ids: List[int] = []
        for pos in range(CONTEXT_LEN):
            sl = row[pos * TOKEN_EMBED_DIM : (pos + 1) * TOKEN_EMBED_DIM]
            key = tuple(np.round(np.asarray(sl, dtype=float), 8).tolist())
            tok_idx = embed_to_idx.get(key)
            if tok_idx is None:
                matched = None
                for cand_key, cand_idx in embed_to_idx.items():
                    if np.allclose(np.asarray(cand_key, dtype=float), sl, atol=1e-8, rtol=0.0):
                        matched = cand_idx
                        break
                if matched is None:
                    raise ValueError(f"Could not reconstruct context token from saved train_x row={row_idx} pos={pos}")
                tok_idx = matched
            ids.append(int(tok_idx))
        ctx[row_idx, :] = np.asarray(ids, dtype=int)
    return ctx


def load_saved_dataset(run_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_x = np.asarray(np.load(run_dir / "train_x.npy"), dtype=float)
    train_y = np.asarray(np.load(run_dir / "train_y.npy"), dtype=int)
    train_ctx_path = run_dir / "train_ctx.npy"
    if train_ctx_path.exists():
        train_ctx = np.asarray(np.load(train_ctx_path), dtype=int)
    else:
        train_ctx = infer_context_ids_from_x(train_x)
    val_x = np.asarray(np.load(run_dir / "val_x.npy"), dtype=float)
    val_y = np.asarray(np.load(run_dir / "val_y.npy"), dtype=int)
    val_ctx = np.asarray(np.load(run_dir / "val_ctx.npy"), dtype=int)
    return train_x, train_y, train_ctx, val_x, val_y, val_ctx


def load_hidden_state(
    run_dir: Path,
    epoch: int,
    topo: DenseIOTopology,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = PATH_PARAM_BLOCKS * topo.num_edges
    edge_active_path = run_dir / f"0_edge_active_epoch{epoch}.npy"
    utility_path = run_dir / f"0_edge_utility_epoch{epoch}.npy"
    backbone_path = run_dir / f"0_edge_backbone_epoch{epoch}.npy"
    age_path = run_dir / f"0_edge_age_epoch{epoch}.npy"
    if edge_active_path.exists():
        edge_active = np.asarray(np.load(edge_active_path), dtype=bool).reshape(n)
    else:
        edge_active = np.ones((n,), dtype=bool)
    if utility_path.exists():
        utility_ema = np.asarray(np.load(utility_path), dtype=float).reshape(n)
    else:
        utility_ema = np.zeros((n,), dtype=float)
    if backbone_path.exists():
        backbone_ema = np.asarray(np.load(backbone_path), dtype=float).reshape(n)
    else:
        backbone_ema = np.zeros((n,), dtype=float)
    if age_path.exists():
        edge_age = np.asarray(np.load(age_path), dtype=int).reshape(n)
    else:
        edge_age = np.zeros((n,), dtype=int)
    return edge_active, utility_ema, backbone_ema, edge_age


def remodel_hidden_edges(
    *,
    topo: DenseIOTopology,
    vg_unique: np.ndarray,
    edge_active: np.ndarray,
    utility_ema: np.ndarray,
    backbone_ema: np.ndarray,
    edge_age: np.ndarray,
    prune_utility_quantile: float,
    prune_backbone_quantile: float,
    prune_rand_prob: float,
    max_prune_frac: float,
    min_edge_age: int,
    birth_vg_lo: float,
    birth_vg_hi: float,
    rng: np.random.Generator,
) -> RemodelOut:
    total_edges = int(edge_active.size)
    eligible = np.asarray(edge_active, dtype=bool) & (np.asarray(edge_age, dtype=int) >= int(min_edge_age))

    utility_threshold = float("nan")
    backbone_threshold = float("nan")
    pruned_edges = 0
    born_edges = 0

    if np.any(eligible):
        utility_threshold = float(np.quantile(utility_ema[eligible], np.clip(prune_utility_quantile, 0.0, 1.0)))
        backbone_threshold = float(np.quantile(backbone_ema[eligible], np.clip(prune_backbone_quantile, 0.0, 1.0)))
        deterministic = eligible & (utility_ema <= utility_threshold) & (backbone_ema <= backbone_threshold)
        random_mask = eligible & (rng.random(total_edges) < float(prune_rand_prob))
        candidates = np.flatnonzero(deterministic | random_mask)
        if candidates.size > 0:
            max_prunes = max(0, int(np.floor(float(max_prune_frac) * max(1, int(np.sum(edge_active))))))
            score = utility_ema[candidates] + backbone_ema[candidates]
            order = candidates[np.argsort(score + 1e-12 * rng.random(candidates.size))]
            for edge_idx in order.tolist():
                if pruned_edges >= max_prunes:
                    break
                edge_active[edge_idx] = False
                edge_age[edge_idx] = 0
                pruned_edges += 1

    births_needed = int(pruned_edges)
    if births_needed > 0:
        inactive = np.flatnonzero(~edge_active)
        if inactive.size > 0:
            birth_count = min(int(births_needed), int(inactive.size))
            birth_idx = rng.choice(inactive, size=birth_count, replace=False)
            edge_active[birth_idx] = True
            edge_age[birth_idx] = 0
            flat = np.asarray(vg_unique, dtype=float).reshape(-1)
            flat[np.asarray(birth_idx, dtype=int)] = rng.uniform(float(birth_vg_lo), float(birth_vg_hi), size=birth_count)
            born_edges = int(birth_count)

    active_mask = np.asarray(edge_active, dtype=bool)
    utility_active_mean = float(np.mean(utility_ema[active_mask])) if np.any(active_mask) else 0.0
    backbone_active_mean = float(np.mean(backbone_ema[active_mask])) if np.any(active_mask) else 0.0
    return RemodelOut(
        active_edges=int(np.sum(active_mask)),
        pruned_edges=pruned_edges,
        born_edges=born_edges,
        utility_threshold=utility_threshold,
        backbone_threshold=backbone_threshold,
        active_edge_fraction=float(np.mean(active_mask.astype(float))),
        utility_active_mean=utility_active_mean,
        backbone_active_mean=backbone_active_mean,
    )


def eval_free_metrics(
    *,
    topo: DenseIOTopology,
    vg_unique: np.ndarray,
    edge_active: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    ctx: Sequence[Tuple[int, ...]],
    q_map_train: Dict[Tuple[int, ...], np.ndarray],
    unigram_q_train: np.ndarray,
    temp: float,
    run_dir: Path | None = None,
    epoch: int | None = None,
    split_name: str = "val",
) -> Tuple[EvalOut, Dict[str, float]]:
    _, _, _, gmat = params_to_masked_conductances(vg_unique, edge_active, topo)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    Vout = solve_outputs_batch_free(gmat, X, sum_g=np.sum(gmat, axis=1))
    n_rows = int(X.shape[0])

    if not np.all(np.isfinite(Vout)):
        nonfinite_rows = ~np.all(np.isfinite(Vout), axis=1)
    else:
        nonfinite_rows = np.zeros((n_rows,), dtype=bool)

    safe_vout = Vout.copy()
    safe_vout[nonfinite_rows] = 0.0

    probs = softmax_rows(safe_vout / float(temp))
    pred = np.argmax(safe_vout, axis=1)
    valid_mask = ~nonfinite_rows

    q, unseen = build_q_matrix(ctx, q_map_train, unigram_q_train)

    confusion = np.zeros((topo.K, topo.K), dtype=int)
    for ytrue, ypred, is_valid in zip(y.tolist(), pred.tolist(), valid_mask.tolist()):
        if is_valid:
            confusion[int(ytrue), int(ypred)] += 1

    if run_dir is not None and epoch is not None and split_name == "val" and n_rows > 0:
        np.save(run_dir / f"0_vout_val_epoch{epoch}.npy", Vout)
        np.save(run_dir / f"0_val_confusion_epoch{epoch}.npy", confusion)

    if not np.any(valid_mask):
        out = EvalOut(
            exact_acc=float("nan"),
            support_acc=float("nan"),
            soft_ce=float("nan"),
            qmass_mean=float("nan"),
            unseen_contexts=float(unseen),
            nonfinite_rows=float(np.sum(nonfinite_rows)),
        )
        return out, {
            f"{split_name}_reloads": 0.0,
            f"{split_name}_nonfinite": float(np.sum(nonfinite_rows)),
            f"{split_name}_qmass_mean": float("nan"),
            f"{split_name}_support_acc": float("nan"),
            f"{split_name}_unseen_contexts": float(unseen),
        }

    q_valid = q[valid_mask]
    probs_valid = np.clip(probs[valid_mask], 1e-12, 1.0)
    y_valid = y[valid_mask]
    pred_valid = pred[valid_mask]

    exact_acc = float(np.mean(pred_valid == y_valid))
    support_acc = float(np.mean((q_valid[np.arange(q_valid.shape[0]), pred_valid] > 0.0).astype(float)))
    soft_ce = float(np.mean(-np.sum(q_valid * np.log(probs_valid), axis=1)))
    qmass_mean = float(np.mean(np.sum(probs_valid * (q_valid > 0.0), axis=1)))
    out = EvalOut(
        exact_acc=exact_acc,
        support_acc=support_acc,
        soft_ce=soft_ce,
        qmass_mean=qmass_mean,
        unseen_contexts=float(unseen),
        nonfinite_rows=float(np.sum(nonfinite_rows)),
    )
    diag = {
        f"{split_name}_reloads": 0.0,
        f"{split_name}_nonfinite": float(np.sum(nonfinite_rows)),
        f"{split_name}_qmass_mean": qmass_mean,
        f"{split_name}_support_acc": support_acc,
        f"{split_name}_unseen_contexts": float(unseen),
    }
    return out, diag


def greedy_generate_from_context(
    *,
    topo: DenseIOTopology,
    vg_unique: np.ndarray,
    edge_active: np.ndarray,
    seed_ctx_ids: List[int],
    max_len: int,
) -> List[str]:
    ctx = list(seed_ctx_ids)
    out_words: List[str] = []
    _, _, _, gmat = params_to_masked_conductances(vg_unique, edge_active, topo)
    sum_g = np.sum(gmat, axis=1)
    for _ in range(max_len):
        xin = encode_context_tokens(ctx)
        Vout = solve_outputs(gmat=gmat, xin=xin, clamp_res=RS_FREE, clamp_target=None, sum_g=sum_g)
        if not np.all(np.isfinite(Vout)):
            break
        yhat = pred_label(Vout)
        w = OUTPUT_ID_TO_WORD[int(yhat)]
        out_words.append(w)
        ctx = ctx[1:] + [INPUT_VOCAB.index(w)]
        if w in TERMINAL_PUNCT:
            break
    return out_words


def save_generation_samples(
    *,
    args: argparse.Namespace,
    topo: DenseIOTopology,
    vg_unique: np.ndarray,
    edge_active: np.ndarray,
    q_map_train: Dict[Tuple[int, ...], np.ndarray],
    unigram_q_train: np.ndarray,
    val_ctx: np.ndarray,
    val_y: np.ndarray,
    run_dir: Path,
    epoch: int,
):
    n = min(int(args.sample_prompts), len(val_ctx))
    if n <= 0:
        return

    lines: List[str] = []
    json_items: List[Dict[str, object]] = []
    prompt_indices = np.linspace(0, len(val_ctx) - 1, n, dtype=int)
    for j, idx in enumerate(prompt_indices.tolist()):
        seed_ids = [int(v) for v in val_ctx[idx].tolist()]
        seed_words = decode_context_ids_to_words(seed_ids)
        generated = greedy_generate_from_context(
            topo=topo,
            vg_unique=vg_unique,
            edge_active=edge_active,
            seed_ctx_ids=seed_ids,
            max_len=int(args.sample_max_len),
        )
        target = OUTPUT_ID_TO_WORD[int(val_y[idx])]
        q_target = q_map_train.get(tuple(seed_ids), unigram_q_train)
        valid_q = {
            OUTPUT_ID_TO_WORD[k]: float(q_target[k])
            for k in range(topo.K)
            if float(q_target[k]) > 0.0
        }
        seed_clean = [w for w in seed_words if w != "<BOS>"]
        gen_clean = [w for w in generated if w != "<BOS>"]
        lines.append(f"[{j}] seed={' '.join(seed_clean) if seed_clean else '<BOS>'}")
        lines.append(f"    observed_next={target}")
        lines.append(f"    valid_next_dist_top={top_words_from_q(q_target, topk=6)}")
        lines.append(f"    generated={' '.join(gen_clean) if gen_clean else '<empty>'}")
        json_items.append({
            "context": seed_clean if seed_clean else ["<BOS>"],
            "observed_next": target,
            "valid_next_token_distribution": valid_q,
            "valid_next_token_distribution_top": top_words_from_q(q_target, topk=6),
            "generated": gen_clean,
        })

    (run_dir / f"samples_epoch{epoch}.txt").write_text("\n".join(lines) + "\n")
    (run_dir / f"samples_epoch{epoch}.json").write_text(json.dumps(json_items, indent=2))


def build_worker_cmd(args: argparse.Namespace, run_dir: Path, epoch: int) -> List[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        str(int(args.seed)),
        "--epochs", str(int(args.epochs)),
        "--gamma", str(float(args.gamma)),
        "--delta", str(float(args.delta)),
        "--softmax-temp", str(float(args.softmax_temp)),
        "--vminus", str(float(args.vminus)),
        "--vplus", str(float(args.vplus)),
        "--num-sentences", str(int(args.num_sentences)),
        "--min-target-count", str(int(args.min_target_count)),
        "--max-sentence-words", str(int(args.max_sentence_words)),
        "--val-frac", str(float(args.val_frac)),
        "--max-train", str(int(args.max_train)),
        "--max-val", str(int(args.max_val)),
        "--template-mode", str(args.template_mode),
        "--device-lib", str(args.device_lib),
        "--body-tie", str(args.body_tie),
        "--solver", str(args.solver),
        "--vg-init", str(args.vg_init),
        "--vg-init-lo", str(float(args.vg_init_lo)),
        "--vg-init-hi", str(float(args.vg_init_hi)),
        "--vg-init-fixed", str(float(args.vg_init_fixed)),
        "--sample-prompts", str(int(args.sample_prompts)),
        "--sample-max-len", str(int(args.sample_max_len)),
        "--process-mode", str(args.process_mode),
        "--eval-every", str(int(args.eval_every)),
        "--sample-every", str(int(args.sample_every)),
        "--plot-every", str(int(args.plot_every)),
        "--remodel-every-epochs", str(int(args.remodel_every_epochs)),
        "--utility-beta", str(float(args.utility_beta)),
        "--prune-utility-quantile", str(float(args.prune_utility_quantile)),
        "--prune-backbone-quantile", str(float(args.prune_backbone_quantile)),
        "--prune-rand-prob", str(float(args.prune_rand_prob)),
        "--max-prune-frac", str(float(args.max_prune_frac)),
        "--min-edge-age", str(int(args.min_edge_age)),
        "--birth-vg-lo", str(float(args.birth_vg_lo)),
        "--birth-vg-hi", str(float(args.birth_vg_hi)),
        "--worker-run-dir", str(run_dir),
        "--worker-epoch", str(int(epoch)),
    ]
    return cmd


def run_worker_epoch(args: argparse.Namespace) -> None:
    run_dir = Path(args.worker_run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Worker run directory not found: {run_dir}")

    log_f = setup_logging(run_dir)
    seed = int(args.seed)
    epoch = int(args.worker_epoch)
    random.seed(seed + max(epoch, 0))
    np.random.seed(seed + max(epoch, 0))

    epochs = int(args.epochs)
    gamma = float(args.gamma)
    delta = float(args.delta)
    temp = float(args.softmax_temp)
    if temp <= 0.0:
        raise ValueError("--softmax-temp must be > 0")

    vminus_val = float(args.vminus)
    vplus_val = float(args.vplus)
    solver = str(args.solver).lower()
    body_tie = str(args.body_tie)
    vg_init_mode = str(args.vg_init)
    vg_init_lo = float(args.vg_init_lo)
    vg_init_hi = float(args.vg_init_hi)
    vg_init_single = float(args.vg_init_fixed)
    device_lib = str(args.device_lib)
    process_mode = str(args.process_mode)
    eval_every = int(args.eval_every)
    sample_every = int(args.sample_every)
    plot_every = int(args.plot_every)
    remodel_every_epochs = int(args.remodel_every_epochs)
    utility_beta = float(args.utility_beta)
    prune_utility_quantile = float(args.prune_utility_quantile)
    prune_backbone_quantile = float(args.prune_backbone_quantile)
    prune_rand_prob = float(args.prune_rand_prob)
    max_prune_frac = float(args.max_prune_frac)
    min_edge_age = int(args.min_edge_age)
    birth_vg_lo = float(args.birth_vg_lo)
    birth_vg_hi = float(args.birth_vg_hi)

    template_mode = str(args.template_mode)
    min_target_count = int(args.min_target_count)
    max_sentence_words = int(args.max_sentence_words)

    train_x, train_y, train_ctx_arr, val_x, val_y, val_ctx = load_saved_dataset(run_dir)
    train_ctx = [tuple(int(v) for v in row.tolist()) for row in np.asarray(train_ctx_arr, dtype=int)]
    val_ctx_list = [tuple(int(v) for v in row.tolist()) for row in np.asarray(val_ctx, dtype=int)]
    val_n = int(val_x.shape[0])
    q_map_train = build_context_target_distributions(train_ctx, train_y.tolist(), OUTPUT_DIM)
    unigram_q_train = build_unigram_target_distribution(train_y.tolist(), OUTPUT_DIM)

    topo = make_dense_io_topology()
    meta = json.loads((run_dir / "run_meta.json").read_text())
    cfg_str = build_cfg_str(
        seed=seed,
        gamma=gamma,
        delta=delta,
        temp=temp,
        topo=topo,
        num_sentences_actual=int(meta["dataset"]["num_sentences_actual"]),
        min_target_count=min_target_count,
        max_sentence_words=max_sentence_words,
        template_mode=template_mode,
        vminus_val=vminus_val,
        vplus_val=vplus_val,
        solver=solver,
        body_tie=body_tie,
        vg_init_mode=vg_init_mode,
        epochs=epochs,
        device_lib=device_lib,
        process_mode=process_mode,
        eval_every=eval_every,
        remodel_every_epochs=remodel_every_epochs,
    )

    prev_epoch = 0 if epoch <= 0 else epoch - 1
    vg_path = run_dir / f"0_vg_unique_epoch{prev_epoch}.npy"
    if not vg_path.exists():
        raise FileNotFoundError(f"Required parameter-state file not found: {vg_path}")
    vg_unique = np.asarray(np.load(vg_path), dtype=float)
    direct_params, in_hidden_params, hidden_out_params = view_param_blocks(vg_unique, topo)
    edge_active, utility_ema, backbone_ema, edge_age = load_hidden_state(run_dir, prev_epoch, topo)
    rng = np.random.default_rng(seed + 1009 * max(epoch, 0) + 17)

    if epoch == 0:
        tr_support_hist: List[float] = []
        val_acc_hist: List[float] = []
        val_support_hist: List[float] = []
        val_ce_hist: List[float] = []
        tr_qmass_hist: List[float] = []
        val_qmass_hist: List[float] = []
        val_unseen_hist: List[float] = []
        v0, diag0 = eval_free_metrics(
            topo=topo,
            vg_unique=vg_unique,
            edge_active=edge_active,
            X=val_x,
            y=val_y,
            ctx=val_ctx_list,
            q_map_train=q_map_train,
            unigram_q_train=unigram_q_train,
            temp=temp,
            run_dir=run_dir,
            epoch=0,
            split_name="val",
        )
        val_acc_hist.append(v0.exact_acc)
        val_support_hist.append(v0.support_acc)
        val_ce_hist.append(v0.soft_ce)
        val_qmass_hist.append(v0.qmass_mean)
        val_unseen_hist.append(v0.unseen_contexts)
        save_history_arrays(
            run_dir,
            tr_acc_hist=[],
            tr_support_hist=tr_support_hist,
            tr_ce_hist=[],
            tr_qmass_hist=tr_qmass_hist,
            val_acc_hist=val_acc_hist,
            val_support_hist=val_support_hist,
            val_ce_hist=val_ce_hist,
            val_qmass_hist=val_qmass_hist,
            val_unseen_hist=val_unseen_hist,
            ep_total_s=[],
            ep_free_s=[],
            ep_clamp_s=[],
            ep_update_s=[],
            reload_free_hist=[],
            reload_clamp_hist=[],
            nonfinite_free_hist=[],
            nonfinite_clamp_hist=[],
            active_edges_hist=[],
            pruned_edges_hist=[],
            born_edges_hist=[],
            utility_thresh_hist=[],
            backbone_thresh_hist=[],
        )
        (run_dir / "0_diag_epoch0.json").write_text(json.dumps(diag0, indent=2))
        if should_run_interval(0, sample_every, epochs == 0):
            save_generation_samples(
                args=args,
                topo=topo,
                vg_unique=vg_unique,
                edge_active=edge_active,
                q_map_train=q_map_train,
                unigram_q_train=unigram_q_train,
                val_ctx=val_ctx,
                val_y=val_y,
                run_dir=run_dir,
                epoch=0,
            )
        print(
            f"[epoch 0] {cfg_str} | VAL exact_acc={v0.exact_acc:.4f} "
            f"support_acc={v0.support_acc:.4f} softCE={v0.soft_ce:.6f} "
            f"qmass_mean={diag0.get('val_qmass_mean', float('nan')):.4f} "
            f"unseen_ctx={int(v0.unseen_contexts)}",
            flush=True,
        )
        if should_run_interval(0, plot_every, epochs == 0):
            try:
                save_plots(run_dir)
            except Exception:
                pass
        try:
            log_f.flush()
            log_f.close()
        except Exception:
            pass
        return

    tr_acc_hist = load_hist_list(run_dir, "0_train_acc.npy", dtype=float)
    tr_support_hist = load_hist_list(run_dir, "0_train_support_acc.npy", dtype=float)
    tr_ce_hist = load_hist_list(run_dir, "0_train_ce.npy", dtype=float)
    tr_qmass_hist = load_hist_list(run_dir, "0_train_qmass.npy", dtype=float)
    val_acc_hist = load_hist_list(run_dir, "0_val_acc.npy", dtype=float)
    val_support_hist = load_hist_list(run_dir, "0_val_support_acc.npy", dtype=float)
    val_ce_hist = load_hist_list(run_dir, "0_val_ce.npy", dtype=float)
    val_qmass_hist = load_hist_list(run_dir, "0_val_qmass.npy", dtype=float)
    val_unseen_hist = load_hist_list(run_dir, "0_val_unseen_contexts.npy", dtype=float)
    ep_total_s = load_hist_list(run_dir, "0_epoch_total_s.npy", dtype=float)
    ep_free_s = load_hist_list(run_dir, "0_epoch_free_s.npy", dtype=float)
    ep_clamp_s = load_hist_list(run_dir, "0_epoch_clamp_s.npy", dtype=float)
    ep_update_s = load_hist_list(run_dir, "0_epoch_update_s.npy", dtype=float)
    reload_free_hist = load_hist_list(run_dir, "0_reload_free.npy", dtype=int)
    reload_clamp_hist = load_hist_list(run_dir, "0_reload_clamp.npy", dtype=int)
    nonfinite_free_hist = load_hist_list(run_dir, "0_nonfinite_free.npy", dtype=int)
    nonfinite_clamp_hist = load_hist_list(run_dir, "0_nonfinite_clamp.npy", dtype=int)
    active_edges_hist = load_hist_list(run_dir, "0_active_edges.npy", dtype=int)
    pruned_edges_hist = load_hist_list(run_dir, "0_pruned_edges.npy", dtype=int)
    born_edges_hist = load_hist_list(run_dir, "0_born_edges.npy", dtype=int)
    utility_thresh_hist = load_hist_list(run_dir, "0_utility_threshold.npy", dtype=float)
    backbone_thresh_hist = load_hist_list(run_dir, "0_backbone_threshold.npy", dtype=float)

    t_ep0 = time.time()
    order = np.arange(train_x.shape[0], dtype=int)
    np.random.shuffle(order)

    reload_free = 0
    reload_clamp = 0
    nonfinite_free = 0
    nonfinite_clamp = 0
    t_free = 0.0
    t_clamp = 0.0
    t_update = 0.0
    n_free = 0
    n_clamp = 0
    is_final_epoch = epoch == epochs
    do_eval = should_run_interval(epoch, eval_every, is_final_epoch)
    active_direct, active_in_hidden, active_hidden_out = view_mask_blocks(edge_active, topo)
    active_direct_f = np.asarray(active_direct, dtype=float)
    active_in_hidden_f = np.asarray(active_in_hidden, dtype=float)
    active_hidden_out_f = np.asarray(active_hidden_out, dtype=float)

    for idx in order:
        x = np.asarray(train_x[idx], dtype=float)
        ytrue = int(train_y[idx])
        q_target = one_hot(ytrue, topo.K)

        g_direct = EDGE_CONDUCTANCE_SCALE * np.asarray(direct_params, dtype=float) * active_direct_f
        g_in_hidden = EDGE_CONDUCTANCE_SCALE * np.asarray(in_hidden_params, dtype=float) * active_in_hidden_f
        g_hidden_out = EDGE_CONDUCTANCE_SCALE * np.asarray(hidden_out_params, dtype=float) * active_hidden_out_f
        den_hidden = g_in_hidden + g_hidden_out
        g_series = np.where(den_hidden > 1e-12, (g_in_hidden * g_hidden_out) / den_hidden, 0.0)
        g_total = g_direct + g_series
        sum_g = np.sum(g_total, axis=1)

        free0 = time.time()
        Vout = solve_outputs(gmat=g_total, xin=x, clamp_res=RS_FREE, clamp_target=None, sum_g=sum_g)
        t_free += float(time.time() - free0)
        n_free += 1

        if not np.all(np.isfinite(Vout)):
            nonfinite_free += 1
            continue

        Vhidden_free = compute_hidden_voltages(
            g_in_hidden=g_in_hidden,
            g_hidden_out=g_hidden_out,
            xin=x,
            vout=Vout,
        )
        clamp_target = clamp_targets_from_free(Vout, q_target=q_target, delta=delta, temp=temp)

        clamp0 = time.time()
        Vout_clamp = solve_outputs(gmat=g_total, xin=x, clamp_res=RS_CLAMP, clamp_target=clamp_target, sum_g=sum_g)
        t_clamp += float(time.time() - clamp0)
        n_clamp += 1

        if not np.all(np.isfinite(Vout_clamp)):
            nonfinite_clamp += 1
            continue

        Vhidden_clamp = compute_hidden_voltages(
            g_in_hidden=g_in_hidden,
            g_hidden_out=g_hidden_out,
            xin=x,
            vout=Vout_clamp,
        )
        upd0 = time.time()
        dV_direct_free = Vout[:, None] - x[None, :]
        dV_direct_clamp = Vout_clamp[:, None] - x[None, :]
        dV_in_hidden_free = Vhidden_free - x[None, :]
        dV_in_hidden_clamp = Vhidden_clamp - x[None, :]
        dV_hidden_out_free = Vout[:, None] - Vhidden_free
        dV_hidden_out_clamp = Vout_clamp[:, None] - Vhidden_clamp

        contrast_direct = np.abs(dV_direct_clamp**2 - dV_direct_free**2)
        contrast_in_hidden = np.abs(dV_in_hidden_clamp**2 - dV_in_hidden_free**2)
        contrast_hidden_out = np.abs(dV_hidden_out_clamp**2 - dV_hidden_out_free**2)
        backbone_direct = g_direct * (dV_direct_free**2)
        backbone_in_hidden = g_in_hidden * (dV_in_hidden_free**2)
        backbone_hidden_out = g_hidden_out * (dV_hidden_out_free**2)
        contrast = np.concatenate(
            [contrast_direct.reshape(-1), contrast_in_hidden.reshape(-1), contrast_hidden_out.reshape(-1)]
        )
        backbone = np.concatenate(
            [backbone_direct.reshape(-1), backbone_in_hidden.reshape(-1), backbone_hidden_out.reshape(-1)]
        )
        utility_ema = utility_beta * utility_ema + (1.0 - utility_beta) * contrast
        backbone_ema = utility_beta * backbone_ema + (1.0 - utility_beta) * backbone

        direct_params += -gamma * (dV_direct_clamp**2 - dV_direct_free**2) * active_direct_f
        in_hidden_params += -gamma * (dV_in_hidden_clamp**2 - dV_in_hidden_free**2) * active_in_hidden_f
        hidden_out_params += -gamma * (dV_hidden_out_clamp**2 - dV_hidden_out_free**2) * active_hidden_out_f

        np.clip(direct_params, VG_CLIP_LO, VG_CLIP_HI, out=direct_params)
        np.clip(in_hidden_params, VG_CLIP_LO, VG_CLIP_HI, out=in_hidden_params)
        np.clip(hidden_out_params, VG_CLIP_LO, VG_CLIP_HI, out=hidden_out_params)
        t_update += float(time.time() - upd0)

    edge_age[edge_active] += 1
    if remodel_every_epochs > 0 and (epoch % remodel_every_epochs == 0):
        remodel = remodel_hidden_edges(
            topo=topo,
            vg_unique=vg_unique,
            edge_active=edge_active,
            utility_ema=utility_ema,
            backbone_ema=backbone_ema,
            edge_age=edge_age,
            prune_utility_quantile=prune_utility_quantile,
            prune_backbone_quantile=prune_backbone_quantile,
            prune_rand_prob=prune_rand_prob,
            max_prune_frac=max_prune_frac,
            min_edge_age=min_edge_age,
            birth_vg_lo=birth_vg_lo,
            birth_vg_hi=birth_vg_hi,
            rng=rng,
        )
    else:
        active_mask = np.asarray(edge_active, dtype=bool)
        remodel = RemodelOut(
            active_edges=int(np.sum(active_mask)),
            pruned_edges=0,
            born_edges=0,
            utility_threshold=float("nan"),
            backbone_threshold=float("nan"),
            active_edge_fraction=float(np.mean(active_mask.astype(float))),
            utility_active_mean=float(np.mean(utility_ema[active_mask])) if np.any(active_mask) else 0.0,
            backbone_active_mean=float(np.mean(backbone_ema[active_mask])) if np.any(active_mask) else 0.0,
        )

    if do_eval:
        tr_eval, tr_diag = eval_free_metrics(
            topo=topo,
            vg_unique=vg_unique,
            edge_active=edge_active,
            X=train_x,
            y=train_y,
            ctx=train_ctx,
            q_map_train=q_map_train,
            unigram_q_train=unigram_q_train,
            temp=temp,
            split_name="train",
        )
        v_eval, diag = eval_free_metrics(
            topo=topo,
            vg_unique=vg_unique,
            edge_active=edge_active,
            X=val_x,
            y=val_y,
            ctx=val_ctx_list,
            q_map_train=q_map_train,
            unigram_q_train=unigram_q_train,
            temp=temp,
            run_dir=run_dir,
            epoch=epoch,
            split_name="val",
        )
    else:
        tr_eval = make_nan_evalout()
        tr_diag = make_nan_diag("train")
        v_eval = make_nan_evalout()
        diag = make_nan_diag("val")

    tr_acc_hist.append(float(tr_eval.exact_acc))
    tr_support_hist.append(float(tr_eval.support_acc))
    tr_ce_hist.append(float(tr_eval.soft_ce))
    tr_qmass_hist.append(float(tr_eval.qmass_mean))
    reload_free_hist.append(int(reload_free))
    reload_clamp_hist.append(int(reload_clamp))
    nonfinite_free_hist.append(int(nonfinite_free))
    nonfinite_clamp_hist.append(int(nonfinite_clamp))
    val_acc_hist.append(float(v_eval.exact_acc))
    val_support_hist.append(float(v_eval.support_acc))
    val_ce_hist.append(float(v_eval.soft_ce))
    val_qmass_hist.append(float(v_eval.qmass_mean))
    val_unseen_hist.append(float(v_eval.unseen_contexts))
    active_edges_hist.append(int(remodel.active_edges))
    pruned_edges_hist.append(int(remodel.pruned_edges))
    born_edges_hist.append(int(remodel.born_edges))
    utility_thresh_hist.append(float(remodel.utility_threshold))
    backbone_thresh_hist.append(float(remodel.backbone_threshold))

    if do_eval and should_run_interval(epoch, sample_every, is_final_epoch):
        save_generation_samples(
            args=args,
            topo=topo,
            vg_unique=vg_unique,
            edge_active=edge_active,
            q_map_train=q_map_train,
            unigram_q_train=unigram_q_train,
            val_ctx=val_ctx,
            val_y=val_y,
            run_dir=run_dir,
            epoch=epoch,
        )

    ep_total = float(time.time() - t_ep0)
    ep_total_s.append(ep_total)
    ep_free_s.append(float(t_free))
    ep_clamp_s.append(float(t_clamp))
    ep_update_s.append(float(t_update))

    save_history_arrays(
        run_dir,
        tr_acc_hist=tr_acc_hist,
        tr_support_hist=tr_support_hist,
        tr_ce_hist=tr_ce_hist,
        tr_qmass_hist=tr_qmass_hist,
        val_acc_hist=val_acc_hist,
        val_support_hist=val_support_hist,
        val_ce_hist=val_ce_hist,
        val_qmass_hist=val_qmass_hist,
        val_unseen_hist=val_unseen_hist,
        ep_total_s=ep_total_s,
        ep_free_s=ep_free_s,
        ep_clamp_s=ep_clamp_s,
        ep_update_s=ep_update_s,
        reload_free_hist=reload_free_hist,
        reload_clamp_hist=reload_clamp_hist,
        nonfinite_free_hist=nonfinite_free_hist,
        nonfinite_clamp_hist=nonfinite_clamp_hist,
        active_edges_hist=active_edges_hist,
        pruned_edges_hist=pruned_edges_hist,
        born_edges_hist=born_edges_hist,
        utility_thresh_hist=utility_thresh_hist,
        backbone_thresh_hist=backbone_thresh_hist,
    )
    np.save(run_dir / f"0_vg_unique_epoch{epoch}.npy", vg_unique.copy())
    np.save(run_dir / f"0_edge_active_epoch{epoch}.npy", np.asarray(edge_active, dtype=bool))
    np.save(run_dir / f"0_edge_utility_epoch{epoch}.npy", np.asarray(utility_ema, dtype=float))
    np.save(run_dir / f"0_edge_backbone_epoch{epoch}.npy", np.asarray(backbone_ema, dtype=float))
    np.save(run_dir / f"0_edge_age_epoch{epoch}.npy", np.asarray(edge_age, dtype=int))

    if do_eval:
        vg_stats = compute_vg_saturation_stats(vg_unique, edge_active=edge_active)
        summary = {
            "epoch": int(epoch),
            "config": {
                "seed": seed,
                "gamma": gamma,
                "delta": delta,
                "softmax_temp": temp,
                "rails": [vminus_val, vplus_val],
                "solver": solver,
                "body_tie": body_tie,
                "rs_clamp": RS_CLAMP,
                "vg_init": {
                    "mode": vg_init_mode,
                    "lo": vg_init_lo,
                    "hi": vg_init_hi,
                    "fixed": vg_init_single,
                },
                "epochs": epochs,
                "device_include_path": device_lib,
                "device_subckt": DEVICE_SUBCKT,
                "loss": "onehot_cross_entropy",
                "template_mode": template_mode,
                "epoch_process_mode": process_mode,
                "backend": "linear_resistor_hidden_path_kcl",
                "conductance_scale": EDGE_CONDUCTANCE_SCALE,
                "parameter_blocks": PATH_PARAM_BLOCKS,
                "eval_every": eval_every,
                "sample_every": sample_every,
                "plot_every": plot_every,
                "remodel_every_epochs": remodel_every_epochs,
                "utility_beta": utility_beta,
                "prune_utility_quantile": prune_utility_quantile,
                "prune_backbone_quantile": prune_backbone_quantile,
                "prune_rand_prob": prune_rand_prob,
                "max_prune_frac": max_prune_frac,
                "min_edge_age": min_edge_age,
                "birth_vg_lo": birth_vg_lo,
                "birth_vg_hi": birth_vg_hi,
            },
            "train": {
                "exact_acc": float(tr_eval.exact_acc),
                "support_acc": float(tr_eval.support_acc),
                "soft_ce": float(tr_eval.soft_ce),
                "qmass_mean": float(tr_eval.qmass_mean),
                "unseen_contexts": float(tr_eval.unseen_contexts),
                "n_free": int(n_free),
                "n_clamp": int(n_clamp),
                "reload_free": int(reload_free),
                "reload_clamp": int(reload_clamp),
                "nonfinite_free": int(nonfinite_free),
                "nonfinite_clamp": int(nonfinite_clamp),
            },
            "val": {
                "exact_acc": float(v_eval.exact_acc),
                "support_acc": float(v_eval.support_acc),
                "soft_ce": float(v_eval.soft_ce),
                **{k: float(v) for k, v in diag.items()},
            },
            "timing_s": {
                "epoch_total": float(ep_total),
                "train_free": float(t_free),
                "train_clamp": float(t_clamp),
                "train_update": float(t_update),
            },
            "remodel": {
                "active_edges": int(remodel.active_edges),
                "pruned_edges": int(remodel.pruned_edges),
                "born_edges": int(remodel.born_edges),
                "utility_threshold": float(remodel.utility_threshold),
                "backbone_threshold": float(remodel.backbone_threshold),
                "active_edge_fraction": float(remodel.active_edge_fraction),
                "utility_active_mean": float(remodel.utility_active_mean),
                "backbone_active_mean": float(remodel.backbone_active_mean),
            },
            "vg_stats": vg_stats,
        }
        (run_dir / f"0_epoch_summary_epoch{epoch}.json").write_text(json.dumps(summary, indent=2))
        (run_dir / f"0_diag_epoch{epoch}.json").write_text(json.dumps(diag, indent=2))

        print(
            f"[epoch {epoch}/{epochs}] {cfg_str} | TRAIN exact_acc={tr_eval.exact_acc:.4f} "
            f"support_acc={tr_eval.support_acc:.4f} softCE={tr_eval.soft_ce:.6f} "
            f"qmass_mean={tr_eval.qmass_mean:.4f} free={n_free} clamp={n_clamp} "
            f"reloadF={reload_free} reloadC={reload_clamp} nonfiniteF={nonfinite_free} nonfiniteC={nonfinite_clamp} "
            f"active_edges={remodel.active_edges} pruned={remodel.pruned_edges} born={remodel.born_edges}",
            flush=True,
        )
        print(
            f"[epoch {epoch}/{epochs}] {cfg_str} | VAL exact_acc={v_eval.exact_acc:.4f} "
            f"support_acc={v_eval.support_acc:.4f} softCE={v_eval.soft_ce:.6f} "
            f"qmass_mean={diag.get('val_qmass_mean', float('nan')):.4f} unseen_ctx={int(v_eval.unseen_contexts)} | "
            f"timing total={ep_total:.2f}s free={t_free:.2f}s clamp={t_clamp:.2f}s upd={t_update:.2f}s",
            flush=True,
        )
    else:
        print(
            f"[epoch {epoch}/{epochs}] {cfg_str} | step_only free={n_free} clamp={n_clamp} "
            f"active_edges={remodel.active_edges} pruned={remodel.pruned_edges} born={remodel.born_edges} "
            f"timing total={ep_total:.2f}s free={t_free:.2f}s clamp={t_clamp:.2f}s upd={t_update:.2f}s",
            flush=True,
        )

    if do_eval and should_run_interval(epoch, plot_every, is_final_epoch):
        try:
            save_plots(run_dir)
        except Exception:
            pass

    try:
        log_f.flush()
        log_f.close()
    except Exception:
        pass


def build_linear_manifest_text(
    *,
    topo: DenseIOTopology,
    vminus_val: float,
    vplus_val: float,
    solver: str,
    body_tie: str,
    device_lib: str,
) -> str:
    return (
        ".title clln_dense_language32_embed4_onehotxent_linear_resistor_hidden_path\n"
        "* Linear resistor hidden-path backend; this file is a manifest, not an executable SPICE netlist.\n"
        f"* topology_kind={topo.meta.get('kind')}\n"
        f"* Nin={topo.Nin} K={topo.K} direct_edges={topo.num_edges} hidden_path_edges={2 * topo.num_edges}\n"
        f"* rails=({vminus_val:.6f}, {vplus_val:.6f})\n"
        f"* solver_arg={solver} body_tie_arg={body_tie}\n"
        f"* device_lib_arg={device_lib}\n"
        f"* edge_conductance_scale={EDGE_CONDUCTANCE_SCALE:.8g}\n"
        f"* rs_free={RS_FREE:.8g} rs_clamp={RS_CLAMP:.8g}\n"
        ".end\n"
    )


def run_controller(args: argparse.Namespace) -> None:
    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)

    epochs = int(args.epochs)
    gamma = float(args.gamma)
    delta = float(args.delta)
    temp = float(args.softmax_temp)
    if temp <= 0.0:
        raise ValueError("--softmax-temp must be > 0")

    vminus_val = float(args.vminus)
    vplus_val = float(args.vplus)
    solver = str(args.solver).lower()
    body_tie = str(args.body_tie)
    vg_init_mode = str(args.vg_init)
    vg_init_lo = float(args.vg_init_lo)
    vg_init_hi = float(args.vg_init_hi)
    vg_init_single = float(args.vg_init_fixed)
    device_lib = str(args.device_lib)
    process_mode = str(args.process_mode)
    eval_every = int(args.eval_every)
    sample_every = int(args.sample_every)
    plot_every = int(args.plot_every)
    remodel_every_epochs = int(args.remodel_every_epochs)
    utility_beta = float(args.utility_beta)
    prune_utility_quantile = float(args.prune_utility_quantile)
    prune_backbone_quantile = float(args.prune_backbone_quantile)
    prune_rand_prob = float(args.prune_rand_prob)
    max_prune_frac = float(args.max_prune_frac)
    min_edge_age = int(args.min_edge_age)
    birth_vg_lo = float(args.birth_vg_lo)
    birth_vg_hi = float(args.birth_vg_hi)

    template_mode = str(args.template_mode)
    num_sentences = int(args.num_sentences)
    min_target_count = int(args.min_target_count)
    if num_sentences < 1:
        raise ValueError("--num-sentences must be >= 1")
    if int(args.max_sentence_words) < 4:
        raise ValueError("--max-sentence-words must be >= 4")

    sentences, total_target_coverage = build_sentence_corpus(
        num_sentences=num_sentences,
        template_mode=template_mode,
        max_sentence_words=int(args.max_sentence_words),
        min_target_count=min_target_count,
    )

    train_sentences, val_sentences = train_test_split(
        sentences,
        test_size=float(args.val_frac),
        random_state=seed,
        shuffle=True,
    )

    X_train, y_train, ctx_train, train_target_counter = build_windows_from_sentences(train_sentences)
    X_val, y_val, ctx_val, val_target_counter = build_windows_from_sentences(val_sentences)

    if args.max_train and args.max_train > 0:
        X_train = X_train[: args.max_train]
        y_train = y_train[: args.max_train]
        ctx_train = ctx_train[: args.max_train]
    if args.max_val and args.max_val > 0:
        X_val = X_val[: args.max_val]
        y_val = y_val[: args.max_val]
        ctx_val = ctx_val[: args.max_val]

    train_x = [np.asarray(v, dtype=float) for v in X_train]
    train_y = [int(v) for v in y_train]
    train_ctx = [tuple(c) for c in ctx_train]
    val_x = [np.asarray(v, dtype=float) for v in X_val]
    val_y = [int(v) for v in y_val]
    val_ctx = [tuple(c) for c in ctx_val]
    q_map_train = build_context_target_distributions(train_ctx, train_y, OUTPUT_DIM)
    unigram_q_train = build_unigram_target_distribution(train_y, OUTPUT_DIM)

    topo = make_dense_io_topology()
    if topo.Nin != INPUT_DIM or topo.K != OUTPUT_DIM:
        raise RuntimeError("Internal topology dimensions do not match language task")

    if vg_init_mode == "fixed":
        vg_unique = np.full((PATH_PARAM_BLOCKS * topo.num_edges,), vg_init_single, dtype=float)
    else:
        if vg_init_hi <= vg_init_lo:
            raise ValueError("--vg-init-hi must be > --vg-init-lo for random init")
        vg_unique = np.random.uniform(vg_init_lo, vg_init_hi, size=(PATH_PARAM_BLOCKS * topo.num_edges,)).astype(float)
    np.clip(vg_unique, VG_CLIP_LO, VG_CLIP_HI, out=vg_unique)

    results_dir = Path(__file__).resolve().parent / "results_language_32_embed4_linear_resistor_hiddenpath_onehotce"
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
    cfg_str = build_cfg_str(
        seed=seed,
        gamma=gamma,
        delta=delta,
        temp=temp,
        topo=topo,
        num_sentences_actual=len(sentences),
        min_target_count=min_target_count,
        max_sentence_words=int(args.max_sentence_words),
        template_mode=template_mode,
        vminus_val=vminus_val,
        vplus_val=vplus_val,
        solver=solver,
        body_tie=body_tie,
        vg_init_mode=vg_init_mode,
        epochs=epochs,
        device_lib=device_lib,
        process_mode=process_mode,
        eval_every=eval_every,
        remodel_every_epochs=remodel_every_epochs,
    )

    print("=== RUN START (clln_dense_language32_embed4_onehotxent_linear_resistor_hidden_path) ===", flush=True)
    print(cfg_str, flush=True)
    print(
        f"train_sentences={len(train_sentences)} val_sentences={len(val_sentences)} "
        f"train_windows={len(train_x)} val_windows={len(val_x)} direct_edges={topo.num_edges} total_params={PATH_PARAM_BLOCKS * topo.num_edges}",
        flush=True,
    )
    print(f"epoch execution mode={process_mode}", flush=True)

    (run_dir / "input_vocab.json").write_text(json.dumps({"input_vocab": INPUT_VOCAB}, indent=2))
    (run_dir / "output_vocab.json").write_text(json.dumps({"output_vocab": OUTPUT_VOCAB}, indent=2))
    (run_dir / "token_embed_4d.json").write_text(json.dumps(TOKEN_EMBED_4D, indent=2))
    (run_dir / "sample_sentences.txt").write_text("\n".join(" ".join(s) for s in sentences[:300]))
    (run_dir / "target_coverage_total.json").write_text(json.dumps(total_target_coverage, indent=2))
    (run_dir / "target_coverage_train.json").write_text(json.dumps({k: int(train_target_counter.get(k, 0)) for k in OUTPUT_VOCAB}, indent=2))
    (run_dir / "target_coverage_val.json").write_text(json.dumps({k: int(val_target_counter.get(k, 0)) for k in OUTPUT_VOCAB}, indent=2))

    preview_items = []
    for ctx, y in list(zip(train_ctx, train_y))[:120]:
        words = [INPUT_ID_TO_WORD[t] for t in ctx]
        q_target = q_map_train.get(tuple(ctx), unigram_q_train)
        preview_items.append({
            "context": words,
            "target_word": OUTPUT_ID_TO_WORD[int(y)],
            "valid_next_token_distribution": {
                OUTPUT_ID_TO_WORD[k]: float(q_target[k])
                for k in range(OUTPUT_DIM)
                if float(q_target[k]) > 0.0
            },
            "valid_next_token_distribution_top": top_words_from_q(q_target, topk=6),
        })
    (run_dir / "sample_target_preview.json").write_text(json.dumps(preview_items, indent=2))

    manifest = build_linear_manifest_text(
        topo=topo,
        vminus_val=vminus_val,
        vplus_val=vplus_val,
        solver=solver,
        body_tie=body_tie,
        device_lib=device_lib,
    )
    (run_dir / "netlist_initial.cir").write_text(manifest)

    meta = {
        "script": str(Path(__file__).resolve()),
        "argv": list(os.sys.argv),
        "seed": seed,
        "timestamp": datetime.now().isoformat(),
        "dataset": {
            "name": "synthetic_language_32token_embed4",
            "num_sentences_requested": num_sentences,
            "num_sentences_actual": len(sentences),
            "min_target_count": min_target_count,
            "max_sentence_words": int(args.max_sentence_words),
            "template_mode": template_mode,
            "context_len": CONTEXT_LEN,
            "token_embed_dim": TOKEN_EMBED_DIM,
            "input_vocab": INPUT_VOCAB,
            "output_vocab": OUTPUT_VOCAB,
            "terminal_punctuation": sorted(list(TERMINAL_PUNCT)),
        },
        "train_sentence_count": len(train_sentences),
        "val_sentence_count": len(val_sentences),
        "train_count": len(train_x),
        "val_count": len(val_x),
        "gamma": gamma,
        "delta": delta,
        "softmax_temp": temp,
        "epochs": epochs,
        "rails": {"vminus": vminus_val, "vplus": vplus_val},
        "solver": solver,
        "body_tie": body_tie,
        "rs_clamp": RS_CLAMP,
        "vg_init": {
            "mode": vg_init_mode,
            "lo": vg_init_lo,
            "hi": vg_init_hi,
            "fixed": vg_init_single,
        },
        "device": {
            "include_path": device_lib,
            "subckt": DEVICE_SUBCKT,
        },
        "topology": topo.meta,
        "loss": "onehot_cross_entropy",
        "generation": {
            "sample_prompts": int(args.sample_prompts),
            "sample_max_len": int(args.sample_max_len),
        },
        "execution": {
            "epoch_process_mode": process_mode,
            "worker_python": sys.executable,
            "eval_every": eval_every,
            "sample_every": sample_every,
            "plot_every": plot_every,
            "worker_dataset_files": [
                "train_x.npy",
                "train_y.npy",
                "train_ctx.npy",
                "val_x.npy",
                "val_y.npy",
                "val_ctx.npy",
            ],
        },
        "backend": {
            "name": "linear_resistor_hidden_path_kcl",
            "conductance_scale": EDGE_CONDUCTANCE_SCALE,
            "stored_parameter_name": "vg_unique",
            "stored_parameter_note": "vg_unique stores clipped linear-resistor parameters in direct/input-hidden/hidden-output blocks",
            "parameter_blocks": PATH_PARAM_BLOCKS,
            "remodel": {
                "every_epochs": remodel_every_epochs,
                "utility_beta": utility_beta,
                "prune_utility_quantile": prune_utility_quantile,
                "prune_backbone_quantile": prune_backbone_quantile,
                "prune_rand_prob": prune_rand_prob,
                "max_prune_frac": max_prune_frac,
                "min_edge_age": min_edge_age,
                "birth_vg_lo": birth_vg_lo,
                "birth_vg_hi": birth_vg_hi,
                "birth_count_policy": "equal_to_pruned_count",
            },
        },
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

    np.save(run_dir / "train_x.npy", np.asarray(train_x, dtype=float))
    np.save(run_dir / "train_y.npy", np.asarray(train_y, dtype=int))
    np.save(run_dir / "train_ctx.npy", np.asarray(train_ctx, dtype=int))
    np.save(run_dir / "val_x.npy", np.asarray(val_x, dtype=float))
    np.save(run_dir / "val_y.npy", np.asarray(val_y, dtype=int))
    np.save(run_dir / "val_ctx.npy", np.asarray(val_ctx, dtype=int))
    np.save(run_dir / "0_vg_unique_epoch0.npy", vg_unique.copy())
    np.save(run_dir / "0_edge_active_epoch0.npy", np.ones((PATH_PARAM_BLOCKS * topo.num_edges,), dtype=bool))
    np.save(run_dir / "0_edge_utility_epoch0.npy", np.zeros((PATH_PARAM_BLOCKS * topo.num_edges,), dtype=float))
    np.save(run_dir / "0_edge_backbone_epoch0.npy", np.zeros((PATH_PARAM_BLOCKS * topo.num_edges,), dtype=float))
    np.save(run_dir / "0_edge_age_epoch0.npy", np.zeros((PATH_PARAM_BLOCKS * topo.num_edges,), dtype=int))

    try:
        net_nodes = [topo.negref, topo.posref] + topo.out_nodes.tolist() + topo.input_nodes.tolist()
        G = nx.DiGraph()
        G.add_nodes_from(sorted(set(net_nodes)))
        for d, s in zip(topo.edges_D.tolist(), topo.edges_S.tolist()):
            G.add_edge(d, s)
        for kout, out_node in enumerate(topo.out_nodes.tolist()):
            for iin, in_node in enumerate(topo.input_nodes.tolist()):
                hid = f"h_{iin}_{kout}"
                G.add_node(hid)
                G.add_edge(in_node, hid)
                G.add_edge(hid, out_node)
        nx.write_graphml(G, str(run_dir / "0.graphml"))
    except Exception:
        pass

    env = os.environ.copy()
    env["RUN_DIR"] = str(run_dir)
    for epoch in range(0, epochs + 1):
        print(f"[controller] launching worker epoch={epoch}", flush=True)
        if process_mode == "in_process":
            worker_args = argparse.Namespace(**vars(args))
            worker_args.worker_run_dir = str(run_dir)
            worker_args.worker_epoch = int(epoch)
            run_worker_epoch(worker_args)
        else:
            subprocess.run(build_worker_cmd(args, run_dir, epoch), check=True, env=env)

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

    if args.final_val:
        val_acc_hist = load_hist_list(run_dir, "0_val_acc.npy", dtype=float)
        print("FINAL val exact acc=", val_acc_hist[-1] if val_acc_hist else float("nan"), flush=True)

    print("=== RUN END (clln_dense_language32_embed4_onehotxent_linear_resistor_hidden_path) ===", flush=True)
    try:
        log_f.flush()
        log_f.close()
    except Exception:
        pass


def main():
    args = parse_args()
    if args.worker_run_dir:
        run_worker_epoch(args)
        return
    run_controller(args)


if __name__ == "__main__":
    main()
