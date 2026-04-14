#!/usr/bin/env python3
"""
CLLN language trainer — dense 24-bit input -> 15-word output NMOS network
One-hot cross-entropy training on a small quantum-themed prompt-completion task.

Design changes relative to the legacy 16-vocab one-hot trainer:
  - <BOS> is input padding only and is never predicted.
  - Output space contains 15 real next-token options: <EOS> plus 14 words.
  - The grammar is quantum-themed and includes semantic compatibility rules so
    earlier context positions matter more than in the old toy grammar.
  - Validation logs grammar-support metrics and can optionally apply extra
    clamp pressure against grammar-forbidden choices.
  - The model can warm-start from the best legacy 16-output checkpoint by
    dropping the BOS row and remapping old grammatical roles onto the new
    quantum vocabulary.

Recommended first use:
  conda run -n p311env python lang_model/vocab16/clln_lang_trainer_onehot_ce_quantum.py 0 --epochs 20
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import networkx as nx
import numpy as np
from PySpice.Spice.NgSpice.Shared import NgSpiceShared
from sklearn.model_selection import train_test_split

from clln_lang_trainer_onehot_ce import (
    DEVICE_SUBCKT,
    RS_CLAMP,
    RS_FREE,
    VG_CLIP_HI,
    VG_CLIP_LO,
    VG_INIT_SINGLE,
    DenseIOTopology,
    alter_inputs_named,
    compute_vg_saturation_stats,
    cross_entropy_from_outputs_soft,
    exec_chunked,
    find_repo_root,
    mk_free_all,
    mk_netlist,
    pred_label,
    restore_gate_voltages,
    run_and_read,
    save_plots,
    setup_logging,
    softmax_logits,
)


REPO_ROOT = find_repo_root()
DEFAULT_DEVICE_LIB_PATH = str(REPO_ROOT / "device_model" / "nmos_lvl1_ald1106.lib")
DEFAULT_RESULTS_DIR = Path(__file__).resolve().parent / "results_language_16_onehotce_quantum"
DEFAULT_LEGACY_WARMSTART_RUN = (
    Path(__file__).resolve().parent
    / "results_language_16_onehotce"
    / "sweeps"
    / "top60_from_softce_20260325-122210"
    / "g0.05_d0.1_t0.01"
)
DEFAULT_LEGACY_WARMSTART_EPOCH = 17

CONTEXT_LEN = 6
TOKEN_BITS = 4
INPUT_DIM = CONTEXT_LEN * TOKEN_BITS  # 24

# BOS is input-only padding token with the all-zero code.
VOCAB: List[str] = [
    "<BOS>",
    "<EOS>",
    "the",
    "a",
    "qubit",
    "photon",
    "electron",
    "detector",
    "field",
    "state",
    "measures",
    "couples",
    "drives",
    "is",
    "coherent",
    "with",
]
VOCAB_SIZE = len(VOCAB)
assert VOCAB_SIZE == 16
WORD_TO_ID: Dict[str, int] = {w: i for i, w in enumerate(VOCAB)}
ID_TO_WORD: Dict[int, str] = {i: w for i, w in enumerate(VOCAB)}

OUTPUT_TOKENS: List[str] = VOCAB[1:]
OUTPUT_DIM = len(OUTPUT_TOKENS)  # 15
OUTPUT_WORD_TO_ID: Dict[str, int] = {w: i for i, w in enumerate(OUTPUT_TOKENS)}
OUTPUT_ID_TO_WORD: Dict[int, str] = {i: w for i, w in enumerate(OUTPUT_TOKENS)}
TOKEN_ID_TO_OUTPUT_ID: Dict[int, int] = {WORD_TO_ID[w]: OUTPUT_WORD_TO_ID[w] for w in OUTPUT_TOKENS}
OUTPUT_ID_TO_TOKEN_ID: Dict[int, int] = {v: k for k, v in TOKEN_ID_TO_OUTPUT_ID.items()}

DETERMINERS = ["the", "a"]
ALL_NOUNS = ["qubit", "photon", "electron", "detector", "field", "state"]

GRAMMAR_MEASURE_OBJECTS = ["qubit", "photon", "electron", "field", "state"]
LOGIC_MEASURE_OBJECTS = ["qubit", "photon", "electron", "state"]
GRAMMAR_COHERENT_SUBJECTS = ["qubit", "photon", "electron", "field", "state"]
LOGIC_COHERENT_SUBJECTS = ["qubit", "photon", "field", "state"]

GRAMMAR_DRIVE_MAP: Dict[str, List[str]] = {
    "qubit": ["state", "field"],
    "photon": ["qubit", "electron", "state"],
    "electron": ["field", "state"],
    "detector": ["state"],
    "field": ["qubit", "electron", "state"],
    "state": ["field", "qubit"],
}
LOGIC_DRIVE_MAP: Dict[str, List[str]] = {
    "photon": ["qubit", "electron"],
    "field": ["qubit", "electron", "state"],
    "detector": ["state"],
    "state": ["qubit"],
}

GRAMMAR_COUPLE_MAP: Dict[str, List[str]] = {
    "qubit": ["photon", "electron", "field", "state"],
    "photon": ["qubit", "electron", "field", "state"],
    "electron": ["qubit", "photon", "field", "state"],
    "detector": ["field", "state"],
    "field": ["qubit", "photon", "electron", "state"],
    "state": ["qubit", "photon", "electron", "field"],
}
LOGIC_COUPLE_MAP: Dict[str, List[str]] = {
    "qubit": ["photon", "field", "electron"],
    "photon": ["qubit", "field", "state"],
    "electron": ["field", "photon", "state"],
    "field": ["qubit", "photon", "electron"],
    "state": ["photon", "electron"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CLLN dense 24->15 quantum language trainer, one-hot cross entropy, ngspice"
    )
    p.add_argument("seed", type=int, nargs="?", default=0)
    p.add_argument("--epochs", type=int, default=20)

    p.add_argument("--gamma", type=float, default=0.30)
    p.add_argument("--delta", type=float, default=0.30)
    p.add_argument("--softmax-temp", type=float, default=1.0)
    p.add_argument("--forbidden-boost", type=float, default=0.50)

    p.add_argument("--bit-v0", type=float, default=0.0)
    p.add_argument("--bit-v1", type=float, default=1.0)
    p.add_argument("--vminus", type=float, default=0.0)
    p.add_argument("--vplus", type=float, default=0.45)

    p.add_argument("--num-sentences", type=int, default=1000)
    p.add_argument("--max-sentence-words", type=int, default=7)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--max-train", type=int, default=0)
    p.add_argument("--max-val", type=int, default=0)
    p.add_argument(
        "--template-mode",
        type=str,
        choices=["grammar", "logic", "curriculum"],
        default="curriculum",
        help="grammar = teach syntax first, logic = enforce semantic compatibility, curriculum = mix both",
    )

    p.add_argument("--device-lib", type=str, default=DEFAULT_DEVICE_LIB_PATH)
    p.add_argument("--body-tie", type=str, choices=["source", "ground", "floating"], default="ground")
    p.add_argument("--solver", type=str, choices=["klu", "sparse"], default="klu")

    p.add_argument("--vg-init", type=str, choices=["random", "fixed"], default="random")
    p.add_argument("--vg-init-lo", type=float, default=1.0)
    p.add_argument("--vg-init-hi", type=float, default=3.0)
    p.add_argument("--vg-init-fixed", type=float, default=VG_INIT_SINGLE)

    p.add_argument("--warm-start-run-dir", type=Path, default=None)
    p.add_argument("--warm-start-epoch", type=int, default=None)
    p.add_argument("--warm-start-best-legacy", action="store_true")

    p.add_argument("--sample-prompts", type=int, default=6)
    p.add_argument("--sample-max-len", type=int, default=10)
    p.add_argument("--final-val", action="store_true")
    return p.parse_args()


def token_id_to_bits(tok_id: int) -> np.ndarray:
    return np.array([(tok_id >> shift) & 1 for shift in range(TOKEN_BITS - 1, -1, -1)], dtype=float)


def encode_context_tokens(ctx_ids: Sequence[int], bit_v0: float, bit_v1: float) -> np.ndarray:
    if len(ctx_ids) != CONTEXT_LEN:
        raise ValueError(f"Expected context length {CONTEXT_LEN}, got {len(ctx_ids)}")
    bits: List[float] = []
    for tid in ctx_ids:
        volts = np.where(token_id_to_bits(int(tid)) > 0.5, bit_v1, bit_v0)
        bits.extend(volts.tolist())
    return np.asarray(bits, dtype=float)


def decode_context_bits_to_words(x: np.ndarray, bit_v0: float, bit_v1: float) -> List[str]:
    x = np.asarray(x, dtype=float)
    out: List[str] = []
    thresh = 0.5 * (bit_v0 + bit_v1)
    for i in range(CONTEXT_LEN):
        seg = x[i * TOKEN_BITS:(i + 1) * TOKEN_BITS]
        bits = [(1 if v > thresh else 0) for v in seg]
        tid = 0
        for b in bits:
            tid = (tid << 1) | b
        out.append(ID_TO_WORD.get(tid, f"<UNK:{tid}>"))
    return out


def make_dense_io_topology() -> DenseIOTopology:
    input_nodes = np.arange(1, INPUT_DIM + 1, dtype=int)
    out_nodes = np.arange(INPUT_DIM + 1, INPUT_DIM + OUTPUT_DIM + 1, dtype=int)
    negref = int(INPUT_DIM + OUTPUT_DIM + 1)
    posref = int(INPUT_DIM + OUTPUT_DIM + 2)

    edges_D: List[int] = []
    edges_S: List[int] = []
    for out_n in out_nodes.tolist():
        for in_n in input_nodes.tolist():
            edges_D.append(int(out_n))
            edges_S.append(int(in_n))

    meta = {
        "kind": "dense_input_output_bipartite",
        "context_len": CONTEXT_LEN,
        "token_bits": TOKEN_BITS,
        "input_vocab_size": VOCAB_SIZE,
        "output_vocab_size": OUTPUT_DIM,
        "num_edges": len(edges_D),
    }
    return DenseIOTopology(
        Nin=INPUT_DIM,
        K=OUTPUT_DIM,
        input_nodes=input_nodes,
        out_nodes=out_nodes,
        negref=negref,
        posref=posref,
        edges_D=np.asarray(edges_D, dtype=int),
        edges_S=np.asarray(edges_S, dtype=int),
        meta=meta,
    )


def det_nps(nouns: Sequence[str]) -> List[List[str]]:
    return [[det, noun] for det in DETERMINERS for noun in nouns]


def enumerate_grammar_sentences() -> List[List[str]]:
    out: List[List[str]] = []
    noun_phrases = det_nps(ALL_NOUNS)

    for subj in noun_phrases:
        subj_noun = subj[-1]

        if subj_noun in GRAMMAR_COHERENT_SUBJECTS:
            out.append(subj + ["is", "coherent"])

        for obj in det_nps(GRAMMAR_MEASURE_OBJECTS):
            out.append(subj + ["measures"] + obj)

        for obj_noun in GRAMMAR_DRIVE_MAP.get(subj_noun, []):
            for det in DETERMINERS:
                out.append(subj + ["drives", det, obj_noun])

        for obj_noun in GRAMMAR_COUPLE_MAP.get(subj_noun, []):
            for det in DETERMINERS:
                out.append(subj + ["couples", "with", det, obj_noun])

    return out


def enumerate_logic_sentences() -> List[List[str]]:
    out: List[List[str]] = []

    for obj in det_nps(LOGIC_MEASURE_OBJECTS):
        out.append(["the", "detector", "measures"] + obj)
        out.append(["a", "detector", "measures"] + obj)

    for subj in det_nps(LOGIC_COHERENT_SUBJECTS):
        out.append(subj + ["is", "coherent"])

    for subj_noun, obj_nouns in LOGIC_DRIVE_MAP.items():
        for det_s in DETERMINERS:
            for det_o in DETERMINERS:
                for obj_noun in obj_nouns:
                    out.append([det_s, subj_noun, "drives", det_o, obj_noun])

    for subj_noun, obj_nouns in LOGIC_COUPLE_MAP.items():
        for det_s in DETERMINERS:
            for det_o in DETERMINERS:
                for obj_noun in obj_nouns:
                    if obj_noun == subj_noun:
                        continue
                    out.append([det_s, subj_noun, "couples", "with", det_o, obj_noun])

    return out


def enumerate_allowed_sentences(template_mode: str) -> List[List[str]]:
    grammar = enumerate_grammar_sentences()
    logic = enumerate_logic_sentences()

    if template_mode == "grammar":
        chosen = grammar
    elif template_mode == "logic":
        chosen = logic
    elif template_mode == "curriculum":
        chosen = grammar + logic
    else:
        raise ValueError(f"Unsupported template mode: {template_mode}")

    unique = sorted({" ".join(sent): sent for sent in chosen}.values())
    return [list(sent) for sent in unique]


def sample_sentence(template_mode: str) -> List[str]:
    grammar = enumerate_grammar_sentences()
    logic = enumerate_logic_sentences()
    if template_mode == "grammar":
        return list(random.choice(grammar))
    if template_mode == "logic":
        return list(random.choice(logic))
    if random.random() < 0.55:
        return list(random.choice(grammar))
    return list(random.choice(logic))


def build_windows_from_sentence(
    tokens: Sequence[str],
    bit_v0: float,
    bit_v1: float,
) -> Tuple[List[np.ndarray], List[int], List[Tuple[int, ...]]]:
    bos_id = WORD_TO_ID["<BOS>"]
    eos_token_id = WORD_TO_ID["<EOS>"]
    ids = [WORD_TO_ID[t] for t in tokens] + [eos_token_id]
    padded = [bos_id] * CONTEXT_LEN + ids

    xs: List[np.ndarray] = []
    ys: List[int] = []
    ctx_keys: List[Tuple[int, ...]] = []
    for i in range(CONTEXT_LEN, len(padded)):
        ctx = padded[i - CONTEXT_LEN:i]
        y_token_id = int(padded[i])
        if y_token_id == bos_id:
            raise RuntimeError("BOS should never appear as a next-token target")
        xs.append(encode_context_tokens(ctx, bit_v0=bit_v0, bit_v1=bit_v1))
        ys.append(TOKEN_ID_TO_OUTPUT_ID[y_token_id])
        ctx_keys.append(tuple(int(t) for t in ctx))
    return xs, ys, ctx_keys


def build_language_dataset(
    num_sentences: int,
    template_mode: str,
    bit_v0: float,
    bit_v1: float,
    max_sentence_words: int,
) -> Tuple[List[np.ndarray], List[int], List[Tuple[int, ...]], List[List[str]]]:
    all_x: List[np.ndarray] = []
    all_y: List[int] = []
    all_ctx_keys: List[Tuple[int, ...]] = []
    sentences: List[List[str]] = []
    while len(sentences) < int(num_sentences):
        sent = sample_sentence(template_mode)
        if len(sent) > int(max_sentence_words):
            continue
        x_s, y_s, ctx_s = build_windows_from_sentence(sent, bit_v0=bit_v0, bit_v1=bit_v1)
        all_x.extend(x_s)
        all_y.extend(y_s)
        all_ctx_keys.extend(ctx_s)
        sentences.append(sent)
    return all_x, all_y, all_ctx_keys, sentences


def build_grammar_support_map(template_mode: str) -> Dict[Tuple[int, ...], Set[int]]:
    support_map: Dict[Tuple[int, ...], Set[int]] = {}
    sentences = enumerate_allowed_sentences(template_mode)
    for sent in sentences:
        _, ys, ctxs = build_windows_from_sentence(sent, bit_v0=0.0, bit_v1=1.0)
        for ctx, y in zip(ctxs, ys):
            support_map.setdefault(tuple(int(v) for v in ctx), set()).add(int(y))
    return support_map


def one_hot(y: int, k: int) -> np.ndarray:
    q = np.zeros(k, dtype=float)
    q[int(y)] = 1.0
    return q


def top_words_from_q(q: np.ndarray, topk: int = 4) -> str:
    idx = np.argsort(-np.asarray(q, dtype=float))[:topk]
    parts: List[str] = []
    for i in idx.tolist():
        if q[i] <= 0.0:
            continue
        parts.append(f"{OUTPUT_ID_TO_WORD[int(i)]}:{float(q[i]):.2f}")
    return ", ".join(parts) if parts else "<none>"


def grammar_stats_from_outputs(
    vout: np.ndarray,
    y_true: int,
    valid_outputs: Set[int],
    temp: float,
) -> Dict[str, float]:
    probs = softmax_logits(np.asarray(vout, dtype=float) / float(temp))
    valid_idx = np.asarray(sorted(valid_outputs), dtype=int)
    valid_mass = float(np.sum(probs[valid_idx])) if valid_idx.size > 0 else 0.0
    return {
        "exact_acc": float(int(int(np.argmax(probs)) == int(y_true))),
        "qmass": float(probs[int(y_true)]),
        "support_acc": float(int(int(np.argmax(probs)) in valid_outputs)),
        "valid_mass": valid_mass,
        "valid_logprob": float(np.log(np.clip(valid_mass, 1e-12, 1.0))),
        "forbidden_mass": float(max(0.0, 1.0 - valid_mass)),
    }


def alter_outputs_onehot_forbidden(
    ng: NgSpiceShared,
    k: int,
    q_target: np.ndarray,
    vout_free: np.ndarray,
    delta: float,
    temp: float,
    valid_outputs: Optional[Set[int]],
    forbidden_boost: float,
) -> None:
    if temp <= 0.0:
        raise ValueError("--softmax-temp must be > 0")

    z = np.asarray(vout_free, dtype=float) / float(temp)
    p = softmax_logits(z)
    q = np.asarray(q_target, dtype=float)
    q = q / np.clip(float(np.sum(q)), 1e-12, None)
    d_v = float(delta) * (q - p)

    if valid_outputs is not None and forbidden_boost > 0.0:
        invalid_mask = np.ones(k, dtype=bool)
        invalid_mask[list(valid_outputs)] = False
        d_v[invalid_mask] -= float(delta) * float(forbidden_boost) * p[invalid_mask]

    v_clamp = np.asarray(vout_free, dtype=float) + d_v
    cmds: List[str] = [f"alter RS{i} {RS_CLAMP:.6g}" for i in range(1, k + 1)]
    for out_id in range(k):
        cmds.append(f"alter VOUT{out_id} dc = {float(v_clamp[out_id]):.16f}")
    exec_chunked(ng, cmds)


def greedy_generate_from_context(
    ng: NgSpiceShared,
    topo: DenseIOTopology,
    vg_unique: np.ndarray,
    netlist: str,
    seed_ctx_ids: List[int],
    bit_v0: float,
    bit_v1: float,
    max_len: int,
) -> List[str]:
    ctx = list(seed_ctx_ids)
    out_words: List[str] = []
    for _ in range(max_len):
        mk_free_all(ng, topo.K)
        xin = encode_context_tokens(ctx, bit_v0=bit_v0, bit_v1=bit_v1)
        alter_inputs_named(ng, xin)
        ok, _, data, _ = run_and_read(ng, {"out": topo.out_nodes.tolist()})
        if not ok or data is None:
            try:
                ng.remove_circuit()
            except Exception:
                pass
            ng.load_circuit(netlist)
            restore_gate_voltages(ng, vg_unique)
            mk_free_all(ng, topo.K)
            break

        vout = np.asarray(data["out"], dtype=float)
        if not np.all(np.isfinite(vout)):
            break
        out_id = pred_label(vout)
        token_id = OUTPUT_ID_TO_TOKEN_ID[int(out_id)]
        word = ID_TO_WORD[int(token_id)]
        out_words.append(word)
        ctx = ctx[1:] + [int(token_id)]
        if word == "<EOS>":
            break
    return out_words


def maybe_warm_start_quantum(
    vg_unique: np.ndarray,
    warm_start_run_dir: Optional[Path],
    warm_start_epoch: Optional[int],
) -> np.ndarray:
    if warm_start_run_dir is None or warm_start_epoch is None:
        return vg_unique

    legacy_path = warm_start_run_dir / f"0_vg_unique_epoch{int(warm_start_epoch)}.npy"
    if not legacy_path.exists():
        raise FileNotFoundError(f"Warm-start checkpoint not found: {legacy_path}")

    legacy = np.load(legacy_path)
    legacy_rows = legacy.reshape(16, INPUT_DIM)

    old_id = {
        "<BOS>": 0,
        "<EOS>": 1,
        "the": 2,
        "a": 3,
        "cat": 4,
        "dog": 5,
        "boy": 6,
        "girl": 7,
        "ball": 8,
        "mat": 9,
        "runs": 10,
        "sees": 11,
        "likes": 12,
        "is": 13,
        "red": 14,
        "on": 15,
    }
    row_map = {
        "<EOS>": legacy_rows[old_id["<EOS>"]],
        "the": legacy_rows[old_id["the"]],
        "a": legacy_rows[old_id["a"]],
        "qubit": legacy_rows[old_id["cat"]],
        "photon": legacy_rows[old_id["dog"]],
        "electron": legacy_rows[old_id["boy"]],
        "detector": legacy_rows[old_id["girl"]],
        "field": legacy_rows[old_id["ball"]],
        "state": legacy_rows[old_id["mat"]],
        "measures": legacy_rows[old_id["sees"]],
        "couples": legacy_rows[old_id["likes"]],
        "drives": legacy_rows[old_id["runs"]],
        "is": legacy_rows[old_id["is"]],
        "coherent": legacy_rows[old_id["red"]],
        "with": legacy_rows[old_id["on"]],
    }
    new_rows = np.stack([row_map[word] for word in OUTPUT_TOKENS], axis=0)
    return new_rows.reshape(-1).astype(float, copy=True)


def warm_start_args(args: argparse.Namespace) -> Tuple[Optional[Path], Optional[int]]:
    if args.warm_start_best_legacy:
        return DEFAULT_LEGACY_WARMSTART_RUN, DEFAULT_LEGACY_WARMSTART_EPOCH
    if args.warm_start_run_dir is None:
        return None, None
    if args.warm_start_epoch is None:
        raise ValueError("--warm-start-epoch is required with --warm-start-run-dir")
    return args.warm_start_run_dir.resolve(), int(args.warm_start_epoch)


def main() -> None:
    args = parse_args()
    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)

    epochs = int(args.epochs)
    gamma = float(args.gamma)
    delta = float(args.delta)
    temp = float(args.softmax_temp)
    forbidden_boost = float(args.forbidden_boost)
    bit_v0 = float(args.bit_v0)
    bit_v1 = float(args.bit_v1)
    if bit_v1 <= bit_v0:
        raise ValueError("--bit-v1 must be > --bit-v0")
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
    device_lib = str(args.device_lib)
    if not Path(device_lib).exists():
        raise FileNotFoundError(f"Device library not found: {device_lib}")

    template_mode = str(args.template_mode)
    num_sentences = int(args.num_sentences)
    max_sentence_words = int(args.max_sentence_words)
    if num_sentences < 1:
        raise ValueError("--num-sentences must be >= 1")
    if max_sentence_words < 3:
        raise ValueError("--max-sentence-words must be >= 3")

    support_map = build_grammar_support_map(template_mode)
    all_x, all_y, all_ctx_keys, sentences = build_language_dataset(
        num_sentences=num_sentences,
        template_mode=template_mode,
        bit_v0=bit_v0,
        bit_v1=bit_v1,
        max_sentence_words=max_sentence_words,
    )

    X_train, X_val, y_train, y_val, ctx_train, ctx_val = train_test_split(
        all_x,
        all_y,
        all_ctx_keys,
        test_size=float(args.val_frac),
        random_state=seed,
        stratify=np.asarray(all_y, dtype=int),
    )
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
    train_ctx = [tuple(int(t) for t in ctx) for ctx in ctx_train]
    val_x = [np.asarray(v, dtype=float) for v in X_val]
    val_y = [int(v) for v in y_val]
    val_ctx = [tuple(int(t) for t in ctx) for ctx in ctx_val]
    val_n = len(val_x)

    topo = make_dense_io_topology()
    if topo.Nin != INPUT_DIM or topo.K != OUTPUT_DIM:
        raise RuntimeError("Internal topology dimensions do not match quantum language task")

    if vg_init_mode == "fixed":
        vg_unique = np.full((topo.num_edges,), vg_init_single, dtype=float)
    else:
        if vg_init_hi <= vg_init_lo:
            raise ValueError("--vg-init-hi must be > --vg-init-lo for random init")
        vg_unique = np.random.uniform(vg_init_lo, vg_init_hi, size=(topo.num_edges,)).astype(float)

    warm_dir, warm_epoch = warm_start_args(args)
    vg_unique = maybe_warm_start_quantum(vg_unique, warm_dir, warm_epoch)

    results_dir = DEFAULT_RESULTS_DIR
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
        f"seed={seed} gamma={gamma} delta={delta} T={temp} forbidden_boost={forbidden_boost} "
        f"bit=[{bit_v0},{bit_v1}] context={CONTEXT_LEN} bits/token={TOKEN_BITS} Nin={topo.Nin} K={topo.K} "
        f"sentences={num_sentences} max_words={max_sentence_words} template={template_mode} "
        f"rails=[{vminus_val},{vplus_val}] solver={solver} body_tie={body_tie} rs_clamp={RS_CLAMP} "
        f"vg_init={vg_init_mode} epochs={epochs} device_include={device_lib} subckt={DEVICE_SUBCKT}"
    )

    print("=== RUN START (clln_dense_language16_quantum_onehotxent) ===", flush=True)
    print(cfg_str, flush=True)
    print(f"train_windows={len(train_x)} val_windows={len(val_x)} edges={topo.num_edges}", flush=True)

    (run_dir / "vocab.json").write_text(
        json.dumps(
            {
                "input_vocab": VOCAB,
                "output_vocab": OUTPUT_TOKENS,
                "word_to_id": WORD_TO_ID,
                "output_word_to_id": OUTPUT_WORD_TO_ID,
            },
            indent=2,
        )
    )
    (run_dir / "sample_sentences.txt").write_text("\n".join(" ".join(s) for s in sentences[:200]))

    preview_items = []
    for ctx, y in list(zip(train_ctx, train_y))[:100]:
        preview_items.append(
            {
                "context": [ID_TO_WORD[t] for t in ctx],
                "target_word": OUTPUT_ID_TO_WORD[int(y)],
                "valid_next_words": [OUTPUT_ID_TO_WORD[i] for i in sorted(support_map.get(ctx, set()))],
            }
        )
    (run_dir / "sample_target_preview.json").write_text(json.dumps(preview_items, indent=2))

    netlist = mk_netlist(
        topo=topo,
        vg_unique=vg_unique,
        vminus_val=vminus_val,
        vplus_val=vplus_val,
        solver=solver,
        body_res=body_res,
        body_tie=body_tie,
        device_lib_path=device_lib,
    )
    (run_dir / "netlist_initial.cir").write_text(netlist)

    meta = {
        "script": str(Path(__file__).resolve()),
        "argv": list(os.sys.argv),
        "seed": seed,
        "timestamp": datetime.now().isoformat(),
        "dataset": {
            "name": "synthetic_quantum_language_16_input_15_output",
            "num_sentences": num_sentences,
            "max_sentence_words": max_sentence_words,
            "template_mode": template_mode,
            "context_len": CONTEXT_LEN,
            "token_bits": TOKEN_BITS,
            "bit_v0": bit_v0,
            "bit_v1": bit_v1,
            "input_vocab": VOCAB,
            "output_vocab": OUTPUT_TOKENS,
            "support_contexts": len(support_map),
        },
        "train_count": len(train_x),
        "val_count": len(val_x),
        "gamma": gamma,
        "delta": delta,
        "softmax_temp": temp,
        "forbidden_boost": forbidden_boost,
        "epochs": epochs,
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
        "warm_start": {
            "run_dir": str(warm_dir) if warm_dir is not None else None,
            "epoch": warm_epoch,
        },
        "device": {
            "include_path": device_lib,
            "subckt": DEVICE_SUBCKT,
        },
        "topology": topo.meta,
        "loss": "onehot_cross_entropy_quantum",
        "generation": {
            "sample_prompts": int(args.sample_prompts),
            "sample_max_len": int(args.sample_max_len),
        },
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

    np.save(run_dir / "val_x.npy", np.asarray(val_x, dtype=float))
    np.save(run_dir / "val_y.npy", np.asarray(val_y, dtype=int))

    ng = NgSpiceShared(send_data=False)
    ng.load_circuit(netlist)

    net_nodes = [topo.negref, topo.posref] + topo.out_nodes.tolist() + topo.input_nodes.tolist()
    nodes_list = np.asarray(sorted(set(net_nodes)), dtype=int)
    index_of = np.full(nodes_list.max() + 1, -1, dtype=int)
    index_of[nodes_list] = np.arange(nodes_list.size, dtype=int)
    e_d = topo.edges_D
    e_s = topo.edges_S

    try:
        g = nx.DiGraph()
        g.add_nodes_from(nodes_list.tolist())
        for d, s in zip(e_d.tolist(), e_s.tolist()):
            g.add_edge(d, s)
        nx.write_graphml(g, str(run_dir / "0.graphml"))
    except Exception:
        pass

    val_acc_hist: List[float] = []
    val_ce_hist: List[float] = []
    val_support_hist: List[float] = []
    val_valid_mass_hist: List[float] = []
    val_valid_logprob_hist: List[float] = []
    val_forbidden_mass_hist: List[float] = []
    tr_acc_hist: List[float] = []
    tr_ce_hist: List[float] = []
    tr_support_hist: List[float] = []
    tr_valid_mass_hist: List[float] = []
    tr_valid_logprob_hist: List[float] = []
    tr_forbidden_mass_hist: List[float] = []
    ep_total_s: List[float] = []
    ep_free_s: List[float] = []
    ep_clamp_s: List[float] = []
    ep_update_s: List[float] = []
    reload_free_hist: List[int] = []
    reload_clamp_hist: List[int] = []
    nonfinite_free_hist: List[int] = []
    nonfinite_clamp_hist: List[int] = []

    def eval_free_metrics(epoch: int) -> Tuple[float, float, Dict[str, float]]:
        mk_free_all(ng, topo.K)
        correct = 0
        total = 0
        loss_sum = 0.0
        count = 0
        reloads = 0
        nonfinite = 0
        qmass_list: List[float] = []
        support_list: List[float] = []
        valid_mass_list: List[float] = []
        valid_logprob_list: List[float] = []
        forbidden_mass_list: List[float] = []
        confusion = np.zeros((topo.K, topo.K), dtype=int)
        vout_val = np.full((val_n, topo.K), np.nan, dtype=float) if val_n > 0 else np.zeros((0, topo.K), dtype=float)

        for i, (xv, yv, ctx) in enumerate(zip(val_x, val_y, val_ctx)):
            alter_inputs_named(ng, xv)
            ok, _, data, _ = run_and_read(ng, {"out": topo.out_nodes.tolist()})
            if not ok or data is None:
                reloads += 1
                try:
                    ng.remove_circuit()
                except Exception:
                    pass
                ng.load_circuit(netlist)
                restore_gate_voltages(ng, vg_unique)
                mk_free_all(ng, topo.K)
                continue

            vout = np.asarray(data["out"], dtype=float)
            if not np.all(np.isfinite(vout)):
                nonfinite += 1
                continue

            pred = pred_label(vout)
            q_target = one_hot(int(yv), topo.K)
            valid_outputs = support_map.get(ctx, {int(yv)})
            stats = grammar_stats_from_outputs(vout, int(yv), valid_outputs, temp=temp)

            correct += int(pred == int(yv))
            total += 1
            confusion[int(yv), pred] += 1

            ce, qmass = cross_entropy_from_outputs_soft(vout, q_target, temp=temp)
            loss_sum += float(ce)
            count += 1
            qmass_list.append(float(qmass))
            support_list.append(float(stats["support_acc"]))
            valid_mass_list.append(float(stats["valid_mass"]))
            valid_logprob_list.append(float(stats["valid_logprob"]))
            forbidden_mass_list.append(float(stats["forbidden_mass"]))
            vout_val[i, :] = vout

        if val_n > 0:
            np.save(run_dir / f"0_vout_val_epoch{epoch}.npy", vout_val)
            np.save(run_dir / f"0_val_confusion_epoch{epoch}.npy", confusion)

        diag = {
            "val_reloads": float(reloads),
            "val_nonfinite": float(nonfinite),
            "val_qmass_mean": float(np.mean(qmass_list)) if qmass_list else float("nan"),
            "val_support_acc": float(np.mean(support_list)) if support_list else float("nan"),
            "val_valid_mass_mean": float(np.mean(valid_mass_list)) if valid_mass_list else float("nan"),
            "val_valid_logprob_mean": float(np.mean(valid_logprob_list)) if valid_logprob_list else float("nan"),
            "val_forbidden_mass_mean": float(np.mean(forbidden_mass_list)) if forbidden_mass_list else float("nan"),
        }
        acc = (correct / total) if total else float("nan")
        loss = (loss_sum / count) if count else float("nan")
        return float(acc), float(loss), diag

    def save_generation_samples(epoch: int) -> None:
        n = min(int(args.sample_prompts), len(val_x))
        if n <= 0:
            return
        lines: List[str] = []
        prompt_indices = np.linspace(0, len(val_x) - 1, n, dtype=int)
        for j, idx in enumerate(prompt_indices.tolist()):
            seed_words = decode_context_bits_to_words(val_x[idx], bit_v0=bit_v0, bit_v1=bit_v1)
            seed_ids = [WORD_TO_ID[w] for w in seed_words]
            generated = greedy_generate_from_context(
                ng=ng,
                topo=topo,
                vg_unique=vg_unique,
                netlist=netlist,
                seed_ctx_ids=seed_ids,
                bit_v0=bit_v0,
                bit_v1=bit_v1,
                max_len=int(args.sample_max_len),
            )
            target = OUTPUT_ID_TO_WORD[int(val_y[idx])]
            q_target = one_hot(int(val_y[idx]), topo.K)
            valid_outputs = support_map.get(val_ctx[idx], {int(val_y[idx])})
            valid_words = [OUTPUT_ID_TO_WORD[i] for i in sorted(valid_outputs)]
            seed_clean = [w for w in seed_words if w != "<BOS>"]
            lines.append(f"[{j}] seed={' '.join(seed_clean) if seed_clean else '<PAD>'}")
            lines.append(f"    observed_next={target}")
            lines.append(f"    valid_next={', '.join(valid_words)}")
            lines.append(f"    target_dist_top={top_words_from_q(q_target, topk=4)}")
            lines.append(f"    generated={' '.join(generated) if generated else '<empty>'}")
        (run_dir / f"samples_epoch{epoch}.txt").write_text("\n".join(lines) + "\n")

    v0, ce0, diag0 = eval_free_metrics(epoch=0)
    val_acc_hist.append(v0)
    val_ce_hist.append(ce0)
    val_support_hist.append(float(diag0["val_support_acc"]))
    val_valid_mass_hist.append(float(diag0["val_valid_mass_mean"]))
    val_valid_logprob_hist.append(float(diag0["val_valid_logprob_mean"]))
    val_forbidden_mass_hist.append(float(diag0["val_forbidden_mass_mean"]))
    np.save(run_dir / "0_val_acc.npy", np.asarray(val_acc_hist, dtype=float))
    np.save(run_dir / "0_val_ce.npy", np.asarray(val_ce_hist, dtype=float))
    np.save(run_dir / "0_val_support_acc.npy", np.asarray(val_support_hist, dtype=float))
    np.save(run_dir / "0_val_valid_mass.npy", np.asarray(val_valid_mass_hist, dtype=float))
    np.save(run_dir / "0_val_valid_logprob.npy", np.asarray(val_valid_logprob_hist, dtype=float))
    np.save(run_dir / "0_val_forbidden_mass.npy", np.asarray(val_forbidden_mass_hist, dtype=float))
    (run_dir / "0_diag_epoch0.json").write_text(json.dumps(diag0, indent=2))
    save_generation_samples(epoch=0)
    print(
        f"[epoch 0] {cfg_str} | VAL exact_acc={v0:.4f} "
        f"softCE={ce0:.6f} valid_mass={diag0['val_valid_mass_mean']:.4f} "
        f"forbidden_mass={diag0['val_forbidden_mass_mean']:.4f} support_acc={diag0['val_support_acc']:.4f}",
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
        qmass_sum = 0.0
        qmass_count = 0
        support_sum = 0.0
        support_count = 0
        valid_mass_sum = 0.0
        valid_mass_count = 0
        valid_logprob_sum = 0.0
        valid_logprob_count = 0
        forbidden_mass_sum = 0.0
        forbidden_mass_count = 0
        reload_free = 0
        reload_clamp = 0
        nonfinite_free = 0
        nonfinite_clamp = 0
        t_free = 0.0
        t_clamp = 0.0
        t_update = 0.0
        n_free = 0
        n_clamp = 0

        for idx in order.tolist():
            ytrue = int(train_y[idx])
            ctx = train_ctx[idx]
            valid_outputs = support_map.get(ctx, {ytrue})
            q_target = one_hot(ytrue, topo.K)
            mk_free_all(ng, topo.K)

            alter_inputs_named(ng, train_x[idx])
            ok, dt, data, _ = run_and_read(ng, {"out": topo.out_nodes.tolist(), "nodes": nodes_list.tolist()})
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
                mk_free_all(ng, topo.K)
                continue

            vout = np.asarray(data["out"], dtype=float)
            vnodes_free = np.asarray(data["nodes"], dtype=float)
            if (not np.all(np.isfinite(vout))) or (not np.all(np.isfinite(vnodes_free))):
                nonfinite_free += 1
                continue

            pred = pred_label(vout)
            train_correct += int(pred == ytrue)
            train_total += 1

            stats = grammar_stats_from_outputs(vout, ytrue, valid_outputs, temp=temp)
            support_sum += float(stats["support_acc"])
            support_count += 1
            valid_mass_sum += float(stats["valid_mass"])
            valid_mass_count += 1
            valid_logprob_sum += float(stats["valid_logprob"])
            valid_logprob_count += 1
            forbidden_mass_sum += float(stats["forbidden_mass"])
            forbidden_mass_count += 1

            ce, qmass = cross_entropy_from_outputs_soft(vout, q_target, temp=temp)
            ce_sum += float(ce)
            ce_count += 1
            qmass_sum += float(qmass)
            qmass_count += 1

            clamp0 = time.time()
            alter_outputs_onehot_forbidden(
                ng=ng,
                k=topo.K,
                q_target=q_target,
                vout_free=vout,
                delta=delta,
                temp=temp,
                valid_outputs=valid_outputs,
                forbidden_boost=forbidden_boost,
            )
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
                mk_free_all(ng, topo.K)
                continue

            vnodes_clamp = np.asarray(data2["nodes"], dtype=float)
            if not np.all(np.isfinite(vnodes_clamp)):
                nonfinite_clamp += 1
                continue

            upd0 = time.time()
            v_d_free = vnodes_free[index_of[e_d]]
            v_s_free = vnodes_free[index_of[e_s]]
            v_d_clamp = vnodes_clamp[index_of[e_d]]
            v_s_clamp = vnodes_clamp[index_of[e_s]]
            d_v_free = v_d_free - v_s_free
            d_v_clamp = v_d_clamp - v_s_clamp
            update = -gamma * (d_v_clamp**2 - d_v_free**2)

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
        tr_ce = (ce_sum / ce_count) if ce_count else float("nan")
        tr_qmass_mean = (qmass_sum / qmass_count) if qmass_count else float("nan")
        tr_support_acc = (support_sum / support_count) if support_count else float("nan")
        tr_valid_mass_mean = (valid_mass_sum / valid_mass_count) if valid_mass_count else float("nan")
        tr_valid_logprob_mean = (valid_logprob_sum / valid_logprob_count) if valid_logprob_count else float("nan")
        tr_forbidden_mass_mean = (forbidden_mass_sum / forbidden_mass_count) if forbidden_mass_count else float("nan")

        tr_acc_hist.append(float(tr_acc))
        tr_ce_hist.append(float(tr_ce))
        tr_support_hist.append(float(tr_support_acc))
        tr_valid_mass_hist.append(float(tr_valid_mass_mean))
        tr_valid_logprob_hist.append(float(tr_valid_logprob_mean))
        tr_forbidden_mass_hist.append(float(tr_forbidden_mass_mean))
        reload_free_hist.append(int(reload_free))
        reload_clamp_hist.append(int(reload_clamp))
        nonfinite_free_hist.append(int(nonfinite_free))
        nonfinite_clamp_hist.append(int(nonfinite_clamp))

        v_acc, v_ce, diag = eval_free_metrics(epoch=ep)
        val_acc_hist.append(float(v_acc))
        val_ce_hist.append(float(v_ce))
        val_support_hist.append(float(diag["val_support_acc"]))
        val_valid_mass_hist.append(float(diag["val_valid_mass_mean"]))
        val_valid_logprob_hist.append(float(diag["val_valid_logprob_mean"]))
        val_forbidden_mass_hist.append(float(diag["val_forbidden_mass_mean"]))
        save_generation_samples(epoch=ep)

        ep_total = float(time.time() - t_ep0)
        ep_total_s.append(ep_total)
        ep_free_s.append(float(t_free))
        ep_clamp_s.append(float(t_clamp))
        ep_update_s.append(float(t_update))

        np.save(run_dir / "0_train_acc.npy", np.asarray(tr_acc_hist, dtype=float))
        np.save(run_dir / "0_train_ce.npy", np.asarray(tr_ce_hist, dtype=float))
        np.save(run_dir / "0_train_support_acc.npy", np.asarray(tr_support_hist, dtype=float))
        np.save(run_dir / "0_train_valid_mass.npy", np.asarray(tr_valid_mass_hist, dtype=float))
        np.save(run_dir / "0_train_valid_logprob.npy", np.asarray(tr_valid_logprob_hist, dtype=float))
        np.save(run_dir / "0_train_forbidden_mass.npy", np.asarray(tr_forbidden_mass_hist, dtype=float))
        np.save(run_dir / "0_val_acc.npy", np.asarray(val_acc_hist, dtype=float))
        np.save(run_dir / "0_val_ce.npy", np.asarray(val_ce_hist, dtype=float))
        np.save(run_dir / "0_val_support_acc.npy", np.asarray(val_support_hist, dtype=float))
        np.save(run_dir / "0_val_valid_mass.npy", np.asarray(val_valid_mass_hist, dtype=float))
        np.save(run_dir / "0_val_valid_logprob.npy", np.asarray(val_valid_logprob_hist, dtype=float))
        np.save(run_dir / "0_val_forbidden_mass.npy", np.asarray(val_forbidden_mass_hist, dtype=float))
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
                "forbidden_boost": forbidden_boost,
                "bit_v0": bit_v0,
                "bit_v1": bit_v1,
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
                "device_include_path": device_lib,
                "device_subckt": DEVICE_SUBCKT,
                "loss": "onehot_cross_entropy_quantum",
                "template_mode": template_mode,
            },
            "train": {
                "exact_acc": float(tr_acc),
                "soft_ce": float(tr_ce),
                "qmass_mean": float(tr_qmass_mean),
                "support_acc": float(tr_support_acc),
                "valid_mass_mean": float(tr_valid_mass_mean),
                "valid_logprob_mean": float(tr_valid_logprob_mean),
                "forbidden_mass_mean": float(tr_forbidden_mass_mean),
                "n_free": int(n_free),
                "n_clamp": int(n_clamp),
                "reload_free": int(reload_free),
                "reload_clamp": int(reload_clamp),
                "nonfinite_free": int(nonfinite_free),
                "nonfinite_clamp": int(nonfinite_clamp),
            },
            "val": {
                "exact_acc": float(v_acc),
                "soft_ce": float(v_ce),
                **{k: float(v) for k, v in diag.items()},
            },
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
            f"[epoch {ep}/{epochs}] {cfg_str} | TRAIN exact_acc={tr_acc:.4f} "
            f"softCE={tr_ce:.6f} support_acc={tr_support_acc:.4f} valid_mass={tr_valid_mass_mean:.4f} "
            f"forbidden_mass={tr_forbidden_mass_mean:.4f} free={n_free} clamp={n_clamp} "
            f"reloadF={reload_free} reloadC={reload_clamp} nonfiniteF={nonfinite_free} nonfiniteC={nonfinite_clamp}",
            flush=True,
        )
        print(
            f"[epoch {ep}/{epochs}] {cfg_str} | VAL exact_acc={v_acc:.4f} "
            f"softCE={v_ce:.6f} support_acc={diag['val_support_acc']:.4f} "
            f"valid_mass={diag['val_valid_mass_mean']:.4f} forbidden_mass={diag['val_forbidden_mass_mean']:.4f} | "
            f"timing total={ep_total:.2f}s free={t_free:.2f}s clamp={t_clamp:.2f}s upd={t_update:.2f}s",
            flush=True,
        )

        try:
            save_plots(run_dir)
        except Exception:
            pass

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
        print("FINAL val exact acc=", val_acc_hist[-1] if val_acc_hist else float("nan"), flush=True)

    print("=== RUN END (clln_dense_language16_quantum_onehotxent) ===", flush=True)
    try:
        log_f.flush()
        log_f.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
