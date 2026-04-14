#!/usr/bin/env python3
"""
Dense linear-resistor CLLN trainer for a 32-token quantum-mechanics toy language.

Key changes relative to the older 32-token embed4 trainer:
  - 6-token context window
  - 15D fixed embedding per token -> 90 input voltages total
  - 32-token vocabulary including <BOS>
  - prefix-driven probabilistic grammar for sentence generation
  - dense 90 -> 32 linear-resistor readout with CE-style nudged clamp

This keeps the same broad training idea as the linear-resistor CE trainer:
  - free phase: solve output voltages with RS_FREE clamp branch
  - clamped phase: nudge outputs toward a CE target with RS_CLAMP
  - local update on each input/output edge from squared drop contrast

The dataset is built from a left-to-right generative process rather than a flat
bag of windows, so the prefix distribution seen during training is aligned with
autoregressive rollout.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.model_selection import train_test_split


# -----------------------------------------------------------------------------
# Vocabulary + embedding
# -----------------------------------------------------------------------------

CONTEXT_LEN = 6
TOKEN_EMBED_DIM = 15
RS_FREE = 1.0e9
RS_CLAMP = 10.0
VG_CLIP_LO = 0.40
VG_CLIP_HI = 8.00
VG_INIT_SINGLE = 1.50
EDGE_CONDUCTANCE_SCALE = 1.0e-2
TERMINAL_PUNCT = {".", "?"}

VOCAB: List[str] = [
    "<BOS>", ".", "?", "the", "a", "what", "why", "in", "of",
    "is", "has", "can", "measure", "shows",
    "electron", "photon", "atom", "qubit",
    "spin", "phase", "energy", "basis",
    "wave", "field", "state", "system",
    "detector", "measurement", "superposition", "entanglement",
    "pure", "mixed",
]

TOKEN_TO_ID: Dict[str, int] = {w: i for i, w in enumerate(VOCAB)}
ID_TO_TOKEN: Dict[int, str] = {i: w for i, w in enumerate(VOCAB)}
INPUT_VOCAB = VOCAB
OUTPUT_VOCAB = VOCAB
INPUT_DIM = CONTEXT_LEN * TOKEN_EMBED_DIM
OUTPUT_DIM = len(OUTPUT_VOCAB)

# Dimension order:
# [BOS, TERM, DET, QW, LINK, COP, HAVE, MOD, LEXV, NOUN, ADJ, C1, C2, C3, C4]
TOKEN_EMBED_15D: Dict[str, List[float]] = {
    "<BOS>":         [1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.05, 0.05, 0.05, 0.05],
    ".":             [0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.08, 0.10, 0.06, 0.32],
    "?":             [0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.10, 0.12, 0.88, 0.68],
    "the":           [0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.12, 0.12, 0.12, 0.35],
    "a":             [0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.14, 0.14, 0.12, 0.65],
    "what":          [0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.15, 0.18, 0.85, 0.35],
    "why":           [0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.17, 0.20, 0.88, 0.65],
    "in":            [0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.22, 0.78, 0.20, 0.35],
    "of":            [0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.20, 0.84, 0.24, 0.65],
    "is":            [0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.55, 0.18, 0.88, 0.50],
    "has":           [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.00, 0.56, 0.50, 0.28, 0.48],
    "can":           [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.00, 0.58, 0.80, 0.22, 0.46],
    "measure":       [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.84, 0.82, 0.24, 0.33],
    "shows":         [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.00, 0.78, 0.82, 0.62, 0.74],
    "electron":      [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.82, 0.35, 0.15, 0.25],
    "photon":        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.78, 0.45, 0.15, 0.28],
    "atom":          [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.75, 0.30, 0.25, 0.35],
    "qubit":         [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.70, 0.40, 0.55, 0.45],
    "spin":          [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.25, 0.92, 0.20, 0.40],
    "phase":         [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.30, 0.88, 0.25, 0.45],
    "energy":        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.35, 0.78, 0.30, 0.55],
    "basis":         [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.25, 0.72, 0.35, 0.70],
    "wave":          [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.45, 0.55, 0.35, 0.50],
    "field":         [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.40, 0.50, 0.30, 0.60],
    "state":         [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.54, 0.58, 0.95, 0.60],
    "system":        [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.66, 0.50, 0.78, 0.66],
    "detector":      [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.70, 0.32, 0.24, 0.80],
    "measurement":   [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.48, 0.46, 0.34, 0.90],
    "superposition": [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.35, 0.70, 0.75, 0.90],
    "entanglement":  [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.00, 0.30, 0.68, 0.78, 0.95],
    "pure":          [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.90, 0.22, 0.90, 0.34],
    "mixed":         [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.94, 0.26, 0.86, 0.68],
}

START_TOKENS = ["the", "a", "what", "why"]

NOUN_CLASSES: Dict[str, str] = {
    "electron": "particle",
    "photon": "particle",
    "atom": "particle",
    "qubit": "particle",
    "spin": "property",
    "phase": "property",
    "energy": "property",
    "basis": "property",
    "wave": "medium",
    "field": "medium",
    "state": "stateful",
    "system": "stateful",
    "detector": "apparatus",
    "measurement": "apparatus",
    "superposition": "outcome",
    "entanglement": "outcome",
}

IS_ADJ_HEADS = {
    "state": ["pure", "mixed"],
    "system": ["pure", "mixed"],
    "qubit": ["pure", "mixed"],
    "superposition": ["pure", "mixed"],
    "entanglement": ["pure", "mixed"],
}

HAS_OBJECTS = {
    "electron": ["spin", "energy", "phase"],
    "photon": ["energy", "phase"],
    "atom": ["state", "energy", "phase"],
    "qubit": ["state", "phase", "basis"],
    "state": ["phase", "basis", "energy"],
    "system": ["state", "energy", "basis"],
}

CAN_MEASURE_OBJECTS = {
    "detector": ["spin", "phase", "energy", "basis", "state"],
    "system": ["spin", "phase", "state", "energy"],
    "measurement": ["spin", "phase", "energy", "basis"],
}

SHOWS_OBJECTS = {
    "measurement": ["state", "superposition", "entanglement", "phase"],
    "detector": ["state", "superposition", "entanglement", "measurement"],
    "wave": ["phase", "energy", "state"],
    "field": ["phase", "energy", "state"],
    "system": ["state", "superposition", "entanglement"],
}

WHAT_SIMPLE_HEADS = [
    "electron", "photon", "atom", "qubit",
    "spin", "phase", "energy", "basis",
    "wave", "field", "state", "system",
    "detector", "measurement", "superposition", "entanglement",
]

OF_HEAD_TO_TAIL = {
    "spin": ["electron", "atom", "qubit", "system"],
    "phase": ["photon", "wave", "field", "system", "state"],
    "energy": ["electron", "photon", "atom", "system", "field"],
    "basis": ["qubit", "state", "system"],
    "state": ["atom", "qubit", "system", "measurement"],
    "superposition": ["qubit", "system", "measurement"],
    "entanglement": ["system", "measurement"],
}

WHY_HEADS = {
    "state": ["pure", "mixed"],
    "system": ["pure", "mixed"],
    "qubit": ["pure", "mixed"],
    "superposition": ["pure", "mixed"],
    "entanglement": ["pure", "mixed"],
}

FAMILY_PRIORS_BALANCED = {
    "stmt_is_adj": 0.20,
    "stmt_has": 0.20,
    "stmt_can_measure": 0.20,
    "stmt_shows": 0.15,
    "q_what_simple": 0.10,
    "q_what_of": 0.07,
    "q_why": 0.08,
}

FAMILY_PRIORS_BROAD = {
    "stmt_is_adj": 0.18,
    "stmt_has": 0.18,
    "stmt_can_measure": 0.17,
    "stmt_shows": 0.13,
    "q_what_simple": 0.13,
    "q_what_of": 0.11,
    "q_why": 0.10,
}

# Tokens present in the vocabulary for representational completeness but not
# emitted by the current sentence generator should not participate in coverage
# topping, or the corpus builder will keep appending sentences until hitting
# its hard stop.
COVERAGE_EXEMPT_TOKENS = {"<BOS>", "in"}


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

@dataclass
class DenseIOTopology:
    Nin: int
    K: int
    num_edges: int
    input_nodes: np.ndarray
    out_nodes: np.ndarray
    meta: Dict[str, object]


@dataclass
class EvalOut:
    exact_acc: float
    support_acc: float
    soft_ce: float
    qmass_mean: float
    unseen_contexts: float
    nonfinite_rows: float


def make_dense_io_topology() -> DenseIOTopology:
    return DenseIOTopology(
        Nin=INPUT_DIM,
        K=OUTPUT_DIM,
        num_edges=INPUT_DIM * OUTPUT_DIM,
        input_nodes=np.arange(1, INPUT_DIM + 1, dtype=int),
        out_nodes=np.arange(INPUT_DIM + 1, INPUT_DIM + OUTPUT_DIM + 1, dtype=int),
        meta={"kind": "dense_io", "Nin": INPUT_DIM, "K": OUTPUT_DIM, "num_edges": INPUT_DIM * OUTPUT_DIM},
    )


def one_hot(idx: int, K: int) -> np.ndarray:
    v = np.zeros(K, dtype=float)
    v[int(idx)] = 1.0
    return v


def softmax_logits(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    z = z - np.max(z)
    ez = np.exp(z)
    s = np.sum(ez)
    if (not np.isfinite(s)) or s <= 0.0:
        return np.full_like(ez, 1.0 / len(ez))
    return ez / s


def softmax_rows(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    z = z - np.max(z, axis=1, keepdims=True)
    ez = np.exp(z)
    s = np.sum(ez, axis=1, keepdims=True)
    s = np.where((~np.isfinite(s)) | (s <= 0.0), 1.0, s)
    return ez / s


def pred_label(logits: np.ndarray) -> int:
    y = int(np.argmax(np.asarray(logits, dtype=float)))
    if ID_TO_TOKEN[y] == "<BOS>":
        # Prevent BOS from being treated as a surface prediction.
        logits2 = np.asarray(logits, dtype=float).copy()
        logits2[TOKEN_TO_ID["<BOS>"]] = -1e30
        y = int(np.argmax(logits2))
    return y


def encode_context_tokens(ctx_ids: Sequence[int]) -> np.ndarray:
    if len(ctx_ids) != CONTEXT_LEN:
        raise ValueError(f"expected context length {CONTEXT_LEN}, got {len(ctx_ids)}")
    vals: List[float] = []
    for tid in ctx_ids:
        vals.extend(TOKEN_EMBED_15D[ID_TO_TOKEN[int(tid)]])
    return np.asarray(vals, dtype=float)


def decode_context_ids_to_words(ctx_ids: Sequence[int]) -> List[str]:
    return [ID_TO_TOKEN[int(t)] for t in ctx_ids]


def top_words_from_q(q: np.ndarray, topk: int = 6) -> List[Tuple[str, float]]:
    q = np.asarray(q, dtype=float)
    idx = np.argsort(-q)[:topk]
    out: List[Tuple[str, float]] = []
    for i in idx:
        if q[i] <= 0.0:
            continue
        out.append((ID_TO_TOKEN[int(i)], float(q[i])))
    return out


# -----------------------------------------------------------------------------
# Prefix-driven grammar
# -----------------------------------------------------------------------------


def _weighted_choice(rng: random.Random, items: Dict[str, float]) -> str:
    keys = list(items.keys())
    probs = np.asarray([items[k] for k in keys], dtype=float)
    probs = probs / probs.sum()
    return keys[int(rng.choices(range(len(keys)), weights=probs, k=1)[0])]


def sample_sentence(rng: random.Random, template_mode: str = "broad") -> List[str]:
    priors = FAMILY_PRIORS_BROAD if template_mode == "broad" else FAMILY_PRIORS_BALANCED
    fam = _weighted_choice(rng, priors)

    det1 = rng.choice(["the", "a"])
    det2 = rng.choice(["the", "a"])

    if fam == "stmt_is_adj":
        subj = rng.choice(list(IS_ADJ_HEADS.keys()))
        adj = rng.choice(IS_ADJ_HEADS[subj])
        return [det1, subj, "is", adj, "."]

    if fam == "stmt_has":
        subj = rng.choice(list(HAS_OBJECTS.keys()))
        obj = rng.choice(HAS_OBJECTS[subj])
        return [det1, subj, "has", det2, obj, "."]

    if fam == "stmt_can_measure":
        subj = rng.choice(list(CAN_MEASURE_OBJECTS.keys()))
        obj = rng.choice(CAN_MEASURE_OBJECTS[subj])
        return [det1, subj, "can", "measure", det2, obj, "."]

    if fam == "stmt_shows":
        subj = rng.choice(list(SHOWS_OBJECTS.keys()))
        obj = rng.choice(SHOWS_OBJECTS[subj])
        return [det1, subj, "shows", det2, obj, "."]

    if fam == "q_what_simple":
        head = rng.choice(WHAT_SIMPLE_HEADS)
        return ["what", "is", det1, head, "?"]

    if fam == "q_what_of":
        head = rng.choice(list(OF_HEAD_TO_TAIL.keys()))
        tail = rng.choice(OF_HEAD_TO_TAIL[head])
        return ["what", "is", det1, head, "of", det2, tail, "?"]

    if fam == "q_why":
        subj = rng.choice(list(WHY_HEADS.keys()))
        adj = rng.choice(WHY_HEADS[subj])
        return ["why", "is", det1, subj, adj, "?"]

    raise RuntimeError(f"unknown family {fam}")


def build_sentence_corpus(
    *,
    num_sentences: int,
    template_mode: str,
    max_sentence_words: int,
    min_target_count: int,
    seed: int,
) -> Tuple[List[List[str]], Dict[str, int]]:
    rng = random.Random(seed)
    sentences: List[List[str]] = []
    target_counter: Counter[str] = Counter()
    max_tries = max(50 * num_sentences, 20000)
    tries = 0

    while len(sentences) < num_sentences and tries < max_tries:
        tries += 1
        s = sample_sentence(rng, template_mode=template_mode)
        if len(s) > max_sentence_words:
            continue
        sentences.append(s)
        target_counter.update(s)

    if len(sentences) < num_sentences:
        raise RuntimeError(f"Could only generate {len(sentences)} sentences out of requested {num_sentences}")

    # Coverage topping: ensure every non-BOS token appears at least min_target_count times.
    missing = True
    while missing and tries < max_tries * 3:
        tries += 1
        missing = False
        for tok in OUTPUT_VOCAB:
            if tok in COVERAGE_EXEMPT_TOKENS:
                continue
            if target_counter[tok] < min_target_count:
                missing = True
                s = sample_sentence(rng, template_mode="balanced")
                if len(s) <= max_sentence_words:
                    sentences.append(s)
                    target_counter.update(s)
                break

    return sentences, {tok: int(target_counter.get(tok, 0)) for tok in OUTPUT_VOCAB}


def build_windows_from_sentences(
    sentences: Sequence[Sequence[str]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Counter[str]]:
    X: List[np.ndarray] = []
    y: List[int] = []
    ctx_ids_all: List[Tuple[int, ...]] = []
    target_counter: Counter[str] = Counter()

    bos_id = TOKEN_TO_ID["<BOS>"]
    for sent in sentences:
        ctx = [bos_id] * CONTEXT_LEN
        for tok in sent:
            X.append(encode_context_tokens(ctx))
            y.append(TOKEN_TO_ID[tok])
            ctx_ids_all.append(tuple(ctx))
            ctx = ctx[1:] + [TOKEN_TO_ID[tok]]
            target_counter[tok] += 1

    return (
        np.asarray(X, dtype=float),
        np.asarray(y, dtype=int),
        np.asarray(ctx_ids_all, dtype=int),
        target_counter,
    )


# -----------------------------------------------------------------------------
# Target distributions + eval
# -----------------------------------------------------------------------------


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


# -----------------------------------------------------------------------------
# Linear-resistor backend
# -----------------------------------------------------------------------------


def params_to_gmat(vg_unique: np.ndarray, topo: DenseIOTopology) -> np.ndarray:
    return EDGE_CONDUCTANCE_SCALE * np.asarray(vg_unique, dtype=float).reshape(topo.K, topo.Nin)


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
    logits = np.asarray(Vout_free, dtype=float).copy()
    logits[TOKEN_TO_ID["<BOS>"]] = -1e30
    p = softmax_logits(logits / float(temp))
    q = np.asarray(q_target, dtype=float)
    q_sum = float(np.sum(q))
    if q_sum <= 0.0:
        raise ValueError("q_target must have positive mass")
    q = q / q_sum
    return np.asarray(Vout_free, dtype=float) + float(delta) * (q - p)


def eval_free_metrics(
    *,
    topo: DenseIOTopology,
    vg_unique: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    ctx: Sequence[Tuple[int, ...]],
    q_map_train: Dict[Tuple[int, ...], np.ndarray],
    unigram_q_train: np.ndarray,
    temp: float,
) -> EvalOut:
    gmat = params_to_gmat(vg_unique, topo)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    Vout = solve_outputs_batch_free(gmat, X)
    Vout[:, TOKEN_TO_ID["<BOS>"]] = -1e30
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

    if not np.any(valid_mask):
        return EvalOut(
            exact_acc=float("nan"),
            support_acc=float("nan"),
            soft_ce=float("nan"),
            qmass_mean=float("nan"),
            unseen_contexts=float(unseen),
            nonfinite_rows=float(np.sum(nonfinite_rows)),
        )

    q_valid = q[valid_mask]
    probs_valid = np.clip(probs[valid_mask], 1e-12, 1.0)
    y_valid = y[valid_mask]
    pred_valid = pred[valid_mask]

    exact_acc = float(np.mean(pred_valid == y_valid))
    support_acc = float(np.mean((q_valid[np.arange(q_valid.shape[0]), pred_valid] > 0.0).astype(float)))
    soft_ce = float(np.mean(-np.sum(q_valid * np.log(probs_valid), axis=1)))
    qmass_mean = float(np.mean(np.sum(probs_valid * (q_valid > 0.0), axis=1)))
    return EvalOut(
        exact_acc=exact_acc,
        support_acc=support_acc,
        soft_ce=soft_ce,
        qmass_mean=qmass_mean,
        unseen_contexts=float(unseen),
        nonfinite_rows=float(np.sum(nonfinite_rows)),
    )


# -----------------------------------------------------------------------------
# Generation
# -----------------------------------------------------------------------------


def autoregressive_generate(
    *,
    topo: DenseIOTopology,
    vg_unique: np.ndarray,
    max_len: int,
    temp: float,
    rng: np.random.Generator,
) -> List[str]:
    gmat = params_to_gmat(vg_unique, topo)
    ctx = [TOKEN_TO_ID["<BOS>"]] * CONTEXT_LEN
    out_words: List[str] = []
    for step in range(max_len):
        xin = encode_context_tokens(ctx)
        v = solve_outputs(gmat=gmat, xin=xin, clamp_res=RS_FREE, clamp_target=None)
        if not np.all(np.isfinite(v)):
            break
        logits = np.asarray(v, dtype=float).copy()
        logits[TOKEN_TO_ID["<BOS>"]] = -1e30
        if step == 0:
            start_mask = np.full_like(logits, -1e30)
            for w in START_TOKENS:
                start_mask[TOKEN_TO_ID[w]] = logits[TOKEN_TO_ID[w]]
            logits = start_mask
        p = softmax_logits(logits / float(temp))
        y = int(rng.choice(np.arange(OUTPUT_DIM), p=p))
        tok = ID_TO_TOKEN[y]
        out_words.append(tok)
        ctx = ctx[1:] + [y]
        if tok in TERMINAL_PUNCT:
            break
    return out_words


# -----------------------------------------------------------------------------
# CLI + training loop
# -----------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(description="Dense 90->32 QM language trainer, fixed 15D token embeddings, one-hot CE, linear resistor KCL backend")
    p.add_argument("seed", type=int, nargs="?", default=0)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--gamma", type=float, default=0.30)
    p.add_argument("--delta", type=float, default=0.30)
    p.add_argument("--softmax-temp", type=float, default=0.05)
    p.add_argument("--num-sentences", type=int, default=16000)
    p.add_argument("--min-target-count", type=int, default=120)
    p.add_argument("--max-sentence-words", type=int, default=9)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--max-train", type=int, default=0)
    p.add_argument("--max-val", type=int, default=0)
    p.add_argument("--template-mode", type=str, choices=["balanced", "broad"], default="broad")
    p.add_argument("--vg-init", type=str, choices=["random", "fixed"], default="random")
    p.add_argument("--vg-init-lo", type=float, default=1.0)
    p.add_argument("--vg-init-hi", type=float, default=3.0)
    p.add_argument("--vg-init-fixed", type=float, default=VG_INIT_SINGLE)
    p.add_argument("--sample-prompts", type=int, default=8)
    p.add_argument("--sample-max-len", type=int, default=12)
    p.add_argument("--gen-temp", type=float, default=0.10)
    p.add_argument("--results-dir", type=str, default="results_language_qm_embed15_linear")
    return p.parse_args()


def save_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2))


def main():
    args = parse_args()
    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    topo = make_dense_io_topology()

    sentences, coverage = build_sentence_corpus(
        num_sentences=int(args.num_sentences),
        template_mode=str(args.template_mode),
        max_sentence_words=int(args.max_sentence_words),
        min_target_count=int(args.min_target_count),
        seed=seed,
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

    train_ctx_tuples = [tuple(int(v) for v in row.tolist()) for row in np.asarray(ctx_train, dtype=int)]
    val_ctx_tuples = [tuple(int(v) for v in row.tolist()) for row in np.asarray(ctx_val, dtype=int)]
    q_map_train = build_context_target_distributions(train_ctx_tuples, y_train.tolist(), OUTPUT_DIM)
    unigram_q_train = build_unigram_target_distribution(y_train.tolist(), OUTPUT_DIM)

    if args.vg_init == "fixed":
        vg_unique = np.full((topo.num_edges,), float(args.vg_init_fixed), dtype=float)
    else:
        vg_unique = np.random.uniform(float(args.vg_init_lo), float(args.vg_init_hi), size=(topo.num_edges,)).astype(float)
    np.clip(vg_unique, VG_CLIP_LO, VG_CLIP_HI, out=vg_unique)

    env_run_dir = os.environ.get("RUN_DIR")
    if env_run_dir:
        run_dir = Path(env_run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        results_dir = run_dir.parent
    else:
        results_dir = Path(args.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        run_dir = results_dir / (datetime.now().strftime("%Y%m%d-%H%M%S") + f"_seed-{seed}")
        run_dir.mkdir(parents=True, exist_ok=True)

    save_json(run_dir / "run_meta.json", {
        "seed": seed,
        "epochs": int(args.epochs),
        "gamma": float(args.gamma),
        "delta": float(args.delta),
        "softmax_temp": float(args.softmax_temp),
        "context_len": CONTEXT_LEN,
        "token_embed_dim": TOKEN_EMBED_DIM,
        "input_dim": INPUT_DIM,
        "output_dim": OUTPUT_DIM,
        "num_sentences_requested": int(args.num_sentences),
        "num_sentences_actual": len(sentences),
        "template_mode": str(args.template_mode),
        "train_sentence_count": len(train_sentences),
        "val_sentence_count": len(val_sentences),
        "train_windows": int(len(X_train)),
        "val_windows": int(len(X_val)),
        "backend": "linear_resistor_kcl",
        "edge_conductance_scale": EDGE_CONDUCTANCE_SCALE,
    })
    save_json(run_dir / "token_embed_15d.json", TOKEN_EMBED_15D)
    save_json(run_dir / "target_coverage_total.json", coverage)
    save_json(run_dir / "target_coverage_train.json", {k: int(train_target_counter.get(k, 0)) for k in OUTPUT_VOCAB})
    save_json(run_dir / "target_coverage_val.json", {k: int(val_target_counter.get(k, 0)) for k in OUTPUT_VOCAB})
    (run_dir / "sample_sentences.txt").write_text("\n".join(" ".join(s) for s in sentences[:300]) + "\n")

    np.save(run_dir / "train_x.npy", X_train)
    np.save(run_dir / "train_y.npy", y_train)
    np.save(run_dir / "train_ctx.npy", ctx_train)
    np.save(run_dir / "val_x.npy", X_val)
    np.save(run_dir / "val_y.npy", y_val)
    np.save(run_dir / "val_ctx.npy", ctx_val)

    gmat = params_to_gmat(vg_unique, topo)
    param_mat = vg_unique.reshape(topo.K, topo.Nin)
    sum_g = np.sum(gmat, axis=1)

    train_hist = []
    val_hist = []
    dV_free = np.empty_like(param_mat)
    dV_clamp = np.empty_like(param_mat)
    update = np.empty_like(param_mat)
    param_next = np.empty_like(param_mat)
    delta_param = np.empty_like(param_mat)

    for epoch in range(int(args.epochs) + 1):
        if epoch == 0:
            tr_eval = eval_free_metrics(
                topo=topo,
                vg_unique=vg_unique,
                X=X_train,
                y=y_train,
                ctx=train_ctx_tuples,
                q_map_train=q_map_train,
                unigram_q_train=unigram_q_train,
                temp=float(args.softmax_temp),
            )
            va_eval = eval_free_metrics(
                topo=topo,
                vg_unique=vg_unique,
                X=X_val,
                y=y_val,
                ctx=val_ctx_tuples,
                q_map_train=q_map_train,
                unigram_q_train=unigram_q_train,
                temp=float(args.softmax_temp),
            )
        else:
            t0 = time.time()
            order = np.arange(X_train.shape[0], dtype=int)
            np.random.shuffle(order)
            for idx in order:
                x = np.asarray(X_train[idx], dtype=float)
                ytrue = int(y_train[idx])
                q_target = one_hot(ytrue, topo.K)

                v_free = solve_outputs(gmat=gmat, xin=x, clamp_res=RS_FREE, clamp_target=None, sum_g=sum_g)
                if not np.all(np.isfinite(v_free)):
                    continue
                v_free[TOKEN_TO_ID["<BOS>"]] = -1e30

                clamp_target = clamp_targets_from_free(v_free, q_target=q_target, delta=float(args.delta), temp=float(args.softmax_temp))
                v_clamp = solve_outputs(gmat=gmat, xin=x, clamp_res=RS_CLAMP, clamp_target=clamp_target, sum_g=sum_g)
                if not np.all(np.isfinite(v_clamp)):
                    continue

                np.subtract(v_free[:, None], x[None, :], out=dV_free)
                np.subtract(v_clamp[:, None], x[None, :], out=dV_clamp)
                np.square(dV_clamp, out=update)
                np.square(dV_free, out=dV_free)
                np.subtract(update, dV_free, out=update)
                update *= -float(args.gamma)
                np.add(param_mat, update, out=param_next)
                np.clip(param_next, VG_CLIP_LO, VG_CLIP_HI, out=param_next)
                np.subtract(param_next, param_mat, out=delta_param)
                param_mat[...] = param_next
                delta_param *= EDGE_CONDUCTANCE_SCALE
                gmat += delta_param
                sum_g += np.sum(delta_param, axis=1)

            tr_eval = eval_free_metrics(
                topo=topo,
                vg_unique=vg_unique,
                X=X_train,
                y=y_train,
                ctx=train_ctx_tuples,
                q_map_train=q_map_train,
                unigram_q_train=unigram_q_train,
                temp=float(args.softmax_temp),
            )
            va_eval = eval_free_metrics(
                topo=topo,
                vg_unique=vg_unique,
                X=X_val,
                y=y_val,
                ctx=val_ctx_tuples,
                q_map_train=q_map_train,
                unigram_q_train=unigram_q_train,
                temp=float(args.softmax_temp),
            )
            print(f"epoch {epoch:03d} done in {time.time()-t0:.2f}s")

        train_hist.append({
            "epoch": epoch,
            "exact_acc": tr_eval.exact_acc,
            "support_acc": tr_eval.support_acc,
            "soft_ce": tr_eval.soft_ce,
            "qmass_mean": tr_eval.qmass_mean,
        })
        val_hist.append({
            "epoch": epoch,
            "exact_acc": va_eval.exact_acc,
            "support_acc": va_eval.support_acc,
            "soft_ce": va_eval.soft_ce,
            "qmass_mean": va_eval.qmass_mean,
            "unseen_contexts": va_eval.unseen_contexts,
        })

        np.save(run_dir / f"vg_unique_epoch{epoch}.npy", vg_unique.copy())
        save_json(run_dir / "train_history.json", train_hist)
        save_json(run_dir / "val_history.json", val_hist)

        print(
            f"[epoch {epoch:03d}] train exact={tr_eval.exact_acc:.4f} support={tr_eval.support_acc:.4f} "
            f"ce={tr_eval.soft_ce:.4f} qmass={tr_eval.qmass_mean:.4f} | "
            f"val exact={va_eval.exact_acc:.4f} support={va_eval.support_acc:.4f} "
            f"ce={va_eval.soft_ce:.4f} qmass={va_eval.qmass_mean:.4f} unseen={va_eval.unseen_contexts:.1f}"
        )

        # save a few greedy-ish sampled generations each epoch
        samples = []
        for _ in range(int(args.sample_prompts)):
            sent = autoregressive_generate(
                topo=topo,
                vg_unique=vg_unique,
                max_len=int(args.sample_max_len),
                temp=float(args.gen_temp),
                rng=rng,
            )
            samples.append(" ".join(sent))
        (run_dir / f"samples_epoch{epoch}.txt").write_text("\n".join(samples) + "\n")

    print(f"saved run to {run_dir}")


if __name__ == "__main__":
    main()
