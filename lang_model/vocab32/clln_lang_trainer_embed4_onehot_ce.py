#!/usr/bin/env python3
"""
CLLN language generation trainer — dense 24-d input (6 context tokens × 4D embedding)
-> 32-token output NMOS network.

One-hot cross-entropy training (SGD, no batching) with two-phase (free / clamp)
ngspice runs.

Task:
  - Synthetic next-token prediction on a 32-token output vocabulary.
  - Input vocabulary has 33 symbols because <BOS> is input-only.
  - Context window: 6 tokens.
  - Token encoding: fixed 4D analog embedding per token.
  - Input dimension: 6 * 4 = 24 input nodes.
  - Output dimension: 32 output nodes, one per output token.

One-hot CE target:
  - For each training window (x, y), use only that sample's next-token label:
        q = one_hot(y)
  - Clamp rule:
        p = softmax(Vout_free / T)
        Vout_clamp = Vout_free + delta * (q - p)

Topology:
  - No hidden nodes.
  - Every input node is connected to every output node by exactly one NMOS edge.
  - Total trainable edges = 24 * 32 = 768.

Recommended first use:
  python lang_model/vocab32/clln_lang_trainer_embed4_onehot_ce.py 0 --epochs 20 --num-sentences 12000
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from PySpice.Spice.NgSpice.Shared import NgSpiceShared
from sklearn.model_selection import train_test_split


# -------------------------
# Paths / device model
# -------------------------
def find_repo_root() -> Path:
    here = Path(__file__).resolve()
    candidates = [here.parent] + list(here.parents)
    for cand in candidates:
        if (cand / "device_model" / "nmos_lvl1_ald1106.lib").exists():
            return cand
    return here.parent


REPO_ROOT = find_repo_root()
DEFAULT_DEVICE_LIB_PATH = str(REPO_ROOT / "device_model" / "nmos_lvl1_ald1106.lib")
DEVICE_SUBCKT = "NMOSWRAP"
VG_CLIP_LO, VG_CLIP_HI = 0.4, 8.0
VG_INIT_SINGLE = 2.0
RS_FREE = 1e9
RS_CLAMP = 10.0

CONTEXT_LEN = 6
TOKEN_EMBED_DIM = 4
INPUT_DIM = CONTEXT_LEN * TOKEN_EMBED_DIM  # 24


# -------------------------
# Vocabulary / embeddings
# -------------------------
INPUT_VOCAB: List[str] = [
    "<BOS>", ".", "!", "?", ",",
    "I", "you", "we", "they",
    "robot", "signal", "city", "river", "fire", "dream",
    "see", "hear", "build", "break", "follow", "remember",
    "am", "are", "is", "not", "why", "where",
    "bright", "lost", "alive", "strange", "here", "again",
]
OUTPUT_VOCAB: List[str] = [tok for tok in INPUT_VOCAB if tok != "<BOS>"]
OUTPUT_DIM = len(OUTPUT_VOCAB)  # 32

assert len(INPUT_VOCAB) == 33
assert len(OUTPUT_VOCAB) == 32

INPUT_WORD_TO_ID: Dict[str, int] = {w: i for i, w in enumerate(INPUT_VOCAB)}
INPUT_ID_TO_WORD: Dict[int, str] = {i: w for i, w in enumerate(INPUT_VOCAB)}
OUTPUT_WORD_TO_ID: Dict[str, int] = {w: i for i, w in enumerate(OUTPUT_VOCAB)}
OUTPUT_ID_TO_WORD: Dict[int, str] = {i: w for i, w in enumerate(OUTPUT_VOCAB)}

TOKEN_EMBED_4D: Dict[str, List[float]] = {
    "<BOS>":    [0.0548, 0.2737, 0.2228, 0.0737],
    ".":        [0.0551, 0.3377, 0.2024, 0.1676],
    "!":        [0.0506, 0.3574, 0.2041, 0.2185],
    "?":        [0.0500, 0.3599, 0.2043, 0.2249],
    ",":        [0.0661, 0.3354, 0.1972, 0.1738],
    "I":        [0.1370, 0.0836, 0.1491, 0.1275],
    "you":      [0.1407, 0.1103, 0.1422, 0.1410],
    "we":       [0.1451, 0.1180, 0.1303, 0.1448],
    "they":     [0.1493, 0.1505, 0.1247, 0.1671],
    "robot":    [0.3759, 0.2272, 0.0814, 0.1824],
    "signal":   [0.3832, 0.2459, 0.0977, 0.2447],
    "city":     [0.3882, 0.3455, 0.0500, 0.3662],
    "river":    [0.3943, 0.3691, 0.0565, 0.3258],
    "fire":     [0.3994, 0.3518, 0.1009, 0.2331],
    "dream":    [0.3396, 0.0500, 0.1658, 0.2851],
    "see":      [0.3871, 0.3228, 0.3351, 0.0500],
    "hear":     [0.3786, 0.2705, 0.3512, 0.0965],
    "build":    [0.4000, 0.3817, 0.2963, 0.0505],
    "break":    [0.3965, 0.3984, 0.3257, 0.1082],
    "follow":   [0.3880, 0.3388, 0.3112, 0.1092],
    "remember": [0.3514, 0.1267, 0.4000, 0.1441],
    "am":       [0.1022, 0.2471, 0.3548, 0.3079],
    "are":      [0.1060, 0.2708, 0.3520, 0.3077],
    "is":       [0.1095, 0.3001, 0.3413, 0.3485],
    "not":      [0.1052, 0.2410, 0.3286, 0.2805],
    "why":      [0.0862, 0.3254, 0.2482, 0.2693],
    "where":    [0.1004, 0.4000, 0.2114, 0.3066],
    "bright":   [0.3556, 0.2715, 0.3055, 0.3872],
    "lost":     [0.3367, 0.1825, 0.3325, 0.3818],
    "alive":    [0.3505, 0.1889, 0.3196, 0.3099],
    "strange":  [0.3401, 0.1833, 0.3414, 0.4000],
    "here":     [0.1728, 0.3828, 0.2071, 0.2963],
    "again":    [0.1395, 0.1937, 0.3100, 0.1916],
}

PRONOUNS = ["I", "you", "we", "they"]
NOUNS = ["robot", "signal", "city", "river", "fire", "dream"]
VERBS = ["see", "hear", "build", "break", "follow", "remember"]
DESCRIPTORS = ["bright", "lost", "alive", "strange"]
STATEMENT_END = [".", "!"]
QUESTION_END = ["?"]
TERMINAL_PUNCT = {".", "!", "?"}


@dataclass
class DenseIOTopology:
    Nin: int
    K: int
    input_nodes: np.ndarray
    out_nodes: np.ndarray
    negref: int
    posref: int
    edges_D: np.ndarray
    edges_S: np.ndarray
    meta: Dict[str, object]

    @property
    def num_edges(self) -> int:
        return int(self.edges_D.size)


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

    sys.stdout = Tee(sys.stdout, log_f)  # type: ignore
    sys.stderr = Tee(sys.stderr, log_f)  # type: ignore
    return log_f


# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="CLLN dense 24->32 language trainer, fixed 4D token embeddings, one-hot cross entropy, ngspice"
    )
    p.add_argument("seed", type=int, nargs="?", default=0)
    p.add_argument("--epochs", type=int, default=20)

    # Learning hyperparams
    p.add_argument("--gamma", type=float, default=0.30)
    p.add_argument("--delta", type=float, default=0.30)
    p.add_argument("--softmax-temp", type=float, default=1.0)

    # Rails
    p.add_argument("--vminus", type=float, default=0.0)
    p.add_argument("--vplus", type=float, default=0.45)

    # Dataset generation
    p.add_argument("--num-sentences", type=int, default=12000, help="Minimum synthetic sentences to generate")
    p.add_argument("--min-target-count", type=int, default=80, help="Continue generating until every output token appears at least this many times as a training target")
    p.add_argument("--max-sentence-words", type=int, default=9, help="Discard generated sentences longer than this many words")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--max-train", type=int, default=0, help="If >0, limit train windows")
    p.add_argument("--max-val", type=int, default=0, help="If >0, limit val windows")
    p.add_argument(
        "--template-mode",
        type=str,
        choices=["balanced", "broad"],
        default="broad",
        help="balanced = roughly uniform template sampling; broad = slightly reweighted to emphasize punctuation, negation, questions, and comma clauses",
    )

    # Device / solver
    p.add_argument("--device-lib", type=str, default=DEFAULT_DEVICE_LIB_PATH)
    p.add_argument("--body-tie", type=str, choices=["source", "ground", "floating"], default="ground")
    p.add_argument("--solver", type=str, choices=["klu", "sparse"], default="klu")

    # Init VG
    p.add_argument("--vg-init", type=str, choices=["random", "fixed"], default="random")
    p.add_argument("--vg-init-lo", type=float, default=1.0)
    p.add_argument("--vg-init-hi", type=float, default=3.0)
    p.add_argument("--vg-init-fixed", type=float, default=VG_INIT_SINGLE)

    p.add_argument("--sample-prompts", type=int, default=8, help="How many sample generations to save each epoch")
    p.add_argument("--sample-max-len", type=int, default=12, help="Max generated tokens after the seed context")
    p.add_argument("--final-val", action="store_true")
    p.add_argument("--worker-run-dir", type=str, default="", help=argparse.SUPPRESS)
    p.add_argument("--worker-epoch", type=int, default=-1, help=argparse.SUPPRESS)
    return p.parse_args()


# -------------------------
# Synthetic language data
# -------------------------
def subject_copula(pron: str) -> str:
    if pron == "I":
        return "am"
    return "are"


def sample_statement_punct() -> str:
    return random.choice(STATEMENT_END)


def sample_subject_token() -> str:
    if random.random() < 0.55:
        return random.choice(PRONOUNS)
    return random.choice(NOUNS)


def sample_predicate_verb() -> str:
    return random.choice(VERBS)


def sentence_templates() -> List[str]:
    return [
        "pron_desc",
        "pron_not_desc",
        "noun_desc",
        "noun_not_desc",
        "pron_here",
        "pron_not_here",
        "noun_here",
        "verb_stmt_pron",
        "verb_stmt_noun",
        "verb_stmt_pron_again",
        "verb_stmt_noun_again",
        "why_desc",
        "where_pron",
        "where_noun",
    ]


def sample_sentence(template_mode: str) -> List[str]:
    templates = sentence_templates()
    if template_mode == "balanced":
        template = random.choice(templates)
    else:
        template = random.choices(
            population=templates,
            weights=[
                1.2, 1.0, 1.2, 1.0,
                1.1, 0.9, 1.0,
                1.2, 1.2, 1.0, 1.0,
                1.1, 1.0, 1.0,
            ],
            k=1,
        )[0]

    if template == "pron_desc":
        subj = random.choice(PRONOUNS)
        cop = subject_copula(subj)
        return [subj, cop, random.choice(DESCRIPTORS), sample_statement_punct()]

    if template == "pron_not_desc":
        subj = random.choice(PRONOUNS)
        cop = subject_copula(subj)
        return [subj, cop, "not", random.choice(DESCRIPTORS), sample_statement_punct()]

    if template == "noun_desc":
        subj = random.choice(NOUNS)
        return [subj, "is", random.choice(DESCRIPTORS), sample_statement_punct()]

    if template == "noun_not_desc":
        subj = random.choice(NOUNS)
        return [subj, "is", "not", random.choice(DESCRIPTORS), sample_statement_punct()]

    if template == "pron_here":
        subj = random.choice(PRONOUNS)
        cop = subject_copula(subj)
        end = sample_statement_punct()
        if random.random() < 0.35:
            return [subj, cop, "here", ",", "again", end]
        return [subj, cop, "here", end]

    if template == "pron_not_here":
        subj = random.choice(PRONOUNS)
        cop = subject_copula(subj)
        return [subj, cop, "not", "here", sample_statement_punct()]

    if template == "noun_here":
        subj = random.choice(NOUNS)
        if random.random() < 0.35:
            return [subj, "is", "here", ",", "again", sample_statement_punct()]
        return [subj, "is", "here", sample_statement_punct()]

    if template == "verb_stmt_pron":
        subj = random.choice(PRONOUNS)
        return [subj, sample_predicate_verb(), random.choice(NOUNS), sample_statement_punct()]

    if template == "verb_stmt_noun":
        subj = random.choice(NOUNS)
        return [subj, sample_predicate_verb(), random.choice(NOUNS), sample_statement_punct()]

    if template == "verb_stmt_pron_again":
        subj = random.choice(PRONOUNS)
        return [subj, sample_predicate_verb(), random.choice(NOUNS), ",", "again", sample_statement_punct()]

    if template == "verb_stmt_noun_again":
        subj = random.choice(NOUNS)
        return [subj, sample_predicate_verb(), random.choice(NOUNS), ",", "again", sample_statement_punct()]

    if template == "why_desc":
        subj = sample_subject_token()
        if subj in PRONOUNS:
            cop = subject_copula(subj)
        else:
            cop = "is"
        if random.random() < 0.5:
            return ["why", subj, cop, random.choice(DESCRIPTORS), random.choice(QUESTION_END)]
        return ["why", subj, cop, "not", random.choice(DESCRIPTORS), random.choice(QUESTION_END)]

    if template == "where_pron":
        subj = random.choice(PRONOUNS)
        cop = subject_copula(subj)
        return ["where", subj, cop, random.choice(QUESTION_END)]

    if template == "where_noun":
        subj = random.choice(NOUNS)
        return ["where", subj, "is", random.choice(QUESTION_END)]

    raise RuntimeError(f"Unhandled template: {template}")


def token_id_to_embed4(tok_id: int) -> np.ndarray:
    tok = INPUT_ID_TO_WORD[int(tok_id)]
    return np.asarray(TOKEN_EMBED_4D[tok], dtype=float)


def encode_context_tokens(ctx_ids: Sequence[int]) -> np.ndarray:
    if len(ctx_ids) != CONTEXT_LEN:
        raise ValueError(f"Expected context length {CONTEXT_LEN}, got {len(ctx_ids)}")
    vals: List[float] = []
    for tid in ctx_ids:
        vals.extend(token_id_to_embed4(int(tid)).tolist())
    return np.asarray(vals, dtype=float)


def build_windows_from_sentence(
    tokens: Sequence[str],
) -> Tuple[List[np.ndarray], List[int], List[Tuple[int, ...]]]:
    bos_id = INPUT_WORD_TO_ID["<BOS>"]
    padded = [bos_id] * CONTEXT_LEN + [INPUT_WORD_TO_ID[t] for t in tokens]

    xs: List[np.ndarray] = []
    ys: List[int] = []
    ctx_keys: List[Tuple[int, ...]] = []

    for i in range(CONTEXT_LEN, len(padded)):
        ctx = padded[i - CONTEXT_LEN:i]
        next_input_id = padded[i]
        next_tok = INPUT_ID_TO_WORD[int(next_input_id)]
        if next_tok == "<BOS>":
            raise RuntimeError("<BOS> should never be a target token")
        y = OUTPUT_WORD_TO_ID[next_tok]
        xs.append(encode_context_tokens(ctx))
        ys.append(int(y))
        ctx_keys.append(tuple(int(t) for t in ctx))
    return xs, ys, ctx_keys


def build_windows_from_sentences(
    sentences: Sequence[Sequence[str]],
) -> Tuple[List[np.ndarray], List[int], List[Tuple[int, ...]], Counter]:
    all_x: List[np.ndarray] = []
    all_y: List[int] = []
    all_ctx_keys: List[Tuple[int, ...]] = []
    target_counter: Counter = Counter()
    for sent in sentences:
        x_s, y_s, ctx_s = build_windows_from_sentence(sent)
        all_x.extend(x_s)
        all_y.extend(y_s)
        all_ctx_keys.extend(ctx_s)
        for y in y_s:
            target_counter[OUTPUT_ID_TO_WORD[int(y)]] += 1
    return all_x, all_y, all_ctx_keys, target_counter


def build_sentence_corpus(
    num_sentences: int,
    template_mode: str,
    max_sentence_words: int,
    min_target_count: int,
) -> Tuple[List[List[str]], Dict[str, int]]:
    if num_sentences < 1:
        raise ValueError("num_sentences must be >= 1")
    if min_target_count < 0:
        raise ValueError("min_target_count must be >= 0")

    sentences: List[List[str]] = []
    target_counter: Counter = Counter()
    attempts = 0
    max_attempts = max(10000, int(num_sentences) * 80)

    def coverage_ok() -> bool:
        if min_target_count <= 0:
            return True
        return all(target_counter.get(tok, 0) >= min_target_count for tok in OUTPUT_VOCAB)

    while (len(sentences) < int(num_sentences)) or (not coverage_ok()):
        if attempts >= max_attempts:
            break
        attempts += 1
        sent = sample_sentence(template_mode)
        if len(sent) > int(max_sentence_words):
            continue
        x_s, y_s, _ = build_windows_from_sentence(sent)
        if not x_s or not y_s:
            continue
        sentences.append(sent)
        for y in y_s:
            target_counter[OUTPUT_ID_TO_WORD[int(y)]] += 1

    coverage = {tok: int(target_counter.get(tok, 0)) for tok in OUTPUT_VOCAB}
    return sentences, coverage


# -------------------------
# Target-distribution helpers
# -------------------------
def one_hot(y: int, K: int) -> np.ndarray:
    q = np.zeros(K, dtype=float)
    q[int(y)] = 1.0
    return q


def top_words_from_q(q: np.ndarray, topk: int = 4) -> str:
    q = np.asarray(q, dtype=float)
    idx = np.argsort(-q)[:topk]
    parts: List[str] = []
    for i in idx:
        if q[i] <= 0.0:
            continue
        parts.append(f"{OUTPUT_ID_TO_WORD[int(i)]}:{float(q[i]):.2f}")
    return ", ".join(parts) if parts else "<none>"


# -------------------------
# Topology builder
# -------------------------
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
        "token_embed_dim": TOKEN_EMBED_DIM,
        "input_vocab_size": len(INPUT_VOCAB),
        "output_vocab_size": len(OUTPUT_VOCAB),
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


def run_and_read(ng: NgSpiceShared, read_specs: Dict[str, List[int]]) -> Tuple[bool, float, Optional[Dict[str, np.ndarray]], Optional[str]]:
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


def alter_outputs_xent_soft(
    ng: NgSpiceShared,
    K: int,
    q_target: np.ndarray,
    Vout_free: np.ndarray,
    delta: float,
    temp: float,
):
    """
    Soft-target CE clamp:
      p = softmax(Vout_free / temp)
      Vout_clamp = Vout_free + delta * (q - p)
    """
    if temp <= 0.0:
        raise ValueError("--softmax-temp must be > 0")

    z = np.asarray(Vout_free, dtype=float) / float(temp)
    p = softmax_logits(z)

    q = np.asarray(q_target, dtype=float)
    s = float(np.sum(q))
    if s <= 0.0:
        raise ValueError("q_target must have positive sum")
    q = q / s

    dV = float(delta) * (q - p)
    Vc = np.asarray(Vout_free, dtype=float) + dV

    cmds: List[str] = []
    for i in range(1, K + 1):
        cmds.append(f"alter RS{i} {RS_CLAMP:.6g}")
    for k in range(K):
        cmds.append(f"alter VOUT{k} dc = {float(Vc[k]):.16f}")
    exec_chunked(ng, cmds)


def mk_netlist(
    topo: DenseIOTopology,
    vg_unique: np.ndarray,
    vminus_val: float,
    vplus_val: float,
    solver: str,
    body_res: float,
    body_tie: str,
    device_lib_path: str,
) -> str:
    if vg_unique.size != topo.num_edges:
        raise ValueError("vg_unique size mismatch")

    lines: List[str] = []
    lines.append(".title clln_dense_language32_embed4_onehotxent")
    lines.append(f'.include "{device_lib_path}"')

    if solver.lower() == "klu":
        lines.append(".options klu")

    lines.append(f"VMINUS {topo.negref} 0 {float(vminus_val):.16f}")
    lines.append(f"VPLUS  {topo.posref} 0 {float(vplus_val):.16f}")

    for i, n in enumerate(topo.input_nodes):
        lines.append(f"VIN{i} {n} 0 0")

    node_pool = [topo.negref, topo.posref] + topo.input_nodes.tolist() + topo.out_nodes.tolist()
    max_node = max(node_pool)
    sink0 = max_node + 1
    K = len(topo.out_nodes)

    for i, on in enumerate(topo.out_nodes, start=1):
        lines.append(f"RS{i} {on} {sink0 + (i - 1)} {RS_FREE:.6g}")
    for j in range(K):
        lines.append(f"VOUT{j} {sink0 + j} 0 0")

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

        lines.append(f"X{eidx} {D} {gate_node} {S} {body_node} {DEVICE_SUBCKT}")

    lines.extend([
        ".options TEMP = 27C",
        ".options TNOM = 27C",
        ".options itl1=40 itl2=40 itl4=6 itl5=60",
        ".options gmin=1e-8 reltol=5e-3 abstol=1e-8 vntol=1e-5",
        ".options rshunt=1e9",
        ".op",
        ".end",
    ])
    return "\n".join(lines) + "\n"


# -------------------------
# Metrics / generation helpers
# -------------------------
def cross_entropy_from_outputs_soft(
    V: np.ndarray,
    q_target: np.ndarray,
    temp: float,
) -> Tuple[float, float]:
    """
    Returns:
      ce    = -sum_k q_k log p_k
      qmass = total predicted probability mass on support(q)
    """
    if temp <= 0.0:
        return float("inf"), float("nan")

    z = np.asarray(V, dtype=float) / float(temp)
    p = softmax_logits(z)

    q = np.asarray(q_target, dtype=float)
    qs = float(np.sum(q))
    if qs <= 0.0:
        return float("inf"), float("nan")
    q = q / qs

    p_safe = np.clip(p, 1e-12, 1.0)
    ce = float(-np.sum(q * np.log(p_safe)))
    qmass = float(np.sum(p[q > 0]))
    return ce, qmass


def pred_label(V: np.ndarray) -> int:
    return int(np.argmax(np.asarray(V, dtype=float)))


def restore_gate_voltages(ng: NgSpiceShared, vg_unique: np.ndarray):
    exec_chunked(ng, [f"alter VG{i} dc = {float(v):.16f}" for i, v in enumerate(vg_unique)])


def compute_vg_saturation_stats(vg_unique: np.ndarray) -> Dict[str, float]:
    return {
        "vg_unique_min": float(np.min(vg_unique)),
        "vg_unique_max": float(np.max(vg_unique)),
        "vg_unique_sat_lo": float(np.sum(vg_unique <= (VG_CLIP_LO + 1e-12))),
        "vg_unique_sat_hi": float(np.sum(vg_unique >= (VG_CLIP_HI - 1e-12))),
    }


def decode_context_ids_to_words(ctx_ids: Sequence[int]) -> List[str]:
    return [INPUT_ID_TO_WORD[int(t)] for t in ctx_ids]


def greedy_generate_from_context(
    ng: NgSpiceShared,
    topo: DenseIOTopology,
    vg_unique: np.ndarray,
    netlist: str,
    seed_ctx_ids: List[int],
    max_len: int,
) -> List[str]:
    ctx = list(seed_ctx_ids)
    out_words: List[str] = []
    for _ in range(max_len):
        mk_free_all(ng, topo.K)
        xin = encode_context_tokens(ctx)
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

        Vout = np.asarray(data["out"], dtype=float)
        if not np.all(np.isfinite(Vout)):
            break
        yhat = pred_label(Vout)
        w = OUTPUT_ID_TO_WORD[int(yhat)]
        out_words.append(w)
        ctx = ctx[1:] + [INPUT_WORD_TO_ID[w]]
        if w in TERMINAL_PUNCT:
            break
    return out_words


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
        plt.plot(np.arange(len(va_acc)), va_acc, label="val exact acc")
        plt.plot(np.arange(1, len(tr_acc) + 1), tr_acc, label="train exact acc")
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
        plt.plot(np.arange(len(va_ce)), va_ce, label="val soft CE")
        plt.plot(np.arange(1, len(tr_ce) + 1), tr_ce, label="train soft CE")
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
) -> str:
    return (
        f"seed={seed} gamma={gamma} delta={delta} T={temp} "
        f"context={CONTEXT_LEN} embed_dim={TOKEN_EMBED_DIM} Nin={topo.Nin} K={topo.K} "
        f"sentences={num_sentences_actual} min_target_count={min_target_count} max_words={max_sentence_words} template={template_mode} "
        f"rails=[{vminus_val},{vplus_val}] solver={solver} body_tie={body_tie} rs_clamp={RS_CLAMP} "
        f"vg_init={vg_init_mode} epochs={epochs} device_include={device_lib} subckt={DEVICE_SUBCKT}"
    )


def load_hist_list(run_dir: Path, name: str, dtype=float) -> List[float]:
    path = run_dir / name
    if not path.exists():
        return []
    arr = np.asarray(np.load(path), dtype=dtype)
    return arr.tolist()


def save_history_arrays(
    run_dir: Path,
    *,
    tr_acc_hist: Sequence[float],
    tr_ce_hist: Sequence[float],
    val_acc_hist: Sequence[float],
    val_ce_hist: Sequence[float],
    ep_total_s: Sequence[float],
    ep_free_s: Sequence[float],
    ep_clamp_s: Sequence[float],
    ep_update_s: Sequence[float],
    reload_free_hist: Sequence[int],
    reload_clamp_hist: Sequence[int],
    nonfinite_free_hist: Sequence[int],
    nonfinite_clamp_hist: Sequence[int],
):
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


def load_saved_dataset(run_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_x = np.asarray(np.load(run_dir / "train_x.npy"), dtype=float)
    train_y = np.asarray(np.load(run_dir / "train_y.npy"), dtype=int)
    val_x = np.asarray(np.load(run_dir / "val_x.npy"), dtype=float)
    val_y = np.asarray(np.load(run_dir / "val_y.npy"), dtype=int)
    val_ctx = np.asarray(np.load(run_dir / "val_ctx.npy"), dtype=int)
    return train_x, train_y, val_x, val_y, val_ctx


def eval_free_metrics(
    *,
    ng: NgSpiceShared,
    topo: DenseIOTopology,
    netlist: str,
    vg_unique: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    val_n: int,
    run_dir: Path,
    temp: float,
    epoch: int,
) -> Tuple[float, float, Dict[str, float]]:
    mk_free_all(ng, topo.K)
    correct = 0
    total = 0
    loss_sum = 0.0
    count = 0
    reloads = 0
    nonfinite = 0
    qmass_list: List[float] = []
    confusion = np.zeros((topo.K, topo.K), dtype=int)
    vout_val = np.full((val_n, topo.K), np.nan, dtype=float) if val_n > 0 else np.zeros((0, topo.K), dtype=float)

    for i, (xv, yv) in enumerate(zip(val_x, val_y)):
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

        Vout = np.asarray(data["out"], dtype=float)
        if not np.all(np.isfinite(Vout)):
            nonfinite += 1
            continue

        pred = pred_label(Vout)
        ytrue = int(yv)
        q_target = one_hot(ytrue, topo.K)

        correct += int(pred == ytrue)
        total += 1
        confusion[ytrue, pred] += 1

        ce, qmass = cross_entropy_from_outputs_soft(Vout, q_target, temp=temp)
        loss_sum += float(ce)
        count += 1
        if np.isfinite(qmass):
            qmass_list.append(float(qmass))
        vout_val[i, :] = Vout

    if val_n > 0:
        np.save(run_dir / f"0_vout_val_epoch{epoch}.npy", vout_val)
        np.save(run_dir / f"0_val_confusion_epoch{epoch}.npy", confusion)

    diag = {
        "val_reloads": float(reloads),
        "val_nonfinite": float(nonfinite),
        "val_qmass_mean": float(np.mean(qmass_list)) if qmass_list else float("nan"),
    }
    acc = (correct / total) if total else float("nan")
    loss = (loss_sum / count) if count else float("nan")
    return float(acc), float(loss), diag


def save_generation_samples(
    *,
    args: argparse.Namespace,
    ng: NgSpiceShared,
    topo: DenseIOTopology,
    vg_unique: np.ndarray,
    netlist: str,
    val_ctx: np.ndarray,
    val_y: np.ndarray,
    run_dir: Path,
    epoch: int,
):
    n = min(int(args.sample_prompts), len(val_ctx))
    if n <= 0:
        return
    lines: List[str] = []
    prompt_indices = np.linspace(0, len(val_ctx) - 1, n, dtype=int)
    for j, idx in enumerate(prompt_indices.tolist()):
        seed_ids = [int(v) for v in val_ctx[idx].tolist()]
        seed_words = decode_context_ids_to_words(seed_ids)
        generated = greedy_generate_from_context(
            ng=ng,
            topo=topo,
            vg_unique=vg_unique,
            netlist=netlist,
            seed_ctx_ids=seed_ids,
            max_len=int(args.sample_max_len),
        )
        target = OUTPUT_ID_TO_WORD[int(val_y[idx])]
        q_target = one_hot(int(val_y[idx]), topo.K)
        seed_clean = [w for w in seed_words if w != "<BOS>"]
        gen_clean = [w for w in generated if w != "<BOS>"]
        lines.append(f"[{j}] seed={' '.join(seed_clean) if seed_clean else '<BOS>'}")
        lines.append(f"    observed_next={target}")
        lines.append(f"    target_dist_top={top_words_from_q(q_target, topk=4)}")
        lines.append(f"    generated={' '.join(gen_clean) if gen_clean else '<empty>'}")
    (run_dir / f"samples_epoch{epoch}.txt").write_text("\n".join(lines) + "\n")


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
    min_target_count = int(args.min_target_count)
    max_sentence_words = int(args.max_sentence_words)

    train_x, train_y, val_x, val_y, val_ctx = load_saved_dataset(run_dir)
    val_n = int(val_x.shape[0])

    topo = make_dense_io_topology()
    cfg_str = build_cfg_str(
        seed=seed,
        gamma=gamma,
        delta=delta,
        temp=temp,
        topo=topo,
        num_sentences_actual=int(json.loads((run_dir / "run_meta.json").read_text())["dataset"]["num_sentences_actual"]),
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
    )

    prev_epoch = 0 if epoch <= 0 else epoch - 1
    vg_path = run_dir / f"0_vg_unique_epoch{prev_epoch}.npy"
    if not vg_path.exists():
        raise FileNotFoundError(f"Required gate-state file not found: {vg_path}")
    vg_unique = np.asarray(np.load(vg_path), dtype=float)

    netlist = (run_dir / "netlist_initial.cir").read_text()
    ng = NgSpiceShared(send_data=False)
    ng.load_circuit(netlist)
    restore_gate_voltages(ng, vg_unique)

    net_nodes = [topo.negref, topo.posref] + topo.out_nodes.tolist() + topo.input_nodes.tolist()
    nodes_list = np.asarray(sorted(set(net_nodes)), dtype=int)
    index_of = np.full(nodes_list.max() + 1, -1, dtype=int)
    index_of[nodes_list] = np.arange(nodes_list.size, dtype=int)
    eD = topo.edges_D
    eS = topo.edges_S

    if epoch == 0:
        val_acc_hist: List[float] = []
        val_ce_hist: List[float] = []
        v0, ce0, diag0 = eval_free_metrics(
            ng=ng,
            topo=topo,
            netlist=netlist,
            vg_unique=vg_unique,
            val_x=val_x,
            val_y=val_y,
            val_n=val_n,
            run_dir=run_dir,
            temp=temp,
            epoch=0,
        )
        val_acc_hist.append(v0)
        val_ce_hist.append(ce0)
        save_history_arrays(
            run_dir,
            tr_acc_hist=[],
            tr_ce_hist=[],
            val_acc_hist=val_acc_hist,
            val_ce_hist=val_ce_hist,
            ep_total_s=[],
            ep_free_s=[],
            ep_clamp_s=[],
            ep_update_s=[],
            reload_free_hist=[],
            reload_clamp_hist=[],
            nonfinite_free_hist=[],
            nonfinite_clamp_hist=[],
        )
        (run_dir / "0_diag_epoch0.json").write_text(json.dumps(diag0, indent=2))
        save_generation_samples(
            args=args,
            ng=ng,
            topo=topo,
            vg_unique=vg_unique,
            netlist=netlist,
            val_ctx=val_ctx,
            val_y=val_y,
            run_dir=run_dir,
            epoch=0,
        )
        print(
            f"[epoch 0] {cfg_str} | VAL exact_acc={v0:.4f} "
            f"softCE={ce0:.6f} qmass_mean={diag0.get('val_qmass_mean', float('nan')):.4f}",
            flush=True,
        )
        try:
            save_plots(run_dir)
        except Exception:
            pass
        try:
            ng.remove_circuit()
        except Exception:
            pass
        try:
            log_f.flush()
            log_f.close()
        except Exception:
            pass
        return

    tr_acc_hist = load_hist_list(run_dir, "0_train_acc.npy", dtype=float)
    tr_ce_hist = load_hist_list(run_dir, "0_train_ce.npy", dtype=float)
    val_acc_hist = load_hist_list(run_dir, "0_val_acc.npy", dtype=float)
    val_ce_hist = load_hist_list(run_dir, "0_val_ce.npy", dtype=float)
    ep_total_s = load_hist_list(run_dir, "0_epoch_total_s.npy", dtype=float)
    ep_free_s = load_hist_list(run_dir, "0_epoch_free_s.npy", dtype=float)
    ep_clamp_s = load_hist_list(run_dir, "0_epoch_clamp_s.npy", dtype=float)
    ep_update_s = load_hist_list(run_dir, "0_epoch_update_s.npy", dtype=float)
    reload_free_hist = load_hist_list(run_dir, "0_reload_free.npy", dtype=int)
    reload_clamp_hist = load_hist_list(run_dir, "0_reload_clamp.npy", dtype=int)
    nonfinite_free_hist = load_hist_list(run_dir, "0_nonfinite_free.npy", dtype=int)
    nonfinite_clamp_hist = load_hist_list(run_dir, "0_nonfinite_clamp.npy", dtype=int)

    t_ep0 = time.time()
    order = np.arange(train_x.shape[0], dtype=int)
    np.random.shuffle(order)

    train_correct = 0
    train_total = 0
    ce_sum = 0.0
    ce_count = 0
    qmass_sum = 0.0
    qmass_count = 0
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

        Vout = np.asarray(data["out"], dtype=float)
        Vnodes_free = np.asarray(data["nodes"], dtype=float)
        if (not np.all(np.isfinite(Vout))) or (not np.all(np.isfinite(Vnodes_free))):
            nonfinite_free += 1
            continue

        pred = pred_label(Vout)
        train_correct += int(pred == ytrue)
        train_total += 1

        ce, qmass = cross_entropy_from_outputs_soft(Vout, q_target, temp=temp)
        ce_sum += float(ce)
        ce_count += 1
        if np.isfinite(qmass):
            qmass_sum += float(qmass)
            qmass_count += 1

        clamp0 = time.time()
        alter_outputs_xent_soft(
            ng,
            K=topo.K,
            q_target=q_target,
            Vout_free=Vout,
            delta=delta,
            temp=temp,
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

        Vnodes_clamp = np.asarray(data2["nodes"], dtype=float)
        if not np.all(np.isfinite(Vnodes_clamp)):
            nonfinite_clamp += 1
            continue

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
    tr_qmass_mean = (qmass_sum / qmass_count) if qmass_count else float("nan")

    tr_acc_hist.append(float(tr_acc))
    tr_ce_hist.append(float(tr_ce))
    reload_free_hist.append(int(reload_free))
    reload_clamp_hist.append(int(reload_clamp))
    nonfinite_free_hist.append(int(nonfinite_free))
    nonfinite_clamp_hist.append(int(nonfinite_clamp))

    v_acc, v_ce, diag = eval_free_metrics(
        ng=ng,
        topo=topo,
        netlist=netlist,
        vg_unique=vg_unique,
        val_x=val_x,
        val_y=val_y,
        val_n=val_n,
        run_dir=run_dir,
        temp=temp,
        epoch=epoch,
    )
    val_acc_hist.append(float(v_acc))
    val_ce_hist.append(float(v_ce))
    save_generation_samples(
        args=args,
        ng=ng,
        topo=topo,
        vg_unique=vg_unique,
        netlist=netlist,
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
        tr_ce_hist=tr_ce_hist,
        val_acc_hist=val_acc_hist,
        val_ce_hist=val_ce_hist,
        ep_total_s=ep_total_s,
        ep_free_s=ep_free_s,
        ep_clamp_s=ep_clamp_s,
        ep_update_s=ep_update_s,
        reload_free_hist=reload_free_hist,
        reload_clamp_hist=reload_clamp_hist,
        nonfinite_free_hist=nonfinite_free_hist,
        nonfinite_clamp_hist=nonfinite_clamp_hist,
    )
    np.save(run_dir / f"0_vg_unique_epoch{epoch}.npy", vg_unique.copy())

    vg_stats = compute_vg_saturation_stats(vg_unique)
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
            "loss": "onehot_cross_entropy",
            "template_mode": template_mode,
            "epoch_process_mode": "fresh_process",
        },
        "train": {
            "exact_acc": float(tr_acc),
            "soft_ce": float(tr_ce),
            "qmass_mean": float(tr_qmass_mean),
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
    (run_dir / f"0_epoch_summary_epoch{epoch}.json").write_text(json.dumps(summary, indent=2))
    (run_dir / f"0_diag_epoch{epoch}.json").write_text(json.dumps(diag, indent=2))

    print(
        f"[epoch {epoch}/{epochs}] {cfg_str} | TRAIN exact_acc={tr_acc:.4f} "
        f"softCE={tr_ce:.6f} qmass_mean={tr_qmass_mean:.4f} free={n_free} clamp={n_clamp} "
        f"reloadF={reload_free} reloadC={reload_clamp} nonfiniteF={nonfinite_free} nonfiniteC={nonfinite_clamp}",
        flush=True,
    )
    print(
        f"[epoch {epoch}/{epochs}] {cfg_str} | VAL exact_acc={v_acc:.4f} "
        f"softCE={v_ce:.6f} qmass_mean={diag.get('val_qmass_mean', float('nan')):.4f} | "
        f"timing total={ep_total:.2f}s free={t_free:.2f}s clamp={t_clamp:.2f}s upd={t_update:.2f}s",
        flush=True,
    )

    try:
        save_plots(run_dir)
    except Exception:
        pass

    try:
        ng.remove_circuit()
    except Exception:
        pass
    try:
        log_f.flush()
        log_f.close()
    except Exception:
        pass


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

    topo = make_dense_io_topology()
    if topo.Nin != INPUT_DIM or topo.K != OUTPUT_DIM:
        raise RuntimeError("Internal topology dimensions do not match language task")

    if vg_init_mode == "fixed":
        vg_unique = np.full((topo.num_edges,), vg_init_single, dtype=float)
    else:
        if vg_init_hi <= vg_init_lo:
            raise ValueError("--vg-init-hi must be > --vg-init-lo for random init")
        vg_unique = np.random.uniform(vg_init_lo, vg_init_hi, size=(topo.num_edges,)).astype(float)

    results_dir = Path(__file__).resolve().parent / "results_language_32_embed4_onehotce"
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
    )

    print("=== RUN START (clln_dense_language32_embed4_onehotxent) ===", flush=True)
    print(cfg_str, flush=True)
    print(
        f"train_sentences={len(train_sentences)} val_sentences={len(val_sentences)} "
        f"train_windows={len(train_x)} val_windows={len(val_x)} edges={topo.num_edges}",
        flush=True,
    )
    print("epoch execution mode=fresh_process", flush=True)

    (run_dir / "input_vocab.json").write_text(json.dumps({"input_vocab": INPUT_VOCAB, "word_to_id": INPUT_WORD_TO_ID}, indent=2))
    (run_dir / "output_vocab.json").write_text(json.dumps({"output_vocab": OUTPUT_VOCAB, "word_to_id": OUTPUT_WORD_TO_ID}, indent=2))
    (run_dir / "token_embed_4d.json").write_text(json.dumps(TOKEN_EMBED_4D, indent=2))
    (run_dir / "sample_sentences.txt").write_text("\n".join(" ".join(s) for s in sentences[:300]))
    (run_dir / "target_coverage_total.json").write_text(json.dumps(total_target_coverage, indent=2))
    (run_dir / "target_coverage_train.json").write_text(json.dumps({k: int(train_target_counter.get(k, 0)) for k in OUTPUT_VOCAB}, indent=2))
    (run_dir / "target_coverage_val.json").write_text(json.dumps({k: int(val_target_counter.get(k, 0)) for k in OUTPUT_VOCAB}, indent=2))

    preview_items = []
    for ctx, y in list(zip(train_ctx, train_y))[:120]:
        words = [INPUT_ID_TO_WORD[t] for t in ctx]
        preview_items.append({
            "context": words,
            "target_word": OUTPUT_ID_TO_WORD[int(y)],
        })
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
        "body_res": body_res,
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
            "epoch_process_mode": "fresh_process",
            "worker_python": sys.executable,
            "worker_dataset_files": [
                "train_x.npy",
                "train_y.npy",
                "val_x.npy",
                "val_y.npy",
                "val_ctx.npy",
            ],
        },
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

    np.save(run_dir / "train_x.npy", np.asarray(train_x, dtype=float))
    np.save(run_dir / "train_y.npy", np.asarray(train_y, dtype=int))
    np.save(run_dir / "val_x.npy", np.asarray(val_x, dtype=float))
    np.save(run_dir / "val_y.npy", np.asarray(val_y, dtype=int))
    np.save(run_dir / "val_ctx.npy", np.asarray(val_ctx, dtype=int))
    np.save(run_dir / "0_vg_unique_epoch0.npy", vg_unique.copy())

    try:
        net_nodes = [topo.negref, topo.posref] + topo.out_nodes.tolist() + topo.input_nodes.tolist()
        G = nx.DiGraph()
        G.add_nodes_from(sorted(set(net_nodes)))
        for d, s in zip(topo.edges_D.tolist(), topo.edges_S.tolist()):
            G.add_edge(d, s)
        nx.write_graphml(G, str(run_dir / "0.graphml"))
    except Exception:
        pass

    env = os.environ.copy()
    env["RUN_DIR"] = str(run_dir)
    for epoch in range(0, epochs + 1):
        print(f"[controller] launching worker epoch={epoch}", flush=True)
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

    print("=== RUN END (clln_dense_language32_embed4_onehotxent) ===", flush=True)
    try:
        log_f.flush()
        log_f.close()
    except Exception:
        pass


# -------------------------
# Main
# -------------------------
def main():
    args = parse_args()
    if args.worker_run_dir:
        run_worker_epoch(args)
        return
    run_controller(args)


if __name__ == "__main__":
    main()
