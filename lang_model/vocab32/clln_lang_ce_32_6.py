#!/usr/bin/env python3
"""
CLLN language generation trainer — dense 30-bit input -> 32-word output NMOS network
Soft-target cross-entropy training (SGD, no batching) with two-phase (free / clamp)
ngspice runs.

Task:
  - Synthetic next-token prediction on a 32-word vocabulary.
  - Context window: 6 tokens.
  - Token encoding: fixed 5-bit binary code per token.
  - Input dimension: 6 * 5 = 30 input nodes.
  - Output dimension: 32 output nodes, one per vocabulary word.

Soft CE target:
  - For each exact 6-token context x, build q(.|x) from the TRAINING split only:
        q_k = count(context=x, next_word=k) / total_count(context=x)
  - Clamp rule:
        p = softmax(Vout_free / T)
        Vout_clamp = Vout_free + delta * (q - p)

Topology:
  - No hidden nodes.
  - Every input node is connected to every output node by exactly one NMOS edge.
  - Total trainable edges = 30 * 32 = 960.

Dataset size:
  - By default, generate until there are ~10,000 total windows.
  - With val_frac=0.2 this gives ~8,000 train windows and ~2,000 val windows.

Recommended first use:
  python lang_model/vocab32/clln_lang_ce_32_6.py 0 --epochs 35
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
TOKEN_BITS = 5
VOCAB_SIZE = 32
INPUT_DIM = CONTEXT_LEN * TOKEN_BITS  # 30
OUTPUT_DIM = VOCAB_SIZE               # 32
STOP_TOKENS = {".", "?"}


# -------------------------
# Vocabulary / synthetic grammar
# -------------------------
VOCAB: List[str] = [
    "<BOS>", ".", "?", "the", "a",
    "cat", "dog", "boy", "girl", "robot",
    "ball", "mat", "tree", "park",
    "red", "blue", "big", "small",
    "runs", "walks", "jumps",
    "sees", "likes", "finds",
    "is", "on", "in", "with", "near",
    "what", "where", "who",
]
assert len(VOCAB) == VOCAB_SIZE
WORD_TO_ID: Dict[str, int] = {w: i for i, w in enumerate(VOCAB)}
ID_TO_WORD: Dict[int, str] = {i: w for i, w in enumerate(VOCAB)}

DETERMINERS = ["the", "a"]
AGENT_NOUNS = ["cat", "dog", "boy", "girl", "robot"]
PLACE_NOUNS = ["mat", "tree", "park"]
OBJECT_NOUNS = ["cat", "dog", "boy", "girl", "robot", "ball", "mat", "tree", "park"]
ADJECTIVES = ["red", "blue", "big", "small"]
INTRANSITIVE_VERBS = ["runs", "walks", "jumps"]
TRANSITIVE_VERBS = ["sees", "likes", "finds"]
PREPOSITIONS = ["on", "in", "with", "near"]
WHO_VERBS = ["sees", "likes", "finds"]


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
        description="CLLN dense 30->32 language trainer, soft cross entropy, ngspice"
    )
    p.add_argument("seed", type=int, nargs="?", default=0)
    p.add_argument("--epochs", type=int, default=20)

    # Learning hyperparams
    p.add_argument("--gamma", type=float, default=0.30)
    p.add_argument("--delta", type=float, default=0.30)
    p.add_argument("--softmax-temp", type=float, default=1.0)

    # Bit voltage mapping
    p.add_argument("--bit-v0", type=float, default=0.0, help="Voltage used for bit 0")
    p.add_argument("--bit-v1", type=float, default=1.0, help="Voltage used for bit 1")

    # Rails
    p.add_argument("--vminus", type=float, default=0.0)
    p.add_argument("--vplus", type=float, default=0.45)

    # Dataset generation
    p.add_argument("--num-windows-total", type=int, default=10000, help="Target total windows before train/val split")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument(
        "--template-mode",
        type=str,
        choices=["tiny", "balanced"],
        default="balanced",
        help="tiny = simpler, shorter grammar; balanced = richer sentence/question mix",
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

    p.add_argument("--sample-prompts", type=int, default=6, help="How many sample generations to save each epoch")
    p.add_argument("--sample-max-len", type=int, default=18, help="Max generated tokens after the seed context")
    p.add_argument("--final-val", action="store_true")
    return p.parse_args()


# -------------------------
# Synthetic language data
# -------------------------
def sample_np(subject: bool = False, allow_adj: bool = True) -> List[str]:
    det = random.choice(DETERMINERS)
    noun_pool = AGENT_NOUNS if subject else OBJECT_NOUNS
    noun = random.choice(noun_pool)
    if allow_adj and random.random() < 0.45:
        return [det, random.choice(ADJECTIVES), noun]
    return [det, noun]


def sample_place_np() -> List[str]:
    det = random.choice(DETERMINERS)
    noun = random.choice(PLACE_NOUNS)
    if random.random() < 0.35:
        return [det, random.choice(ADJECTIVES), noun]
    return [det, noun]


def sample_statement() -> List[str]:
    subj = sample_np(subject=True, allow_adj=True)
    template = random.choice(["intransitive", "attribute", "transitive", "locative"])

    if template == "intransitive":
        return subj + [random.choice(INTRANSITIVE_VERBS), "."]
    if template == "attribute":
        return subj + ["is", random.choice(ADJECTIVES), "."]
    if template == "transitive":
        return subj + [random.choice(TRANSITIVE_VERBS)] + sample_np(subject=False, allow_adj=True) + ["."]

    prep = random.choice(PREPOSITIONS)
    target = sample_place_np() if prep in {"on", "in", "near"} else sample_np(subject=False, allow_adj=True)
    return subj + ["is", prep] + target + ["."]
        

def sample_question() -> List[str]:
    template = random.choice(["where_is", "what_is_on", "who_verb"])
    if template == "where_is":
        return ["where", "is"] + sample_np(subject=True, allow_adj=True) + ["?"]
    if template == "what_is_on":
        return ["what", "is", "on"] + sample_place_np() + ["?"]
    verb = random.choice(WHO_VERBS)
    return ["who", verb] + sample_np(subject=False, allow_adj=True) + ["?"]


def sample_sentence(template_mode: str) -> List[str]:
    if template_mode == "tiny":
        r = random.random()
        if r < 0.50:
            subj = sample_np(subject=True, allow_adj=False)
            if random.random() < 0.5:
                return subj + [random.choice(INTRANSITIVE_VERBS), "."]
            return subj + ["is", random.choice(["red", "blue"]), "."]
        if r < 0.80:
            return ["where", "is"] + sample_np(subject=True, allow_adj=False) + ["?"]
        return ["who", random.choice(["sees", "likes"])] + sample_np(subject=False, allow_adj=False) + ["?"]

    if random.random() < 0.72:
        return sample_statement()
    return sample_question()


def token_id_to_bits(tok_id: int) -> np.ndarray:
    return np.array([(tok_id >> shift) & 1 for shift in range(TOKEN_BITS - 1, -1, -1)], dtype=float)


def encode_context_tokens(ctx_ids: Sequence[int], bit_v0: float, bit_v1: float) -> np.ndarray:
    if len(ctx_ids) != CONTEXT_LEN:
        raise ValueError(f"Expected context length {CONTEXT_LEN}, got {len(ctx_ids)}")
    bits: List[float] = []
    for tid in ctx_ids:
        code = token_id_to_bits(int(tid))
        volts = np.where(code > 0.5, bit_v1, bit_v0)
        bits.extend(volts.tolist())
    return np.asarray(bits, dtype=float)


def build_windows_from_sentence(
    tokens: Sequence[str],
    bit_v0: float,
    bit_v1: float,
) -> Tuple[List[np.ndarray], List[int], List[Tuple[int, ...]]]:
    bos_id = WORD_TO_ID["<BOS>"]
    ids = [WORD_TO_ID[t] for t in tokens]
    padded = [bos_id] * CONTEXT_LEN + ids

    xs: List[np.ndarray] = []
    ys: List[int] = []
    ctx_keys: List[Tuple[int, ...]] = []

    for i in range(CONTEXT_LEN, len(padded)):
        ctx = padded[i - CONTEXT_LEN:i]
        y = padded[i]
        xs.append(encode_context_tokens(ctx, bit_v0=bit_v0, bit_v1=bit_v1))
        ys.append(int(y))
        ctx_keys.append(tuple(int(t) for t in ctx))
    return xs, ys, ctx_keys


def build_language_dataset_until_windows(
    target_windows_total: int,
    template_mode: str,
    bit_v0: float,
    bit_v1: float,
) -> Tuple[List[np.ndarray], List[int], List[Tuple[int, ...]], List[List[str]]]:
    all_x: List[np.ndarray] = []
    all_y: List[int] = []
    all_ctx_keys: List[Tuple[int, ...]] = []
    sentences: List[List[str]] = []

    while len(all_x) < int(target_windows_total):
        sent = sample_sentence(template_mode)
        x_s, y_s, ctx_s = build_windows_from_sentence(sent, bit_v0=bit_v0, bit_v1=bit_v1)
        all_x.extend(x_s)
        all_y.extend(y_s)
        all_ctx_keys.extend(ctx_s)
        sentences.append(sent)

    # Trim to exact total windows for reproducibility/speed.
    all_x = all_x[:target_windows_total]
    all_y = all_y[:target_windows_total]
    all_ctx_keys = all_ctx_keys[:target_windows_total]
    return all_x, all_y, all_ctx_keys, sentences


# -------------------------
# Target-distribution helpers
# -------------------------
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


def top_words_from_q(q: np.ndarray, topk: int = 5) -> str:
    q = np.asarray(q, dtype=float)
    idx = np.argsort(-q)[:topk]
    parts: List[str] = []
    for i in idx:
        if q[i] <= 0.0:
            continue
        parts.append(f"{ID_TO_WORD[int(i)]}:{float(q[i]):.2f}")
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
        "token_bits": TOKEN_BITS,
        "vocab_size": VOCAB_SIZE,
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
    lines.append(".title clln_dense_language32_softxent")
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

        Vout = np.asarray(data["out"], dtype=float)
        if not np.all(np.isfinite(Vout)):
            break
        yhat = pred_label(Vout)
        w = ID_TO_WORD[int(yhat)]
        out_words.append(w)
        ctx = ctx[1:] + [int(yhat)]
        if w in STOP_TOKENS:
            break
    return out_words


def save_plots(run_dir: Path):
    def _load(name: str):
        p = run_dir / name
        return np.load(p) if p.exists() else None

    tr_acc = _load("0_train_acc.npy")
    tr_support_acc = _load("0_train_support_acc.npy")
    tr_ce = _load("0_train_ce.npy")
    va_acc = _load("0_val_acc.npy")
    va_support_acc = _load("0_val_support_acc.npy")
    va_ce = _load("0_val_ce.npy")
    ep_total = _load("0_epoch_total_s.npy")
    ep_free = _load("0_epoch_free_s.npy")
    ep_clamp = _load("0_epoch_clamp_s.npy")
    ep_upd = _load("0_epoch_update_s.npy")

    if tr_acc is not None and va_acc is not None:
        plt.figure()
        plt.plot(np.arange(len(va_acc)), va_acc, label="val exact acc")
        plt.plot(np.arange(1, len(tr_acc) + 1), tr_acc, label="train exact acc")
        if tr_support_acc is not None and va_support_acc is not None:
            plt.plot(np.arange(len(va_support_acc)), va_support_acc, label="val support acc")
            plt.plot(np.arange(1, len(tr_support_acc) + 1), tr_support_acc, label="train support acc")
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
        plt.title("Soft cross entropy vs epoch")
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
    gamma = float(args.gamma)
    delta = float(args.delta)
    temp = float(args.softmax_temp)
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
    num_windows_total = int(args.num_windows_total)
    if num_windows_total < 100:
        raise ValueError("--num-windows-total must be >= 100")

    all_x, all_y, all_ctx_keys, sentences = build_language_dataset_until_windows(
        target_windows_total=num_windows_total,
        template_mode=template_mode,
        bit_v0=bit_v0,
        bit_v1=bit_v1,
    )

    X_train, X_val, y_train, y_val, ctx_train, ctx_val = train_test_split(
        all_x,
        all_y,
        all_ctx_keys,
        test_size=float(args.val_frac),
        random_state=seed,
        stratify=np.asarray(all_y, dtype=int),
    )

    train_x = [np.asarray(v, dtype=float) for v in X_train]
    train_y = [int(v) for v in y_train]
    train_ctx = [tuple(c) for c in ctx_train]

    val_x = [np.asarray(v, dtype=float) for v in X_val]
    val_y = [int(v) for v in y_val]
    val_ctx = [tuple(c) for c in ctx_val]
    val_n = len(val_x)

    topo = make_dense_io_topology()
    if topo.Nin != INPUT_DIM or topo.K != OUTPUT_DIM:
        raise RuntimeError("Internal topology dimensions do not match language task")

    ctx_q_train = build_context_target_distributions(train_ctx, train_y, topo.K)
    unigram_q_train = build_unigram_target_distribution(train_y, topo.K)

    if vg_init_mode == "fixed":
        vg_unique = np.full((topo.num_edges,), vg_init_single, dtype=float)
    else:
        if vg_init_hi <= vg_init_lo:
            raise ValueError("--vg-init-hi must be > --vg-init-lo for random init")
        vg_unique = np.random.uniform(vg_init_lo, vg_init_hi, size=(topo.num_edges,)).astype(float)

    results_dir = Path(__file__).resolve().parent / "results_language_32_softce"
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
        f"seed={seed} gamma={gamma} delta={delta} T={temp} bit=[{bit_v0},{bit_v1}] "
        f"context={CONTEXT_LEN} bits/token={TOKEN_BITS} Nin={topo.Nin} K={topo.K} "
        f"total_windows={num_windows_total} template={template_mode} rails=[{vminus_val},{vplus_val}] "
        f"solver={solver} body_tie={body_tie} rs_clamp={RS_CLAMP} "
        f"vg_init={vg_init_mode} epochs={epochs} device_include={device_lib} subckt={DEVICE_SUBCKT}"
    )

    print("=== RUN START (clln_dense_language32_softxent) ===", flush=True)
    print(cfg_str, flush=True)
    print(f"train_windows={len(train_x)} val_windows={len(val_x)} edges={topo.num_edges}", flush=True)
    print(f"sentences_generated={len(sentences)} unique_train_contexts={len(ctx_q_train)}", flush=True)

    (run_dir / "vocab.json").write_text(json.dumps({"vocab": VOCAB, "word_to_id": WORD_TO_ID}, indent=2))
    (run_dir / "sample_sentences.txt").write_text("\n".join(" ".join(s) for s in sentences[:300]))

    preview_items = []
    for ctx, q in list(ctx_q_train.items())[:100]:
        words = [ID_TO_WORD[t] for t in ctx]
        preview_items.append({
            "context": words,
            "target_dist": {ID_TO_WORD[j]: float(q[j]) for j in np.where(q > 0)[0].tolist()},
        })
    (run_dir / "context_target_preview.json").write_text(json.dumps(preview_items, indent=2))

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
            "name": "synthetic_language_32word",
            "target_windows_total": num_windows_total,
            "template_mode": template_mode,
            "context_len": CONTEXT_LEN,
            "token_bits": TOKEN_BITS,
            "bit_v0": bit_v0,
            "bit_v1": bit_v1,
            "vocab": VOCAB,
        },
        "train_count": len(train_x),
        "val_count": len(val_x),
        "unique_train_contexts": len(ctx_q_train),
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
        "loss": "soft_cross_entropy",
        "generation": {
            "sample_prompts": int(args.sample_prompts),
            "sample_max_len": int(args.sample_max_len),
            "stop_tokens": sorted(list(STOP_TOKENS)),
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
    eD = topo.edges_D
    eS = topo.edges_S

    try:
        G = nx.DiGraph()
        G.add_nodes_from(nodes_list.tolist())
        for d, s in zip(eD.tolist(), eS.tolist()):
            G.add_edge(d, s)
        nx.write_graphml(G, str(run_dir / "0.graphml"))
    except Exception:
        pass

    val_acc_hist: List[float] = []
    val_support_acc_hist: List[float] = []
    val_ce_hist: List[float] = []
    tr_acc_hist: List[float] = []
    tr_support_acc_hist: List[float] = []
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
        mk_free_all(ng, topo.K)
        correct = 0
        support_correct = 0
        total = 0
        loss_sum = 0.0
        count = 0
        reloads = 0
        nonfinite = 0
        qmass_list: List[float] = []
        unseen_ctx = 0
        confusion = np.zeros((topo.K, topo.K), dtype=int)
        vout_val = np.full((val_n, topo.K), np.nan, dtype=float) if val_n > 0 else np.zeros((0, topo.K), dtype=float)

        for i, (xv, yv, ctxv) in enumerate(zip(val_x, val_y, val_ctx)):
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
            if ctxv in ctx_q_train:
                q_target = ctx_q_train[ctxv]
            else:
                unseen_ctx += 1
                q_target = unigram_q_train

            correct += int(pred == ytrue)
            support_correct += int(q_target[pred] > 0.0)
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
            "val_support_acc": float(support_correct / total) if total else float("nan"),
            "val_unseen_contexts": float(unseen_ctx),
        }
        acc = (correct / total) if total else float("nan")
        loss = (loss_sum / count) if count else float("nan")
        return float(acc), float(loss), diag

    def save_generation_samples(epoch: int):
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
            target = ID_TO_WORD[int(val_y[idx])]
            q_target = ctx_q_train.get(val_ctx[idx], unigram_q_train)
            seed_clean = [w for w in seed_words if w != "<BOS>"]
            gen_clean = [w for w in generated if w != "<BOS>"]
            lines.append(f"[{j}] seed={' '.join(seed_clean) if seed_clean else '<BOS>'}")
            lines.append(f"    observed_next={target}")
            lines.append(f"    target_dist_top={top_words_from_q(q_target, topk=5)}")
            lines.append(f"    generated={' '.join(gen_clean) if gen_clean else '<empty>'}")
        (run_dir / f"samples_epoch{epoch}.txt").write_text("\n".join(lines) + "\n")

    v0, ce0, diag0 = eval_free_metrics(epoch=0)
    val_acc_hist.append(v0)
    val_support_acc_hist.append(float(diag0.get("val_support_acc", float("nan"))))
    val_ce_hist.append(ce0)
    np.save(run_dir / "0_val_acc.npy", np.asarray(val_acc_hist, dtype=float))
    np.save(run_dir / "0_val_support_acc.npy", np.asarray(val_support_acc_hist, dtype=float))
    np.save(run_dir / "0_val_ce.npy", np.asarray(val_ce_hist, dtype=float))
    (run_dir / "0_diag_epoch0.json").write_text(json.dumps(diag0, indent=2))
    save_generation_samples(epoch=0)
    print(
        f"[epoch 0] {cfg_str} | VAL exact_acc={v0:.4f} support_acc={diag0.get('val_support_acc', float('nan')):.4f} "
        f"softCE={ce0:.6f} qmass_mean={diag0.get('val_qmass_mean', float('nan')):.4f} unseen_ctx={int(diag0.get('val_unseen_contexts', 0.0))}",
        flush=True,
    )

    for ep in range(1, epochs + 1):
        t_ep0 = time.time()
        order = np.arange(len(train_x))
        np.random.shuffle(order)

        train_correct = 0
        train_support_correct = 0
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
            ctx_key = train_ctx[idx]
            q_target = ctx_q_train[ctx_key]
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
            train_support_correct += int(q_target[pred] > 0.0)
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
        tr_support_acc = (train_support_correct / train_total) if train_total else float("nan")
        tr_ce = (ce_sum / ce_count) if ce_count else float("nan")
        tr_qmass_mean = (qmass_sum / qmass_count) if qmass_count else float("nan")

        tr_acc_hist.append(float(tr_acc))
        tr_support_acc_hist.append(float(tr_support_acc))
        tr_ce_hist.append(float(tr_ce))
        reload_free_hist.append(int(reload_free))
        reload_clamp_hist.append(int(reload_clamp))
        nonfinite_free_hist.append(int(nonfinite_free))
        nonfinite_clamp_hist.append(int(nonfinite_clamp))

        v_acc, v_ce, diag = eval_free_metrics(epoch=ep)
        val_acc_hist.append(float(v_acc))
        val_support_acc_hist.append(float(diag.get("val_support_acc", float("nan"))))
        val_ce_hist.append(float(v_ce))
        save_generation_samples(epoch=ep)

        ep_total = float(time.time() - t_ep0)
        ep_total_s.append(ep_total)
        ep_free_s.append(float(t_free))
        ep_clamp_s.append(float(t_clamp))
        ep_update_s.append(float(t_update))

        np.save(run_dir / "0_train_acc.npy", np.asarray(tr_acc_hist, dtype=float))
        np.save(run_dir / "0_train_support_acc.npy", np.asarray(tr_support_acc_hist, dtype=float))
        np.save(run_dir / "0_train_ce.npy", np.asarray(tr_ce_hist, dtype=float))
        np.save(run_dir / "0_val_acc.npy", np.asarray(val_acc_hist, dtype=float))
        np.save(run_dir / "0_val_support_acc.npy", np.asarray(val_support_acc_hist, dtype=float))
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
                "loss": "soft_cross_entropy",
                "template_mode": template_mode,
            },
            "train": {
                "exact_acc": float(tr_acc),
                "support_acc": float(tr_support_acc),
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
                "support_acc": float(diag.get("val_support_acc", float("nan"))),
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
            f"[epoch {ep}/{epochs}] {cfg_str} | TRAIN exact_acc={tr_acc:.4f} support_acc={tr_support_acc:.4f} "
            f"softCE={tr_ce:.6f} qmass_mean={tr_qmass_mean:.4f} free={n_free} clamp={n_clamp} "
            f"reloadF={reload_free} reloadC={reload_clamp} nonfiniteF={nonfinite_free} nonfiniteC={nonfinite_clamp}",
            flush=True,
        )
        print(
            f"[epoch {ep}/{epochs}] {cfg_str} | VAL exact_acc={v_acc:.4f} support_acc={diag.get('val_support_acc', float('nan')):.4f} "
            f"softCE={v_ce:.6f} qmass_mean={diag.get('val_qmass_mean', float('nan')):.4f} unseen_ctx={int(diag.get('val_unseen_contexts', 0.0))} | "
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

    print("=== RUN END (clln_dense_language32_softxent) ===", flush=True)
    try:
        log_f.flush()
        log_f.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
