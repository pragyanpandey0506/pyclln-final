#!/usr/bin/env python3
"""
Linear-MLP baseline for the 32-token embed4 one-hot language task.

Model:
  z = W x + b   (24 -> 32, no hidden nonlinearity)
  p = softmax(z / T)

Training loss:
  One-hot cross-entropy against the observed next-token label.

Evaluation metrics:
  - exact_acc: argmax prediction matches observed label
  - support_acc: argmax prediction lies in the empirical support of q(.|context),
    where q is built from the TRAIN split exactly like the 32-token soft-CE trainer
  - soft_ce / qmass_mean: measured against that train-split empirical q(.|context)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from clln_lang_trainer_embed4_onehot_ce import (
    INPUT_DIM,
    OUTPUT_DIM,
    INPUT_VOCAB,
    OUTPUT_VOCAB,
    build_sentence_corpus,
    build_windows_from_sentences,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Linear-MLP 24->32 baseline, one-hot CE with support-acc eval")
    p.add_argument("seed", type=int, nargs="?", default=0)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--softmax-temp", type=float, default=0.01)
    p.add_argument("--init-std", type=float, default=0.02)

    p.add_argument("--num-sentences", type=int, default=12000)
    p.add_argument("--min-target-count", type=int, default=80)
    p.add_argument("--max-sentence-words", type=int, default=9)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--max-train", type=int, default=0)
    p.add_argument("--max-val", type=int, default=0)
    p.add_argument("--template-mode", type=str, choices=["balanced", "broad"], default="broad")
    return p.parse_args()


def softmax_rows(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    s = np.sum(e, axis=1, keepdims=True)
    s = np.where(s <= 0.0, 1.0, s)
    return e / s


@dataclass
class EvalOut:
    exact_acc: float
    support_acc: float
    soft_ce: float
    qmass_mean: float
    unseen_contexts: float


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


def eval_metrics(
    X: np.ndarray,
    y: np.ndarray,
    ctx_list: Sequence[Tuple[int, ...]],
    W: np.ndarray,
    b: np.ndarray,
    q_map_train: Dict[Tuple[int, ...], np.ndarray],
    unigram_q_train: np.ndarray,
    temp: float,
) -> EvalOut:
    logits = X @ W.T + b[None, :]
    probs = softmax_rows(logits / float(temp))
    pred = np.argmax(logits, axis=1)

    q, unseen = build_q_matrix(ctx_list, q_map_train, unigram_q_train)
    p_safe = np.clip(probs, 1e-12, 1.0)
    ce = float(np.mean(-np.sum(q * np.log(p_safe), axis=1)))
    qmass = float(np.mean(np.sum(probs * (q > 0.0), axis=1)))

    exact_acc = float(np.mean(pred == y))
    support_acc = float(np.mean((q[np.arange(q.shape[0]), pred] > 0.0).astype(float)))
    return EvalOut(
        exact_acc=exact_acc,
        support_acc=support_acc,
        soft_ce=ce,
        qmass_mean=qmass,
        unseen_contexts=float(unseen),
    )


def main() -> None:
    args = parse_args()
    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)

    if args.softmax_temp <= 0.0:
        raise ValueError("--softmax-temp must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")
    if args.epochs <= 0:
        raise ValueError("--epochs must be > 0")

    sentences, total_target_coverage = build_sentence_corpus(
        num_sentences=int(args.num_sentences),
        template_mode=str(args.template_mode),
        max_sentence_words=int(args.max_sentence_words),
        min_target_count=int(args.min_target_count),
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

    Xtr = np.asarray(X_train, dtype=float)
    ytr = np.asarray(y_train, dtype=int)
    ctr = [tuple(c) for c in ctx_train]
    Xva = np.asarray(X_val, dtype=float)
    yva = np.asarray(y_val, dtype=int)
    cva = [tuple(c) for c in ctx_val]

    if Xtr.shape[1] != INPUT_DIM:
        raise RuntimeError(f"expected {INPUT_DIM} input dims, got {Xtr.shape[1]}")

    q_map_train = build_context_target_distributions(ctr, ytr.tolist(), OUTPUT_DIM)
    unigram_q_train = build_unigram_target_distribution(ytr.tolist(), OUTPUT_DIM)

    rng = np.random.default_rng(seed)
    W = rng.normal(0.0, float(args.init_std), size=(OUTPUT_DIM, INPUT_DIM)).astype(float)
    b = np.zeros((OUTPUT_DIM,), dtype=float)

    results_dir = Path(__file__).resolve().parent / "results_language_32_embed4_linear_onehotce"
    env_run_dir = os.environ.get("RUN_DIR")
    if env_run_dir:
        run_dir = Path(env_run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        runs_dir = results_dir / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)
        run_id = datetime.now().strftime("%Y%m%d-%H%M%S-%f") + f"_seed-{seed}"
        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "script": str(Path(__file__).resolve()),
        "argv": list(__import__("sys").argv),
        "seed": seed,
        "timestamp": datetime.now().isoformat(),
        "model": "linear_softmax_24_to_32_onehot",
        "dataset": {
            "name": "synthetic_language_32token_embed4",
            "num_sentences_requested": int(args.num_sentences),
            "num_sentences_actual": len(sentences),
            "min_target_count": int(args.min_target_count),
            "max_sentence_words": int(args.max_sentence_words),
            "template_mode": str(args.template_mode),
            "input_vocab": INPUT_VOCAB,
            "output_vocab": OUTPUT_VOCAB,
            "target_coverage_total": total_target_coverage,
            "target_coverage_train": {k: int(train_target_counter.get(k, 0)) for k in OUTPUT_VOCAB},
            "target_coverage_val": {k: int(val_target_counter.get(k, 0)) for k in OUTPUT_VOCAB},
        },
        "train_sentence_count": int(len(train_sentences)),
        "val_sentence_count": int(len(val_sentences)),
        "train_count": int(len(Xtr)),
        "val_count": int(len(Xva)),
        "hparams": {
            "epochs": int(args.epochs),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "batch_size": int(args.batch_size),
            "softmax_temp": float(args.softmax_temp),
            "init_std": float(args.init_std),
        },
    }
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    (run_dir / "sample_sentences.txt").write_text("\n".join(" ".join(s) for s in sentences[:300]))

    tr_exact_hist: List[float] = []
    tr_support_hist: List[float] = []
    tr_ce_hist: List[float] = []
    tr_qmass_hist: List[float] = []
    va_exact_hist: List[float] = []
    va_support_hist: List[float] = []
    va_ce_hist: List[float] = []
    va_qmass_hist: List[float] = []
    va_unseen_hist: List[float] = []

    bs = int(args.batch_size)
    lr = float(args.lr)
    wd = float(args.weight_decay)
    temp = float(args.softmax_temp)

    t0 = time.time()
    for ep in range(1, int(args.epochs) + 1):
        order = rng.permutation(len(Xtr))
        for s in range(0, len(order), bs):
            idx = order[s : s + bs]
            xb = Xtr[idx]
            yb = ytr[idx]

            logits = xb @ W.T + b[None, :]
            probs = softmax_rows(logits / temp)
            qb = np.zeros_like(probs)
            qb[np.arange(len(idx)), yb] = 1.0
            dlogits = (probs - qb) / (float(len(idx)) * temp)

            gW = dlogits.T @ xb
            if wd > 0.0:
                gW = gW + wd * W
            gb = np.sum(dlogits, axis=0)

            W -= lr * gW
            b -= lr * gb

        tr = eval_metrics(Xtr, ytr, ctr, W, b, q_map_train, unigram_q_train, temp=temp)
        va = eval_metrics(Xva, yva, cva, W, b, q_map_train, unigram_q_train, temp=temp)

        tr_exact_hist.append(tr.exact_acc)
        tr_support_hist.append(tr.support_acc)
        tr_ce_hist.append(tr.soft_ce)
        tr_qmass_hist.append(tr.qmass_mean)
        va_exact_hist.append(va.exact_acc)
        va_support_hist.append(va.support_acc)
        va_ce_hist.append(va.soft_ce)
        va_qmass_hist.append(va.qmass_mean)
        va_unseen_hist.append(va.unseen_contexts)

        print(
            f"[epoch {ep}/{args.epochs}] lr={lr} wd={wd} T={temp} "
            f"| TRAIN exact={tr.exact_acc:.4f} support={tr.support_acc:.4f} ce={tr.soft_ce:.4f} qmass={tr.qmass_mean:.4f} "
            f"| VAL exact={va.exact_acc:.4f} support={va.support_acc:.4f} ce={va.soft_ce:.4f} qmass={va.qmass_mean:.4f} unseen_ctx={int(va.unseen_contexts)}",
            flush=True,
        )

        np.save(run_dir / "0_train_acc.npy", np.asarray(tr_exact_hist, dtype=float))
        np.save(run_dir / "0_train_support_acc.npy", np.asarray(tr_support_hist, dtype=float))
        np.save(run_dir / "0_train_ce.npy", np.asarray(tr_ce_hist, dtype=float))
        np.save(run_dir / "0_train_qmass.npy", np.asarray(tr_qmass_hist, dtype=float))
        np.save(run_dir / "0_val_acc.npy", np.asarray(va_exact_hist, dtype=float))
        np.save(run_dir / "0_val_support_acc.npy", np.asarray(va_support_hist, dtype=float))
        np.save(run_dir / "0_val_ce.npy", np.asarray(va_ce_hist, dtype=float))
        np.save(run_dir / "0_val_qmass.npy", np.asarray(va_qmass_hist, dtype=float))
        np.save(run_dir / "0_val_unseen_contexts.npy", np.asarray(va_unseen_hist, dtype=float))

        (run_dir / f"0_epoch_summary_epoch{ep}.json").write_text(
            json.dumps(
                {
                    "epoch": ep,
                    "train": tr.__dict__,
                    "val": va.__dict__,
                    "hparams": {
                        "lr": lr,
                        "weight_decay": wd,
                        "batch_size": bs,
                        "softmax_temp": temp,
                        "init_std": float(args.init_std),
                    },
                },
                indent=2,
            )
        )

    np.savez(run_dir / "linear_model_final.npz", W=W, b=b)

    best_idx = int(np.argmax(np.asarray(va_support_hist, dtype=float)))
    summary = {
        "run_dir": str(run_dir),
        "elapsed_s": float(time.time() - t0),
        "best_val_support_epoch": best_idx + 1,
        "best_val_support": float(va_support_hist[best_idx]),
        "best_val_exact": float(va_exact_hist[best_idx]),
        "best_val_ce": float(va_ce_hist[best_idx]),
        "best_val_qmass": float(va_qmass_hist[best_idx]),
        "best_val_unseen_contexts": float(va_unseen_hist[best_idx]),
        "final_val_support": float(va_support_hist[-1]),
        "final_val_exact": float(va_exact_hist[-1]),
        "final_val_ce": float(va_ce_hist[-1]),
        "final_val_qmass": float(va_qmass_hist[-1]),
        "final_val_unseen_contexts": float(va_unseen_hist[-1]),
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("=== LINEAR RUN END ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
