#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Sequence, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PySpice.Spice.NgSpice.Shared import NgSpiceShared
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
DEFAULT_SWEEP_DIR = ROOT / "results_language_16_onehotce" / "sweeps" / "top60_from_softce_20260325-122210"
DEFAULT_TRAINER_SCRIPT = ROOT / "clln_lang_trainer_onehot_ce.py"
DEFAULT_OUTPUT_DIR = ROOT / "analysis_onehot_best_support_val"


@dataclass
class SelectedCheckpoint:
    run_dir: Path
    epoch: int
    support_acc: float
    qmass_mean: float
    exact_acc: float
    soft_ce: float
    run_name: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyze the best 16-vocab one-hot CE checkpoint among support_acc=1.0 runs."
    )
    p.add_argument("--sweep-dir", type=Path, default=DEFAULT_SWEEP_DIR)
    p.add_argument("--trainer-script", type=Path, default=DEFAULT_TRAINER_SCRIPT)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--run-dir", type=Path, default=None, help="Override automatic checkpoint selection")
    p.add_argument("--epoch", type=int, default=None, help="Epoch for --run-dir")
    p.add_argument("--sample-temp", type=float, default=0.05, help="Sampling temperature for rollout diagnostics")
    p.add_argument("--num-rollouts", type=int, default=20)
    p.add_argument("--max-rollout-steps", type=int, default=8)
    return p.parse_args()


def load_module(script_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_flag(argv: Sequence[str], flag: str, cast, default):
    for i, tok in enumerate(argv):
        if tok == flag and i + 1 < len(argv):
            return cast(argv[i + 1])
        if tok.startswith(flag + "="):
            return cast(tok.split("=", 1)[1])
    return default


def softmax_from_logits(logits: np.ndarray, temp: float) -> np.ndarray:
    z = np.asarray(logits, dtype=float) / float(temp)
    z = z - np.max(z, axis=-1, keepdims=True)
    ez = np.exp(z)
    denom = np.sum(ez, axis=-1, keepdims=True)
    denom = np.clip(denom, 1e-12, None)
    return ez / denom


def entropy_bits(probs: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probs, dtype=float), 1e-12, 1.0)
    return -np.sum(p * (np.log(p) / np.log(2.0)), axis=-1)


def rank_of_target(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    order = np.argsort(-probs, axis=1)
    inv = np.empty_like(order)
    inv[np.arange(order.shape[0])[:, None], order] = np.arange(order.shape[1])[None, :]
    return inv[np.arange(order.shape[0]), y] + 1


def support_accuracy_from_probs(
    probs: np.ndarray,
    contexts: Sequence[Tuple[int, ...]],
    support_map: Mapping[Tuple[int, ...], Set[int]],
) -> float:
    pred = np.argmax(probs, axis=1)
    hits = []
    for ctx, p in zip(contexts, pred.tolist()):
        valid = support_map.get(tuple(ctx), set())
        hits.append(float(int(p) in valid))
    return float(np.mean(hits)) if hits else float("nan")


def topk_mass(probs: np.ndarray, k: int) -> np.ndarray:
    idx = np.argsort(-probs, axis=1)[:, :k]
    vals = np.take_along_axis(probs, idx, axis=1)
    return np.sum(vals, axis=1)


def find_best_support_checkpoint(sweep_dir: Path) -> Tuple[SelectedCheckpoint, List[Dict[str, object]]]:
    csv_path = sweep_dir / "support_acc_vs_epoch_all_runs.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing sweep CSV: {csv_path}")

    rows = list(csv.DictReader(csv_path.open()))
    ranked_rows: List[Dict[str, object]] = []
    for row in rows:
        support_acc = float(row["support_acc"])
        if abs(support_acc - 1.0) >= 1e-12:
            continue
        run_name = str(row["run"])
        epoch = int(row["epoch"])
        run_dir = sweep_dir / run_name
        summary = json.loads((run_dir / f"0_epoch_summary_epoch{epoch}.json").read_text())
        ranked_rows.append(
            {
                "run": run_name,
                "epoch": epoch,
                "support_acc": support_acc,
                "val_qmass_mean": float(summary["val"]["val_qmass_mean"]),
                "val_exact_acc": float(summary["val"]["exact_acc"]),
                "val_soft_ce": float(summary["val"]["soft_ce"]),
            }
        )

    if not ranked_rows:
        raise RuntimeError("No rows with support_acc = 1.0 were found in the sweep CSV")

    ranked_rows.sort(
        key=lambda r: (
            float(r["val_qmass_mean"]),
            float(r["val_exact_acc"]),
            -int(r["epoch"]),
        ),
        reverse=True,
    )
    best = ranked_rows[0]
    return (
        SelectedCheckpoint(
            run_dir=sweep_dir / str(best["run"]),
            epoch=int(best["epoch"]),
            support_acc=float(best["support_acc"]),
            qmass_mean=float(best["val_qmass_mean"]),
            exact_acc=float(best["val_exact_acc"]),
            soft_ce=float(best["val_soft_ce"]),
            run_name=str(best["run"]),
        ),
        ranked_rows,
    )


def reconstruct_dataset(trainer, meta: Mapping[str, object]) -> Dict[str, object]:
    seed = int(meta["seed"])
    argv = [str(v) for v in meta.get("argv", [])]
    val_frac = float(parse_flag(argv, "--val-frac", float, 0.2))
    max_train = int(parse_flag(argv, "--max-train", int, 0))
    max_val = int(parse_flag(argv, "--max-val", int, 0))
    dataset = meta["dataset"]

    random.seed(seed)
    np.random.seed(seed)
    all_x, all_y, all_ctx, sentences = trainer.build_language_dataset(
        num_sentences=int(dataset["num_sentences"]),
        template_mode=str(dataset["template_mode"]),
        bit_v0=float(dataset["bit_v0"]),
        bit_v1=float(dataset["bit_v1"]),
        max_sentence_words=int(dataset["max_sentence_words"]),
    )

    X_train, X_val, y_train, y_val, ctx_train, ctx_val = train_test_split(
        all_x,
        all_y,
        all_ctx,
        test_size=val_frac,
        random_state=seed,
        stratify=np.asarray(all_y, dtype=int),
    )

    if max_train > 0:
        X_train = X_train[:max_train]
        y_train = y_train[:max_train]
        ctx_train = ctx_train[:max_train]
    if max_val > 0:
        X_val = X_val[:max_val]
        y_val = y_val[:max_val]
        ctx_val = ctx_val[:max_val]

    return {
        "all_x": [np.asarray(v, dtype=float) for v in all_x],
        "all_y": [int(v) for v in all_y],
        "all_ctx": [tuple(int(t) for t in ctx) for ctx in all_ctx],
        "sentences": sentences,
        "train_x": [np.asarray(v, dtype=float) for v in X_train],
        "train_y": [int(v) for v in y_train],
        "train_ctx": [tuple(int(t) for t in ctx) for ctx in ctx_train],
        "val_x": [np.asarray(v, dtype=float) for v in X_val],
        "val_y": [int(v) for v in y_val],
        "val_ctx": [tuple(int(t) for t in ctx) for ctx in ctx_val],
    }


def build_support_map(trainer, bit_v0: float, bit_v1: float) -> Dict[Tuple[int, ...], Set[int]]:
    support_map: Dict[Tuple[int, ...], Set[int]] = {}
    for det_s in trainer.DETERMINERS:
        for subj in trainer.AGENT_NOUNS:
            subj_np = [det_s, subj]

            sentence_variants = [
                subj_np + ["runs"],
                subj_np + ["is", "red"],
            ]
            for sent in sentence_variants:
                _, ys, ctxs = trainer.build_windows_from_sentence(sent, bit_v0=bit_v0, bit_v1=bit_v1)
                for ctx, y in zip(ctxs, ys):
                    support_map.setdefault(tuple(int(v) for v in ctx), set()).add(int(y))

            for verb in trainer.TRANSITIVE_VERBS:
                for det_o in trainer.DETERMINERS:
                    for obj in trainer.OBJECT_NOUNS:
                        sent = subj_np + [verb, det_o, obj]
                        _, ys, ctxs = trainer.build_windows_from_sentence(sent, bit_v0=bit_v0, bit_v1=bit_v1)
                        for ctx, y in zip(ctxs, ys):
                            support_map.setdefault(tuple(int(v) for v in ctx), set()).add(int(y))

            for det_o in trainer.DETERMINERS:
                for obj in trainer.OBJECT_NOUNS:
                    sent = subj_np + ["is", "on", det_o, obj]
                    _, ys, ctxs = trainer.build_windows_from_sentence(sent, bit_v0=bit_v0, bit_v1=bit_v1)
                    for ctx, y in zip(ctxs, ys):
                        support_map.setdefault(tuple(int(v) for v in ctx), set()).add(int(y))
    return support_map


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def ctx_to_words(ctx: Sequence[int], id_to_word: Mapping[int, str]) -> List[str]:
    return [str(id_to_word[int(t)]) for t in ctx]


class FreePhaseRunner:
    def __init__(self, trainer, meta: Mapping[str, object], vg_unique: np.ndarray):
        self.trainer = trainer
        self.meta = meta
        self.vg_unique = np.asarray(vg_unique, dtype=float).copy()
        self.dataset = meta["dataset"]
        self.topo = trainer.make_dense_io_topology()
        self.netlist = trainer.mk_netlist(
            topo=self.topo,
            vg_unique=self.vg_unique,
            vminus_val=float(meta["rails"]["vminus"]),
            vplus_val=float(meta["rails"]["vplus"]),
            solver=str(meta["solver"]),
            body_res=float(meta["body_res"]),
            body_tie=str(meta["body_tie"]),
            device_lib_path=str(meta["device"]["include_path"]),
        )
        self.ng = NgSpiceShared(send_data=False)
        self.ng.load_circuit(self.netlist)
        self.trainer.restore_gate_voltages(self.ng, self.vg_unique)
        self.cache: Dict[Tuple[int, ...], np.ndarray] = {}

    def close(self) -> None:
        try:
            self.ng.remove_circuit()
        except Exception:
            pass

    def _predict_logits_uncached(self, ctx: Tuple[int, ...]) -> np.ndarray:
        xin = self.trainer.encode_context_tokens(
            list(ctx),
            bit_v0=float(self.dataset["bit_v0"]),
            bit_v1=float(self.dataset["bit_v1"]),
        )
        self.trainer.mk_free_all(self.ng, self.topo.K)
        self.trainer.alter_inputs_named(self.ng, xin)
        ok, _, data, err = self.trainer.run_and_read(self.ng, {"out": self.topo.out_nodes.tolist()})
        if (not ok) or data is None:
            try:
                self.ng.remove_circuit()
            except Exception:
                pass
            self.ng.load_circuit(self.netlist)
            self.trainer.restore_gate_voltages(self.ng, self.vg_unique)
            self.trainer.mk_free_all(self.ng, self.topo.K)
            self.trainer.alter_inputs_named(self.ng, xin)
            ok, _, data, err = self.trainer.run_and_read(self.ng, {"out": self.topo.out_nodes.tolist()})
        if (not ok) or data is None:
            raise RuntimeError(f"Free-phase inference failed for context {ctx}: {err}")
        logits = np.asarray(data["out"], dtype=float)
        if not np.all(np.isfinite(logits)):
            raise RuntimeError(f"Non-finite logits for context {ctx}")
        return logits

    def logits(self, ctx: Tuple[int, ...]) -> np.ndarray:
        if ctx not in self.cache:
            self.cache[ctx] = self._predict_logits_uncached(ctx)
        return self.cache[ctx].copy()

    def probs(self, ctx: Tuple[int, ...], temp: float) -> np.ndarray:
        return softmax_from_logits(self.logits(ctx)[None, :], temp=temp)[0]


def build_context_distributions(
    contexts: Sequence[Tuple[int, ...]],
    targets: Sequence[int],
    vocab_size: int,
) -> Tuple[Dict[Tuple[int, ...], np.ndarray], Dict[Tuple[int, ...], int]]:
    counts: Dict[Tuple[int, ...], np.ndarray] = {}
    total: Dict[Tuple[int, ...], int] = Counter()
    for ctx, y in zip(contexts, targets):
        if ctx not in counts:
            counts[ctx] = np.zeros(vocab_size, dtype=float)
        counts[ctx][int(y)] += 1.0
        total[ctx] += 1
    dist = {ctx: c / max(1.0, float(np.sum(c))) for ctx, c in counts.items()}
    return dist, total


def build_bigram_distributions(
    contexts: Sequence[Tuple[int, ...]],
    targets: Sequence[int],
    probs_by_ctx: Mapping[Tuple[int, ...], np.ndarray],
    vocab_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    data_counts = np.zeros((vocab_size, vocab_size), dtype=float)
    model_counts = np.zeros((vocab_size, vocab_size), dtype=float)
    row_counts = np.zeros(vocab_size, dtype=float)
    for ctx, y in zip(contexts, targets):
        last_tok = int(ctx[-1])
        data_counts[last_tok, int(y)] += 1.0
        model_counts[last_tok, :] += probs_by_ctx[ctx]
        row_counts[last_tok] += 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        data_bigram = data_counts / np.clip(np.sum(data_counts, axis=1, keepdims=True), 1e-12, None)
        model_bigram = model_counts / np.clip(row_counts[:, None], 1e-12, None)
    return data_bigram, model_bigram


def build_bigram_validity(
    support_map: Mapping[Tuple[int, ...], Set[int]],
    vocab_size: int,
) -> np.ndarray:
    valid = np.zeros((vocab_size, vocab_size), dtype=bool)
    for ctx, ys in support_map.items():
        last_tok = int(ctx[-1])
        for y in ys:
            valid[last_tok, int(y)] = True
    return valid


def select_representative_contexts(
    support_map: Mapping[Tuple[int, ...], Set[int]],
    word_to_id: Mapping[str, int],
) -> List[Tuple[int, ...]]:
    bos = int(word_to_id["<BOS>"])
    candidates = [
        (bos, bos, bos, bos, bos, bos),
        (bos, bos, bos, bos, bos, int(word_to_id["the"])),
        (bos, bos, bos, bos, bos, int(word_to_id["a"])),
        (bos, bos, bos, bos, int(word_to_id["the"]), int(word_to_id["boy"])),
        (bos, bos, bos, bos, int(word_to_id["the"]), int(word_to_id["cat"])),
        (bos, bos, bos, int(word_to_id["the"]), int(word_to_id["boy"]), int(word_to_id["is"])),
        (bos, bos, bos, int(word_to_id["the"]), int(word_to_id["boy"]), int(word_to_id["likes"])),
        (bos, bos, int(word_to_id["the"]), int(word_to_id["boy"]), int(word_to_id["is"]), int(word_to_id["on"])),
    ]
    return [ctx for ctx in candidates if ctx in support_map]


def plot_training_uncertainty(
    out_path: Path,
    run_dir: Path,
    selected_epoch: int,
    val_y: np.ndarray,
    val_ctx: Sequence[Tuple[int, ...]],
    support_map: Mapping[Tuple[int, ...], Set[int]],
    temp: float,
) -> Dict[str, object]:
    epoch_files = sorted(run_dir.glob("0_vout_val_epoch*.npy"), key=lambda p: int(p.stem.split("epoch")[-1]))
    epochs: List[int] = []
    exact_acc: List[float] = []
    support_accs: List[float] = []
    qmass: List[float] = []
    entropy_mean: List[float] = []
    mean_rank: List[float] = []
    perplexity: List[float] = []

    final_probs = None
    final_entropy = None
    final_rank = None
    final_conf = None
    final_correct = None

    for f in epoch_files:
        epoch = int(f.stem.split("epoch")[-1])
        logits = np.load(f)
        probs = softmax_from_logits(logits, temp=temp)
        pred = np.argmax(probs, axis=1)
        corr = (pred == val_y)
        rank = rank_of_target(probs, val_y)
        ent = entropy_bits(probs)
        ce = float(-np.mean(np.log(np.clip(probs[np.arange(probs.shape[0]), val_y], 1e-12, 1.0))))

        epochs.append(epoch)
        exact_acc.append(float(np.mean(corr)))
        support_accs.append(support_accuracy_from_probs(probs, val_ctx, support_map))
        qmass.append(float(np.mean(probs[np.arange(probs.shape[0]), val_y])))
        entropy_mean.append(float(np.mean(ent)))
        mean_rank.append(float(np.mean(rank)))
        perplexity.append(float(math.exp(ce)))

        if epoch == selected_epoch:
            final_probs = probs
            final_entropy = ent
            final_rank = rank
            final_conf = np.max(probs, axis=1)
            final_correct = corr.astype(float)

    if final_probs is None or final_entropy is None or final_rank is None or final_conf is None or final_correct is None:
        raise RuntimeError(f"Could not locate validation logits for selected epoch {selected_epoch}")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax = axes[0, 0]
    ax.plot(epochs, exact_acc, label="val exact acc", linewidth=2.0)
    ax.plot(epochs, support_accs, label="val support acc", linewidth=2.0)
    ax.plot(epochs, qmass, label="mean p(correct)", linewidth=2.0)
    ax.set_title("Accuracy and Support")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot(epochs, perplexity, label="val perplexity", linewidth=2.0, color="#d35400")
    ax.set_title("Perplexity")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("exp(soft CE)")
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    ax.plot(epochs, entropy_mean, label="mean entropy", linewidth=2.0, color="#8e44ad")
    ax.plot(epochs, mean_rank, label="mean correct rank", linewidth=2.0, color="#16a085")
    ax.set_title("Uncertainty Over Training")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Bits / Rank")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    bins = np.linspace(0.0, 1.0, 11)
    bin_ids = np.digitize(final_conf, bins[1:-1], right=False)
    xs = []
    ys = []
    for b in range(10):
        mask = bin_ids == b
        if np.any(mask):
            xs.append(float(np.mean(final_conf[mask])))
            ys.append(float(np.mean(final_correct[mask])))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#7f8c8d", linewidth=1.0)
    ax.plot(xs, ys, marker="o", linewidth=2.0, color="#2980b9")
    ax.set_title("Reliability At Selected Epoch")
    ax.set_xlabel("Top-1 confidence")
    ax.set_ylabel("Accuracy")
    ax.grid(alpha=0.25)

    fig.suptitle("Figure 1. Training and Uncertainty", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    return {
        "epochs": epochs,
        "exact_acc": exact_acc,
        "support_acc": support_accs,
        "qmass_mean": qmass,
        "entropy_mean": entropy_mean,
        "mean_rank": mean_rank,
        "perplexity": perplexity,
        "final_probs": final_probs,
        "final_entropy": final_entropy,
        "final_rank": final_rank,
        "final_conf": final_conf,
        "final_correct": final_correct,
    }


def plot_distributional_competence(
    out_path: Path,
    vocab: Sequence[str],
    data_bigram: np.ndarray,
    model_bigram: np.ndarray,
    valid_bigram: np.ndarray,
) -> Dict[str, float]:
    residual = model_bigram - data_bigram
    forbidden_leak = np.where(valid_bigram, np.nan, model_bigram)

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    mats = [
        (data_bigram, "Empirical bigram p_data(y | last token)", "viridis"),
        (model_bigram, "Model bigram p_model(y | last token)", "viridis"),
        (residual, "Residual p_model - p_data", "coolwarm"),
        (forbidden_leak, "Forbidden-transition leak", "magma"),
    ]
    for ax, (mat, title, cmap) in zip(axes.flatten(), mats):
        vmax = None
        vmin = None
        if title.startswith("Residual"):
            vmax = float(np.max(np.abs(residual)))
            vmin = -vmax
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("Next token")
        ax.set_ylabel("Last context token")
        ax.set_xticks(np.arange(len(vocab)))
        ax.set_yticks(np.arange(len(vocab)))
        ax.set_xticklabels(vocab, rotation=90, fontsize=7)
        ax.set_yticklabels(vocab, fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Figure 2. Distributional Competence", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    forbidden_mass = float(np.nansum(forbidden_leak))
    residual_l1 = float(np.sum(np.abs(residual)))
    return {"forbidden_bigram_mass": forbidden_mass, "bigram_residual_l1": residual_l1}


def plot_context_panels(
    out_path: Path,
    vocab: Sequence[str],
    id_to_word: Mapping[int, str],
    contexts: Sequence[Tuple[int, ...]],
    probs_by_ctx: Mapping[Tuple[int, ...], np.ndarray],
    data_dist_by_ctx: Mapping[Tuple[int, ...], np.ndarray],
    support_map: Mapping[Tuple[int, ...], Set[int]],
    val_probs: np.ndarray,
    val_y: np.ndarray,
) -> None:
    final_entropy = entropy_bits(val_probs)
    final_rank = rank_of_target(val_probs, val_y)
    final_top3 = topk_mass(val_probs, 3)
    final_correct = (np.argmax(val_probs, axis=1) == val_y)

    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    for ax, ctx in zip(axes.flatten()[: len(contexts)], contexts):
        probs = probs_by_ctx[ctx]
        data_dist = data_dist_by_ctx.get(ctx, np.zeros(len(vocab), dtype=float))
        valid = support_map.get(ctx, set())
        colors = ["#95a5a6"] * len(vocab)
        for i in valid:
            colors[int(i)] = "#2ecc71"
        ax.bar(np.arange(len(vocab)), probs, color=colors, alpha=0.82)
        ax.plot(np.arange(len(vocab)), data_dist, color="black", marker="o", linewidth=1.2, markersize=2.5)
        support_mass = float(np.sum(probs[list(valid)])) if valid else 0.0
        best_valid_rank = min(
            [int(np.sum(probs > probs[int(i)])) + 1 for i in valid],
            default=len(vocab) + 1,
        )
        ax.set_title(
            "ctx="
            + " ".join(w for w in ctx_to_words(ctx, id_to_word) if w != "<BOS>")
            + f"\nH={float(entropy_bits(probs[None, :])[0]):.2f} bits  support_mass={support_mass:.2f}  best_valid_rank={best_valid_rank}",
            fontsize=8,
        )
        ax.set_ylim(0.0, max(0.45, float(np.max(probs) * 1.15)))
        ax.set_xticks(np.arange(len(vocab)))
        ax.set_xticklabels(vocab, rotation=90, fontsize=6)
        ax.grid(alpha=0.2, axis="y")

    ax = axes.flatten()[8]
    ax.hist(final_rank, bins=np.arange(1, len(vocab) + 2) - 0.5, color="#3498db", alpha=0.85)
    ax.set_title("Correct-token rank")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.2)

    ax = axes.flatten()[9]
    ax.hist(final_entropy[final_correct], bins=16, alpha=0.7, label="correct", color="#27ae60")
    ax.hist(final_entropy[~final_correct], bins=16, alpha=0.7, label="incorrect", color="#c0392b")
    ax.set_title("Entropy by correctness")
    ax.set_xlabel("Entropy (bits)")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)

    ax = axes.flatten()[10]
    bins = np.linspace(0.0, float(np.max(final_entropy)), 12)
    xs = []
    ys = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (final_entropy >= lo) & (final_entropy < hi if hi < bins[-1] else final_entropy <= hi)
        if np.any(mask):
            xs.append(float(np.mean(final_entropy[mask])))
            ys.append(float(np.mean(final_correct[mask])))
    ax.plot(xs, ys, marker="o", linewidth=2.0, color="#8e44ad")
    ax.set_title("Accuracy vs entropy")
    ax.set_xlabel("Entropy (bits)")
    ax.set_ylabel("Accuracy")
    ax.grid(alpha=0.2)

    ax = axes.flatten()[11]
    ax.hist(final_top3, bins=16, color="#f39c12", alpha=0.85)
    ax.set_title("Top-3 probability mass")
    ax.set_xlabel("Top-3 mass")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.2)

    fig.suptitle("Figure 3. Per-context Next-token Distributions", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def build_val_context_stats(
    val_ctx: Sequence[Tuple[int, ...]],
    val_y: Sequence[int],
) -> Tuple[Dict[Tuple[int, ...], Counter], Counter]:
    by_ctx_target: Dict[Tuple[int, ...], Counter] = defaultdict(Counter)
    ctx_counts: Counter = Counter()
    for ctx, y in zip(val_ctx, val_y):
        by_ctx_target[ctx][int(y)] += 1
        ctx_counts[ctx] += 1
    return by_ctx_target, ctx_counts


def compute_ablation_effects(
    runner: FreePhaseRunner,
    val_ctx: Sequence[Tuple[int, ...]],
    val_y: Sequence[int],
    support_map: Mapping[Tuple[int, ...], Set[int]],
    temp: float,
    bos_id: int,
    vocab_size: int,
) -> Dict[str, object]:
    by_ctx_target, ctx_counts = build_val_context_stats(val_ctx, val_y)
    unique_ctx = sorted(ctx_counts.keys())

    base_probs: Dict[Tuple[int, ...], np.ndarray] = {}
    ablated_probs: Dict[Tuple[Tuple[int, ...], int], np.ndarray] = {}
    target_pos_drop = np.zeros((vocab_size, len(unique_ctx[0])), dtype=float)
    target_pos_weight = np.zeros_like(target_pos_drop)
    supportsize_pos_drop: Dict[int, np.ndarray] = {}
    supportsize_pos_weight: Dict[int, np.ndarray] = {}
    global_pos_drop = np.zeros(len(unique_ctx[0]), dtype=float)
    global_pos_weight = np.zeros(len(unique_ctx[0]), dtype=float)
    token_influence = np.zeros((vocab_size, vocab_size), dtype=float)
    token_influence_weight = np.zeros((vocab_size, vocab_size), dtype=float)

    for ctx in unique_ctx:
        base = runner.probs(ctx, temp=temp)
        base_probs[ctx] = base
        support_size = len(support_map.get(ctx, set()))
        if support_size not in supportsize_pos_drop:
            supportsize_pos_drop[support_size] = np.zeros(len(ctx), dtype=float)
            supportsize_pos_weight[support_size] = np.zeros(len(ctx), dtype=float)

        for pos in range(len(ctx)):
            ablated_ctx = list(ctx)
            ablated_ctx[pos] = bos_id
            ablated_ctx_t = tuple(ablated_ctx)
            ablated = runner.probs(ablated_ctx_t, temp=temp)
            ablated_probs[(ctx, pos)] = ablated
            delta = base - ablated

            occ_weight = float(ctx_counts[ctx])
            token_influence[int(ctx[pos]), :] += occ_weight * delta
            token_influence_weight[int(ctx[pos]), :] += occ_weight

            for y, count in by_ctx_target[ctx].items():
                weight = float(count)
                drop = float(np.log(np.clip(base[int(y)], 1e-12, 1.0)) - np.log(np.clip(ablated[int(y)], 1e-12, 1.0)))
                target_pos_drop[int(y), pos] += weight * drop
                target_pos_weight[int(y), pos] += weight
                global_pos_drop[pos] += weight * drop
                global_pos_weight[pos] += weight
                supportsize_pos_drop[support_size][pos] += weight * drop
                supportsize_pos_weight[support_size][pos] += weight

    target_pos = target_pos_drop / np.clip(target_pos_weight, 1e-12, None)
    global_pos = global_pos_drop / np.clip(global_pos_weight, 1e-12, None)
    supportsize_pos = {
        k: supportsize_pos_drop[k] / np.clip(supportsize_pos_weight[k], 1e-12, None)
        for k in sorted(supportsize_pos_drop)
    }
    token_matrix = token_influence / np.clip(token_influence_weight, 1e-12, None)

    return {
        "base_probs_by_ctx": base_probs,
        "ablated_probs": ablated_probs,
        "target_position_drop": target_pos,
        "global_position_drop": global_pos,
        "supportsize_position_drop": supportsize_pos,
        "token_influence_matrix": token_matrix,
    }


def plot_context_use(
    out_path: Path,
    vocab: Sequence[str],
    target_position_drop: np.ndarray,
    global_position_drop: np.ndarray,
    supportsize_position_drop: Mapping[int, np.ndarray],
    token_influence_matrix: np.ndarray,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    ax = axes[0, 0]
    ax.bar(np.arange(1, len(global_position_drop) + 1), global_position_drop, color="#2980b9")
    ax.set_title("Average causal drop in log p(correct) by corrupted position")
    ax.set_xlabel("Context position (1 = oldest, 6 = newest)")
    ax.set_ylabel("Delta log p(correct)")
    ax.grid(alpha=0.25, axis="y")

    ax = axes[0, 1]
    im = ax.imshow(target_position_drop, aspect="auto", cmap="magma")
    ax.set_title("Target-token x position influence")
    ax.set_xlabel("Context position")
    ax.set_ylabel("True next token")
    ax.set_xticks(np.arange(len(global_position_drop)))
    ax.set_xticklabels(np.arange(1, len(global_position_drop) + 1))
    ax.set_yticks(np.arange(len(vocab)))
    ax.set_yticklabels(vocab, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 0]
    for support_size, vals in supportsize_position_drop.items():
        ax.plot(np.arange(1, len(vals) + 1), vals, marker="o", linewidth=2.0, label=f"|support|={support_size}")
    ax.set_title("Position influence by grammar branching")
    ax.set_xlabel("Context position")
    ax.set_ylabel("Delta log p(correct)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    vmax = float(np.max(np.abs(token_influence_matrix)))
    im = ax.imshow(token_influence_matrix, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_title("Context-token influence on next-token distribution")
    ax.set_xlabel("Candidate next token")
    ax.set_ylabel("Ablated context token")
    ax.set_xticks(np.arange(len(vocab)))
    ax.set_yticks(np.arange(len(vocab)))
    ax.set_xticklabels(vocab, rotation=90, fontsize=7)
    ax.set_yticklabels(vocab, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Figure 4. Context Use", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def load_gate_trajectory(run_dir: Path) -> Tuple[np.ndarray, List[int]]:
    files = sorted(run_dir.glob("0_vg_unique_epoch*.npy"), key=lambda p: int(p.stem.split("epoch")[-1]))
    epochs = [int(p.stem.split("epoch")[-1]) for p in files]
    arr = np.stack([np.load(p) for p in files], axis=0)
    return arr, epochs


def select_probe_contexts(val_ctx: Sequence[Tuple[int, ...]], support_map: Mapping[Tuple[int, ...], Set[int]], word_to_id: Mapping[str, int]) -> List[Tuple[int, ...]]:
    bos = int(word_to_id["<BOS>"])
    desired = [
        (bos, bos, bos, bos, bos, bos),
        (bos, bos, bos, bos, bos, int(word_to_id["the"])),
        (bos, bos, bos, int(word_to_id["the"]), int(word_to_id["boy"]), int(word_to_id["is"])),
        (bos, bos, int(word_to_id["the"]), int(word_to_id["boy"]), int(word_to_id["likes"]), int(word_to_id["the"])),
    ]
    val_set = set(val_ctx)
    return [ctx for ctx in desired if ctx in val_set and ctx in support_map]


def plot_internal_circuits(
    out_path: Path,
    run_dir: Path,
    id_to_word: Mapping[int, str],
    final_gate: np.ndarray,
    probe_contexts: Sequence[Tuple[int, ...]],
    support_map: Mapping[Tuple[int, ...], Set[int]],
    temp: float,
) -> List[Dict[str, object]]:
    gate_traj, gate_epochs = load_gate_trajectory(run_dir)
    pca = PCA(n_components=2)
    gate_xy = pca.fit_transform(gate_traj)

    files = {int(p.stem.split("epoch")[-1]): p for p in run_dir.glob("0_vout_val_epoch*.npy")}
    val_x = np.load(run_dir / "val_x.npy")
    vocab = [id_to_word[i] for i in range(len(id_to_word))]

    trainer = load_module(DEFAULT_TRAINER_SCRIPT, "trainer16_onehot_internal")
    bit_v0 = 0.0
    bit_v1 = 1.0
    decoded_val_ctx = [
        tuple(int(trainer.WORD_TO_ID[w]) for w in trainer.decode_context_bits_to_words(x, bit_v0=bit_v0, bit_v1=bit_v1))
        for x in val_x
    ]
    index_for_ctx: Dict[Tuple[int, ...], int] = {}
    for i, ctx in enumerate(decoded_val_ctx):
        index_for_ctx.setdefault(ctx, i)

    probe_rows: List[Dict[str, object]] = []
    probe_support_series: Dict[Tuple[int, ...], List[float]] = {ctx: [] for ctx in probe_contexts}
    for epoch in gate_epochs:
        probs = softmax_from_logits(np.load(files[epoch]), temp=temp)
        for ctx in probe_contexts:
            idx = index_for_ctx[ctx]
            valid = list(sorted(support_map[ctx]))
            probe_support_series[ctx].append(float(np.sum(probs[idx, valid])))

    gate_matrix = np.asarray(final_gate, dtype=float).reshape(len(vocab), -1)
    mean_by_position = gate_matrix.reshape(len(vocab), 6, 4).mean(axis=2).mean(axis=0)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    ax = axes[0, 0]
    sc = ax.scatter(gate_xy[:, 0], gate_xy[:, 1], c=gate_epochs, cmap="viridis", s=60)
    for x, y, ep in zip(gate_xy[:, 0], gate_xy[:, 1], gate_epochs):
        ax.text(x, y, str(ep), fontsize=7)
    ax.set_title("Gate-voltage trajectory PCA")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(alpha=0.25)
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="Epoch")

    ax = axes[0, 1]
    im = ax.imshow(gate_matrix, aspect="auto", cmap="viridis")
    ax.set_title("Final gate matrix (output token x input bit)")
    ax.set_xlabel("Input bit index")
    ax.set_ylabel("Output token")
    ax.set_yticks(np.arange(len(vocab)))
    ax.set_yticklabels(vocab, fontsize=7)
    for xline in [3.5, 7.5, 11.5, 15.5, 19.5]:
        ax.axvline(xline, color="white", linewidth=0.7, alpha=0.7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax = axes[1, 0]
    for ctx in probe_contexts:
        label = " ".join(w for w in ctx_to_words(ctx, id_to_word) if w != "<BOS>")
        label = label if label else "<BOS>"
        ax.plot(gate_epochs, probe_support_series[ctx], marker="o", linewidth=2.0, label=label)
        probe_rows.append(
            {
                "context": label,
                "final_support_mass": probe_support_series[ctx][-1],
            }
        )
    ax.set_title("Support-mass refinement for probe contexts")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Support mass")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.bar(np.arange(1, 7), mean_by_position, color="#16a085")
    ax.set_title("Mean gate strength by context position")
    ax.set_xlabel("Context position")
    ax.set_ylabel("Mean VG")
    ax.grid(alpha=0.25, axis="y")

    fig.suptitle("Figure 5. Internal Representations and Subcircuits", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    return probe_rows


def minimal_pair_specs(word_to_id: Mapping[str, int]) -> List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]]:
    bos = int(word_to_id["<BOS>"])
    return [
        (
            "is vs likes",
            (bos, bos, bos, int(word_to_id["the"]), int(word_to_id["boy"]), int(word_to_id["is"])),
            (bos, bos, bos, int(word_to_id["the"]), int(word_to_id["boy"]), int(word_to_id["likes"])),
        ),
        (
            "runs vs is",
            (bos, bos, bos, int(word_to_id["the"]), int(word_to_id["cat"]), int(word_to_id["runs"])),
            (bos, bos, bos, int(word_to_id["the"]), int(word_to_id["cat"]), int(word_to_id["is"])),
        ),
        (
            "on vs red",
            (bos, bos, int(word_to_id["the"]), int(word_to_id["boy"]), int(word_to_id["is"]), int(word_to_id["on"])),
            (bos, bos, int(word_to_id["the"]), int(word_to_id["boy"]), int(word_to_id["is"]), int(word_to_id["red"])),
        ),
    ]


def generate_rollouts(
    runner: FreePhaseRunner,
    word_to_id: Mapping[str, int],
    id_to_word: Mapping[int, str],
    support_map: Mapping[Tuple[int, ...], Set[int]],
    sample_temp: float,
    num_rollouts: int,
    max_steps: int,
) -> Dict[str, object]:
    bos = int(word_to_id["<BOS>"])
    rng = np.random.default_rng(0)
    sequences: List[List[str]] = []
    lengths: List[int] = []
    invalid_rollouts = 0
    invalid_steps = 0

    for _ in range(num_rollouts):
        ctx = (bos, bos, bos, bos, bos, bos)
        seq: List[str] = []
        invalid_here = False
        for _step in range(max_steps):
            probs = runner.probs(ctx, temp=sample_temp)
            next_id = int(rng.choice(np.arange(len(id_to_word)), p=probs))
            if next_id not in support_map.get(ctx, set()):
                invalid_steps += 1
                invalid_here = True
            next_word = str(id_to_word[next_id])
            seq.append(next_word)
            ctx = tuple(list(ctx[1:]) + [next_id])
            if next_word == "<EOS>":
                break
        sequences.append(seq)
        lengths.append(len(seq))
        invalid_rollouts += int(invalid_here)

    return {
        "sequences": sequences,
        "lengths": lengths,
        "invalid_rollout_rate": float(invalid_rollouts / max(1, num_rollouts)),
        "invalid_step_rate": float(invalid_steps / max(1, sum(lengths))),
    }


def plot_minimal_pairs_calibration_generation(
    out_path: Path,
    vocab: Sequence[str],
    id_to_word: Mapping[int, str],
    pair_specs: Sequence[Tuple[str, Tuple[int, ...], Tuple[int, ...]]],
    probs_by_ctx: Mapping[Tuple[int, ...], np.ndarray],
    final_conf: np.ndarray,
    final_correct: np.ndarray,
    rollout_stats: Mapping[str, object],
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(16, 14))
    for ax, (title, ctx_a, ctx_b) in zip(axes.flatten()[:3], pair_specs):
        pa = probs_by_ctx[ctx_a]
        pb = probs_by_ctx[ctx_b]
        delta = pb - pa
        order = np.argsort(-np.abs(delta))[:6]
        width = 0.35
        ax.bar(np.arange(order.size) - width / 2, pa[order], width=width, label="A", color="#3498db")
        ax.bar(np.arange(order.size) + width / 2, pb[order], width=width, label="B", color="#e74c3c")
        ax.set_xticks(np.arange(order.size))
        ax.set_xticklabels([vocab[int(i)] for i in order], rotation=35, ha="right")
        ax.set_title(
            f"{title}\nA={' '.join(w for w in ctx_to_words(ctx_a, id_to_word) if w != '<BOS>') or '<BOS>'}\n"
            f"B={' '.join(w for w in ctx_to_words(ctx_b, id_to_word) if w != '<BOS>') or '<BOS>'}",
            fontsize=8,
        )
        ax.grid(alpha=0.2, axis="y")
        ax.legend(fontsize=8)

    ax = axes[1, 1]
    bins = np.linspace(0.0, 1.0, 11)
    bin_ids = np.digitize(final_conf, bins[1:-1], right=False)
    xs = []
    ys = []
    ns = []
    for b in range(10):
        mask = bin_ids == b
        if np.any(mask):
            xs.append(float(np.mean(final_conf[mask])))
            ys.append(float(np.mean(final_correct[mask])))
            ns.append(int(np.sum(mask)))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#7f8c8d", linewidth=1.0)
    ax.plot(xs, ys, marker="o", linewidth=2.0, color="#2c3e50")
    for x, y, n in zip(xs, ys, ns):
        ax.text(x, y, str(n), fontsize=7)
    ax.set_title("Reliability diagram")
    ax.set_xlabel("Top-1 confidence")
    ax.set_ylabel("Accuracy")
    ax.grid(alpha=0.2)

    ax = axes[2, 0]
    lengths = np.asarray(rollout_stats["lengths"], dtype=int)
    ax.hist(lengths, bins=np.arange(lengths.min(), lengths.max() + 2) - 0.5, color="#9b59b6", alpha=0.85)
    ax.set_title("Rollout length distribution")
    ax.set_xlabel("Generated length")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.2)

    ax = axes[2, 1]
    ax.axis("off")
    lines = [
        f"sample_temp={rollout_stats['sample_temp']:.3f}",
        f"invalid_rollout_rate={rollout_stats['invalid_rollout_rate']:.3f}",
        f"invalid_step_rate={rollout_stats['invalid_step_rate']:.3f}",
        "",
        "sample rollouts:",
    ]
    for seq in rollout_stats["sequences"][:10]:
        lines.append("  " + " ".join(seq))
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=8)
    ax.set_title("Generation diagnostics")

    fig.suptitle("Figure 6. Minimal Pairs, Calibration, and Generation", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def top_edges_table(final_gate: np.ndarray, id_to_word: Mapping[int, str]) -> List[Dict[str, object]]:
    rows = []
    gate_matrix = np.asarray(final_gate, dtype=float).reshape(len(id_to_word), 24)
    for out_id in range(gate_matrix.shape[0]):
        for in_bit in range(gate_matrix.shape[1]):
            rows.append(
                {
                    "output_token": id_to_word[out_id],
                    "input_bit": in_bit,
                    "context_position": (in_bit // 4) + 1,
                    "token_bit": (in_bit % 4) + 1,
                    "vg": float(gate_matrix[out_id, in_bit]),
                }
            )
    rows.sort(key=lambda r: float(r["vg"]), reverse=True)
    return rows[:25]


def write_report(
    out_path: Path,
    checkpoint: SelectedCheckpoint,
    ranking_rows: Sequence[Mapping[str, object]],
    bigram_stats: Mapping[str, float],
    training_stats: Mapping[str, object],
    probe_rows: Sequence[Mapping[str, object]],
    rollout_stats: Mapping[str, object],
    top_edges: Sequence[Mapping[str, object]],
) -> None:
    final_entropy = np.asarray(training_stats["final_entropy"], dtype=float)
    final_rank = np.asarray(training_stats["final_rank"], dtype=float)
    final_correct = np.asarray(training_stats["final_correct"], dtype=float)
    lines = [
        "# 16-vocab one-hot CE best-support analysis",
        "",
        "## Selected checkpoint",
        f"- Run: `{checkpoint.run_name}`",
        f"- Epoch: `{checkpoint.epoch}`",
        f"- Sweep criterion: `support_acc == 1.0`, then highest `val_qmass_mean`",
        f"- Selected metrics: `val_qmass_mean = {checkpoint.qmass_mean:.6f}`, `val_exact_acc = {checkpoint.exact_acc:.6f}`, `val_soft_ce = {checkpoint.soft_ce:.6f}`",
        "",
        "## High-level findings",
        f"- The selected checkpoint is the best of `{len(ranking_rows)}` support-perfect epochs by `val_qmass_mean`.",
        f"- Final validation mean entropy is `{float(np.mean(final_entropy)):.3f}` bits, with mean correct-token rank `{float(np.mean(final_rank)):.3f}`.",
        f"- Final validation exact accuracy is `{float(np.mean(final_correct)):.3f}`, while support-perfect selection still leaves substantial uncertainty inside the valid set.",
        f"- Bigram residual L1 is `{float(bigram_stats['bigram_residual_l1']):.3f}` and forbidden bigram probability mass is `{float(bigram_stats['forbidden_bigram_mass']):.3f}`.",
        f"- Rollout invalid-transition rate at sampling temperature `{float(rollout_stats['sample_temp']):.3f}` is `{float(rollout_stats['invalid_rollout_rate']):.3f}`.",
        "",
        "## Figure bundle",
        "- `figure1_training_uncertainty.png`: training curves, perplexity, uncertainty, reliability",
        "- `figure2_distributional_competence.png`: empirical/model bigrams, residuals, forbidden-transition leaks",
        "- `figure3_context_distributions.png`: representative contexts plus rank/entropy/top-k summaries",
        "- `figure4_context_use.png`: causal position ablations and token influence matrix",
        "- `figure5_internal_circuits.png`: gate trajectory PCA, gate matrix, probe refinement, mean gate by position",
        "- `figure6_minimal_pairs_calibration_generation.png`: minimal-pair panels, reliability, rollout diagnostics",
        "",
        "## Probe contexts",
    ]
    for row in probe_rows:
        lines.append(f"- `{row['context']}`: final support mass `{float(row['final_support_mass']):.3f}`")

    lines.extend(
        [
            "",
            "## Strongest final edges",
        ]
    )
    for row in top_edges[:10]:
        lines.append(
            f"- `{row['output_token']}` <= position `{row['context_position']}` bit `{row['token_bit']}` with `VG = {float(row['vg']):.4f}`"
        )

    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    sweep_dir = args.sweep_dir.resolve()
    trainer_script = args.trainer_script.resolve()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    trainer = load_module(trainer_script, "trainer16_onehot_analysis")

    if args.run_dir is not None:
        if args.epoch is None:
            raise ValueError("--epoch is required when --run-dir is provided")
        checkpoint = SelectedCheckpoint(
            run_dir=args.run_dir.resolve(),
            epoch=int(args.epoch),
            support_acc=float("nan"),
            qmass_mean=float("nan"),
            exact_acc=float("nan"),
            soft_ce=float("nan"),
            run_name=args.run_dir.name,
        )
        ranking_rows: List[Dict[str, object]] = []
    else:
        checkpoint, ranking_rows = find_best_support_checkpoint(sweep_dir)

    meta = json.loads((checkpoint.run_dir / "run_meta.json").read_text())
    dataset = reconstruct_dataset(trainer, meta)
    val_x_saved = np.load(checkpoint.run_dir / "val_x.npy")
    val_y_saved = np.load(checkpoint.run_dir / "val_y.npy")
    if not np.allclose(val_x_saved, np.asarray(dataset["val_x"], dtype=float)):
        raise RuntimeError("Reconstructed val_x does not match saved val_x.npy")
    if not np.array_equal(val_y_saved, np.asarray(dataset["val_y"], dtype=int)):
        raise RuntimeError("Reconstructed val_y does not match saved val_y.npy")

    vocab = list(meta["dataset"]["vocab"])
    word_to_id = {w: i for i, w in enumerate(vocab)}
    id_to_word = {i: w for i, w in enumerate(vocab)}
    support_map = build_support_map(trainer, bit_v0=float(meta["dataset"]["bit_v0"]), bit_v1=float(meta["dataset"]["bit_v1"]))
    data_dist_by_ctx, data_ctx_counts = build_context_distributions(dataset["all_ctx"], dataset["all_y"], len(vocab))
    final_gate = np.load(checkpoint.run_dir / f"0_vg_unique_epoch{checkpoint.epoch}.npy")
    temp = float(meta["softmax_temp"])

    selection_summary = {
        "selected_run": checkpoint.run_name,
        "selected_epoch": checkpoint.epoch,
        "selected_support_acc": checkpoint.support_acc,
        "selected_val_qmass_mean": checkpoint.qmass_mean,
        "selected_val_exact_acc": checkpoint.exact_acc,
        "selected_val_soft_ce": checkpoint.soft_ce,
        "run_dir": str(checkpoint.run_dir),
        "support_context_count": len(support_map),
        "unique_dataset_contexts": len(set(dataset["all_ctx"])),
        "unique_val_contexts": len(set(dataset["val_ctx"])),
    }
    (out_dir / "selection_summary.json").write_text(json.dumps(selection_summary, indent=2))

    if ranking_rows:
        write_csv(
            out_dir / "checkpoint_ranking_support1.csv",
            ["run", "epoch", "support_acc", "val_qmass_mean", "val_exact_acc", "val_soft_ce"],
            ranking_rows,
        )

    runner = FreePhaseRunner(trainer, meta, final_gate)
    try:
        grammar_contexts = sorted(support_map.keys())
        probs_by_ctx = {ctx: runner.probs(ctx, temp=temp) for ctx in grammar_contexts}
        data_bigram, model_bigram = build_bigram_distributions(dataset["all_ctx"], dataset["all_y"], probs_by_ctx, len(vocab))
        bigram_valid = build_bigram_validity(support_map, len(vocab))
        bigram_stats = plot_distributional_competence(
            out_dir / "figure2_distributional_competence.png",
            vocab=vocab,
            data_bigram=data_bigram,
            model_bigram=model_bigram,
            valid_bigram=bigram_valid,
        )

        training_stats = plot_training_uncertainty(
            out_dir / "figure1_training_uncertainty.png",
            run_dir=checkpoint.run_dir,
            selected_epoch=checkpoint.epoch,
            val_y=np.asarray(dataset["val_y"], dtype=int),
            val_ctx=dataset["val_ctx"],
            support_map=support_map,
            temp=temp,
        )

        representative_contexts = select_representative_contexts(support_map, word_to_id)
        plot_context_panels(
            out_dir / "figure3_context_distributions.png",
            vocab=vocab,
            id_to_word=id_to_word,
            contexts=representative_contexts,
            probs_by_ctx=probs_by_ctx,
            data_dist_by_ctx=data_dist_by_ctx,
            support_map=support_map,
            val_probs=np.asarray(training_stats["final_probs"], dtype=float),
            val_y=np.asarray(dataset["val_y"], dtype=int),
        )

        ablation = compute_ablation_effects(
            runner=runner,
            val_ctx=dataset["val_ctx"],
            val_y=dataset["val_y"],
            support_map=support_map,
            temp=temp,
            bos_id=word_to_id["<BOS>"],
            vocab_size=len(vocab),
        )
        plot_context_use(
            out_dir / "figure4_context_use.png",
            vocab=vocab,
            target_position_drop=np.asarray(ablation["target_position_drop"], dtype=float),
            global_position_drop=np.asarray(ablation["global_position_drop"], dtype=float),
            supportsize_position_drop=ablation["supportsize_position_drop"],
            token_influence_matrix=np.asarray(ablation["token_influence_matrix"], dtype=float),
        )

        probe_contexts = select_probe_contexts(dataset["val_ctx"], support_map, word_to_id)
        probe_rows = plot_internal_circuits(
            out_dir / "figure5_internal_circuits.png",
            run_dir=checkpoint.run_dir,
            id_to_word=id_to_word,
            final_gate=final_gate,
            probe_contexts=probe_contexts,
            support_map=support_map,
            temp=temp,
        )

        pair_specs = minimal_pair_specs(word_to_id)
        for _, ctx_a, ctx_b in pair_specs:
            probs_by_ctx.setdefault(ctx_a, runner.probs(ctx_a, temp=temp))
            probs_by_ctx.setdefault(ctx_b, runner.probs(ctx_b, temp=temp))
        rollout_stats = generate_rollouts(
            runner=runner,
            word_to_id=word_to_id,
            id_to_word=id_to_word,
            support_map=support_map,
            sample_temp=float(args.sample_temp),
            num_rollouts=int(args.num_rollouts),
            max_steps=int(args.max_rollout_steps),
        )
        rollout_stats["sample_temp"] = float(args.sample_temp)
        plot_minimal_pairs_calibration_generation(
            out_dir / "figure6_minimal_pairs_calibration_generation.png",
            vocab=vocab,
            id_to_word=id_to_word,
            pair_specs=pair_specs,
            probs_by_ctx=probs_by_ctx,
            final_conf=np.asarray(training_stats["final_conf"], dtype=float),
            final_correct=np.asarray(training_stats["final_correct"], dtype=float),
            rollout_stats=rollout_stats,
        )

        top_edges = top_edges_table(final_gate, id_to_word)
        write_csv(
            out_dir / "top_edges.csv",
            ["output_token", "input_bit", "context_position", "token_bit", "vg"],
            top_edges,
        )
        context_rows = []
        for ctx in representative_contexts:
            probs = probs_by_ctx[ctx]
            support = sorted(support_map.get(ctx, set()))
            context_rows.append(
                {
                    "context": " ".join(ctx_to_words(ctx, id_to_word)),
                    "support_tokens": " ".join(id_to_word[i] for i in support),
                    "entropy_bits": float(entropy_bits(probs[None, :])[0]),
                    "support_mass": float(np.sum(probs[support])) if support else 0.0,
                    "top3_mass": float(np.sum(np.sort(probs)[-3:])),
                    "top_prediction": id_to_word[int(np.argmax(probs))],
                }
            )
        write_csv(
            out_dir / "representative_context_metrics.csv",
            ["context", "support_tokens", "entropy_bits", "support_mass", "top3_mass", "top_prediction"],
            context_rows,
        )

        write_report(
            out_dir / "report.md",
            checkpoint=checkpoint,
            ranking_rows=ranking_rows,
            bigram_stats=bigram_stats,
            training_stats=training_stats,
            probe_rows=probe_rows,
            rollout_stats=rollout_stats,
            top_edges=top_edges,
        )
    finally:
        runner.close()

    print(f"analysis_dir={out_dir}")
    print(f"selected_run={checkpoint.run_name}")
    print(f"selected_epoch={checkpoint.epoch}")


if __name__ == "__main__":
    main()
