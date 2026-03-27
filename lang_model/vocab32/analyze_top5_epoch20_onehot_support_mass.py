#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_SWEEP_DIR = ROOT / "results_language_32_embed4_onehotce" / "sweeps" / "top40_freshproc_p20_20260326-113811"
DEFAULT_OUTPUT_DIR = ROOT / "analysis_top5_completed20_support_mass_20260326"
DEFAULT_TRAINER_SCRIPT = ROOT / "clln_lang_trainer_embed4_onehot_ce.py"


@dataclass
class RunArtifacts:
    run_dir: Path
    run_name: str
    temp: float
    train_ctx: np.ndarray
    train_y: np.ndarray
    val_ctx: np.ndarray
    val_y: np.ndarray
    support_map: Mapping[Tuple[int, ...], np.ndarray]
    unigram_support: np.ndarray


@dataclass
class EpochMetrics:
    run_name: str
    epoch: int
    support_mass_mean: float
    support_acc: float
    true_token_prob_mean: float
    exact_acc: float
    soft_ce: float
    entropy_bits_mean: float
    unseen_contexts: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze top 5 completed 32-vocab one-hot CE runs by corrected support-mass.")
    p.add_argument("--sweep-dir", type=Path, default=DEFAULT_SWEEP_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--trainer-script", type=Path, default=DEFAULT_TRAINER_SCRIPT)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--epoch", type=int, default=20)
    return p.parse_args()


def load_trainer_constants(script_path: Path):
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("vocab32_onehot_trainer", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load trainer module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def softmax_rows(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=float)
    z = z - np.max(z, axis=1, keepdims=True)
    ez = np.exp(z)
    denom = np.sum(ez, axis=1, keepdims=True)
    denom = np.clip(denom, 1e-12, None)
    return ez / denom


def entropy_bits(probs: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probs, dtype=float), 1e-12, 1.0)
    return -np.sum(p * (np.log(p) / np.log(2.0)), axis=1)


def infer_context_ids_from_x(X: np.ndarray, trainer) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != int(trainer.INPUT_DIM):
        raise ValueError(f"Expected X with shape (N, {trainer.INPUT_DIM}), got {X.shape}")

    embed_to_idx: Dict[Tuple[float, ...], int] = {}
    for idx, tok in enumerate(trainer.INPUT_VOCAB):
        embed = tuple(np.round(np.asarray(trainer.TOKEN_EMBED_4D[tok], dtype=float), 8).tolist())
        embed_to_idx[embed] = idx

    ctx = np.zeros((X.shape[0], int(trainer.CONTEXT_LEN)), dtype=int)
    for row_idx, row in enumerate(X):
        ids: List[int] = []
        for pos in range(int(trainer.CONTEXT_LEN)):
            sl = row[pos * int(trainer.TOKEN_EMBED_DIM):(pos + 1) * int(trainer.TOKEN_EMBED_DIM)]
            key = tuple(np.round(np.asarray(sl, dtype=float), 8).tolist())
            tok_idx = embed_to_idx.get(key)
            if tok_idx is None:
                for cand_key, cand_idx in embed_to_idx.items():
                    if np.allclose(np.asarray(cand_key, dtype=float), sl, atol=1e-8, rtol=0.0):
                        tok_idx = cand_idx
                        break
            if tok_idx is None:
                raise RuntimeError(f"Failed to decode context row={row_idx} pos={pos}")
            ids.append(int(tok_idx))
        ctx[row_idx, :] = np.asarray(ids, dtype=int)
    return ctx


def load_run_artifacts(run_dir: Path, trainer) -> RunArtifacts:
    meta = json.loads((run_dir / "run_meta.json").read_text())
    train_x = np.asarray(np.load(run_dir / "train_x.npy"), dtype=float)
    train_y = np.asarray(np.load(run_dir / "train_y.npy"), dtype=int)
    val_y = np.asarray(np.load(run_dir / "val_y.npy"), dtype=int)
    val_ctx = np.asarray(np.load(run_dir / "val_ctx.npy"), dtype=int)
    train_ctx_path = run_dir / "train_ctx.npy"
    if train_ctx_path.exists():
        train_ctx = np.asarray(np.load(train_ctx_path), dtype=int)
    else:
        train_ctx = infer_context_ids_from_x(train_x, trainer)

    q_counts: Dict[Tuple[int, ...], np.ndarray] = {}
    unigram = np.zeros((int(trainer.OUTPUT_DIM),), dtype=float)
    for ctx_row, y in zip(train_ctx, train_y):
        key = tuple(int(v) for v in ctx_row.tolist())
        if key not in q_counts:
            q_counts[key] = np.zeros((int(trainer.OUTPUT_DIM),), dtype=float)
        q_counts[key][int(y)] += 1.0
        unigram[int(y)] += 1.0
    support_map = {k: (v > 0.0) for k, v in q_counts.items()}
    unigram_support = unigram > 0.0

    return RunArtifacts(
        run_dir=run_dir,
        run_name=run_dir.name,
        temp=float(meta["softmax_temp"]),
        train_ctx=train_ctx,
        train_y=train_y,
        val_ctx=val_ctx,
        val_y=val_y,
        support_map=support_map,
        unigram_support=unigram_support,
    )


def evaluate_epoch(run: RunArtifacts, epoch: int) -> EpochMetrics:
    vout = np.asarray(np.load(run.run_dir / f"0_vout_val_epoch{epoch}.npy"), dtype=float)
    probs = softmax_rows(vout / float(run.temp))
    pred = np.argmax(vout, axis=1)

    support_mass = []
    true_prob = []
    valid_hits = []
    unseen = 0
    for i, ctx_row in enumerate(run.val_ctx):
        key = tuple(int(v) for v in ctx_row.tolist())
        support = run.support_map.get(key)
        if support is None:
            support = run.unigram_support
            unseen += 1
        support_mass.append(float(np.sum(probs[i][support])))
        true_prob.append(float(probs[i, int(run.val_y[i])]))
        valid_hits.append(float(bool(support[int(pred[i])])) )

    summary = json.loads((run.run_dir / f"0_epoch_summary_epoch{epoch}.json").read_text())
    return EpochMetrics(
        run_name=run.run_name,
        epoch=int(epoch),
        support_mass_mean=float(np.mean(support_mass)),
        support_acc=float(np.mean(valid_hits)),
        true_token_prob_mean=float(np.mean(true_prob)),
        exact_acc=float(summary["val"]["exact_acc"]),
        soft_ce=float(summary["val"]["soft_ce"]),
        entropy_bits_mean=float(np.mean(entropy_bits(probs))),
        unseen_contexts=int(unseen),
    )


def major_error_family(pred_word: str, true_word: str) -> str:
    if pred_word == "are" and true_word == "am":
        return "are_for_am"
    if pred_word == "?" and true_word != "?":
        return "early_question_mark"
    if pred_word == "is" and true_word in {"robot", "signal", "city", "river", "fire", "dream"}:
        return "is_for_object_noun"
    if pred_word in {"again", "lost", ".", "!"} and true_word in {".", "!"}:
        return "bad_terminal_transition"
    if pred_word in {"again", "lost"}:
        return "repeat_or_stall"
    return "other"


def summarize_run_failures(run: RunArtifacts, trainer, epoch: int) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, float]]:
    vout = np.asarray(np.load(run.run_dir / f"0_vout_val_epoch{epoch}.npy"), dtype=float)
    probs = softmax_rows(vout / float(run.temp))
    pred = np.argmax(vout, axis=1)

    family_counter: Counter[str] = Counter()
    low_context_mass: Dict[Tuple[int, ...], List[float]] = defaultdict(list)
    valid_support_sizes: List[int] = []
    for i, ctx_row in enumerate(run.val_ctx):
        key = tuple(int(v) for v in ctx_row.tolist())
        support = run.support_map.get(key, run.unigram_support)
        valid_support_sizes.append(int(np.sum(support)))
        mass = float(np.sum(probs[i][support]))
        low_context_mass[key].append(mass)
        if not bool(support[int(pred[i])]):
            family_counter[major_error_family(trainer.OUTPUT_ID_TO_WORD[int(pred[i])], trainer.OUTPUT_ID_TO_WORD[int(run.val_y[i])])] += 1

    family_rows = [
        {"run": run.run_name, "epoch": epoch, "family": fam, "count": int(cnt)}
        for fam, cnt in family_counter.most_common()
    ]
    low_rows = []
    for ctx_key, masses in sorted(low_context_mass.items(), key=lambda kv: float(np.mean(kv[1])))[:12]:
        low_rows.append(
            {
                "run": run.run_name,
                "epoch": epoch,
                "context": " ".join(trainer.INPUT_ID_TO_WORD[int(v)] for v in ctx_key),
                "mean_support_mass": float(np.mean(masses)),
                "count": int(len(masses)),
            }
        )
    extras = {
        "mean_support_size": float(np.mean(valid_support_sizes)),
        "median_support_size": float(np.median(valid_support_sizes)),
        "invalid_count": int(sum(family_counter.values())),
    }
    return family_rows, low_rows, extras


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_epoch20_ranking(out_path: Path, ranked_rows: Sequence[Mapping[str, object]]) -> None:
    top = list(ranked_rows[:5])
    labels = [str(r["run"]) for r in top]
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width, [float(r["support_mass_mean"]) for r in top], width=width, label="support_mass_mean")
    ax.bar(x, [float(r["support_acc"]) for r in top], width=width, label="support_acc")
    ax.bar(x + width, [float(r["exact_acc"]) for r in top], width=width, label="exact_acc")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("Top 5 Completed Epoch-20 Runs by Corrected Support-Mass")
    ax.set_ylabel("Metric Value")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_metric_space(out_path: Path, ranked_rows: Sequence[Mapping[str, object]], selected_runs: Sequence[str]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    selected = set(selected_runs)
    temps = sorted({float(r["temp"]) for r in ranked_rows})
    temp_to_color = {t: c for t, c in zip(temps, ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"])}
    for row in ranked_rows:
        color = temp_to_color[float(row["temp"])]
        marker = "o" if str(row["run"]) not in selected else "*"
        size = 40 if str(row["run"]) not in selected else 180
        axes[0].scatter(float(row["support_mass_mean"]), float(row["support_acc"]), c=color, marker=marker, s=size, alpha=0.9)
        axes[1].scatter(float(row["support_mass_mean"]), float(row["exact_acc"]), c=color, marker=marker, s=size, alpha=0.9)
    axes[0].set_xlabel("support_mass_mean")
    axes[0].set_ylabel("support_acc")
    axes[1].set_xlabel("support_mass_mean")
    axes[1].set_ylabel("exact_acc")
    axes[0].set_title("Epoch-20 Support Mass vs Support Accuracy")
    axes[1].set_title("Epoch-20 Support Mass vs Exact Accuracy")
    for ax in axes:
        ax.grid(alpha=0.25)
    handles = [plt.Line2D([0], [0], marker="o", linestyle="", color=temp_to_color[t], label=f"T={t:g}") for t in temps]
    handles.append(plt.Line2D([0], [0], marker="*", linestyle="", color="black", label="selected top5"))
    axes[1].legend(handles=handles, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_training_curves(out_path: Path, top5_curves: Mapping[str, Sequence[EpochMetrics]]) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    for run_name, curve in top5_curves.items():
        epochs = [m.epoch for m in curve]
        axes[0].plot(epochs, [m.support_mass_mean for m in curve], label=run_name)
        axes[1].plot(epochs, [m.support_acc for m in curve], label=run_name)
        axes[2].plot(epochs, [m.true_token_prob_mean for m in curve], label=run_name)
    axes[0].set_ylabel("support_mass_mean")
    axes[1].set_ylabel("support_acc")
    axes[2].set_ylabel("true_token_prob_mean")
    axes[2].set_xlabel("epoch")
    axes[0].set_title("Top-5 Training Curves")
    for ax in axes:
        ax.grid(alpha=0.25)
    axes[0].legend(loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_error_families(out_path: Path, family_rows: Sequence[Mapping[str, object]], selected_runs: Sequence[str]) -> None:
    families = ["are_for_am", "early_question_mark", "bad_terminal_transition", "is_for_object_noun", "repeat_or_stall", "other"]
    by_run = {run: {fam: 0 for fam in families} for run in selected_runs}
    for row in family_rows:
        by_run[str(row["run"])][str(row["family"])] = int(row["count"])
    x = np.arange(len(selected_runs))
    fig, ax = plt.subplots(figsize=(12, 6))
    bottom = np.zeros((len(selected_runs),), dtype=float)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:gray"]
    for fam, color in zip(families, colors):
        vals = np.asarray([by_run[run][fam] for run in selected_runs], dtype=float)
        ax.bar(x, vals, bottom=bottom, label=fam, color=color)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(selected_runs, rotation=25, ha="right")
    ax.set_ylabel("invalid argmax count")
    ax.set_title("Epoch-20 Invalid-Prediction Families for Top 5 Runs")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_support_mass_histograms(out_path: Path, top5_runs: Sequence[RunArtifacts], epoch: int) -> None:
    fig, axes = plt.subplots(len(top5_runs), 1, figsize=(12, 2.5 * len(top5_runs)), sharex=True)
    if len(top5_runs) == 1:
        axes = [axes]
    for ax, run in zip(axes, top5_runs):
        vout = np.asarray(np.load(run.run_dir / f"0_vout_val_epoch{epoch}.npy"), dtype=float)
        probs = softmax_rows(vout / float(run.temp))
        masses = []
        for i, ctx_row in enumerate(run.val_ctx):
            support = run.support_map.get(tuple(int(v) for v in ctx_row.tolist()), run.unigram_support)
            masses.append(float(np.sum(probs[i][support])))
        ax.hist(masses, bins=30, color="tab:blue", alpha=0.85)
        ax.set_ylabel(run.run_name)
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("support mass on valid next-token set")
    axes[0].set_title("Epoch-20 Support-Mass Distributions for Top 5 Runs")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_low_mass_contexts(out_path: Path, low_rows: Sequence[Mapping[str, object]], selected_runs: Sequence[str]) -> None:
    top_by_run: Dict[str, List[Mapping[str, object]]] = {run: [] for run in selected_runs}
    for row in low_rows:
        run = str(row["run"])
        if len(top_by_run[run]) < 6:
            top_by_run[run].append(row)
    fig, axes = plt.subplots(len(selected_runs), 1, figsize=(14, 2.7 * len(selected_runs)))
    if len(selected_runs) == 1:
        axes = [axes]
    for ax, run in zip(axes, selected_runs):
        rows = top_by_run[run]
        labels = [str(r["context"]) for r in rows]
        values = [float(r["mean_support_mass"]) for r in rows]
        ax.barh(np.arange(len(rows)), values, color="tab:red", alpha=0.85)
        ax.set_yticks(np.arange(len(rows)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0.0, max(0.4, max(values) * 1.15 if values else 0.4))
        ax.set_title(run)
        ax.grid(axis="x", alpha=0.25)
    axes[-1].set_xlabel("mean support mass")
    fig.suptitle("Lowest-Support-Mass Contexts per Top Run", y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def build_report(
    out_path: Path,
    sweep_dir: Path,
    ranked_rows: Sequence[Mapping[str, object]],
    top_rows: Sequence[Mapping[str, object]],
    family_rows: Sequence[Mapping[str, object]],
    extras_by_run: Mapping[str, Mapping[str, float]],
) -> None:
    top = top_rows[0]
    families_by_run: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    for row in family_rows:
        families_by_run[str(row["run"])].append((str(row["family"]), int(row["count"])))

    lines = []
    lines.append("# 32-vocab one-hot CE top-5 completed-run analysis")
    lines.append("")
    lines.append("## Selection")
    lines.append(f"- Sweep: `{sweep_dir.name}`")
    lines.append("- Population: runs from March 26, 2026 with completed `epoch 20` and saved `0_vout_val_epoch20.npy`")
    lines.append("- Ranking criterion: corrected `support_mass_mean` at epoch 20, then `support_acc`, then `exact_acc`")
    lines.append(f"- Selected runs: `{', '.join(str(r['run']) for r in top_rows)}`")
    lines.append("")
    lines.append("## Top Run")
    lines.append(f"- Run: `{top['run']}`")
    lines.append(f"- Epoch-20 support_mass_mean: `{float(top['support_mass_mean']):.6f}`")
    lines.append(f"- Epoch-20 support_acc: `{float(top['support_acc']):.6f}`")
    lines.append(f"- Epoch-20 exact_acc: `{float(top['exact_acc']):.6f}`")
    lines.append(f"- Epoch-20 true_token_prob_mean: `{float(top['true_token_prob_mean']):.6f}`")
    lines.append(f"- Epoch-20 soft_ce: `{float(top['soft_ce']):.6f}`")
    lines.append("")
    lines.append("## High-level findings")
    lines.append(f"- The best completed epoch-20 run by corrected support mass is `{top['run']}` with `support_mass_mean = {float(top['support_mass_mean']):.3f}`.")
    lines.append(f"- Across the selected top 5, epoch-20 support mass spans `{min(float(r['support_mass_mean']) for r in top_rows):.3f}` to `{max(float(r['support_mass_mean']) for r in top_rows):.3f}`.")
    best_temp_rows = Counter(float(r['temp']) for r in top_rows)
    lines.append(f"- Temperature mix in the top 5: `{dict(best_temp_rows)}`.")
    lines.append("- Error mass remains concentrated in a few recurring families: subject agreement (`are` for `am`), early question punctuation, bad terminal transitions, and `is` replacing object nouns.")
    lines.append("- This report uses corrected support mass over all valid next tokens; it does not use the old single-target `val_qmass_mean` from the original sweep summaries.")
    lines.append("")
    lines.append("## Figure bundle")
    lines.append("- `figure1_epoch20_ranking.png`: top-5 epoch-20 ranking by support mass, support accuracy, exact accuracy")
    lines.append("- `figure2_metric_space.png`: epoch-20 metric-space scatter for all completed runs, top 5 highlighted")
    lines.append("- `figure3_top5_training_curves.png`: support mass, support accuracy, and true-token probability across epochs for the top 5")
    lines.append("- `figure4_error_families.png`: epoch-20 invalid-prediction family breakdown for the top 5")
    lines.append("- `figure5_support_mass_histograms.png`: validation support-mass distributions for the top 5")
    lines.append("- `figure6_low_mass_contexts.png`: lowest-support-mass contexts for each selected run")
    lines.append("")
    lines.append("## Top 5 epoch-20 runs")
    for row in top_rows:
        run = str(row["run"])
        extra = extras_by_run[run]
        lines.append(
            f"- `{run}`: support_mass_mean `{float(row['support_mass_mean']):.6f}`, support_acc `{float(row['support_acc']):.6f}`, exact_acc `{float(row['exact_acc']):.6f}`, true_token_prob_mean `{float(row['true_token_prob_mean']):.6f}`, invalid_count `{int(extra['invalid_count'])}`"
        )
    lines.append("")
    lines.append("## Dominant error families by run")
    for row in top_rows:
        run = str(row["run"])
        fams = sorted(families_by_run[run], key=lambda x: x[1], reverse=True)[:4]
        fam_text = ", ".join(f"{fam}={cnt}" for fam, cnt in fams) if fams else "none"
        lines.append(f"- `{run}`: {fam_text}")

    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    trainer = load_trainer_constants(args.trainer_script)
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = [
        p for p in sorted(args.sweep_dir.iterdir())
        if p.is_dir() and (p / f"0_epoch_summary_epoch{args.epoch}.json").exists() and (p / f"0_vout_val_epoch{args.epoch}.npy").exists()
    ]
    if not run_dirs:
        raise RuntimeError(f"No completed runs with epoch {args.epoch} artifacts found in {args.sweep_dir}")

    artifacts = [load_run_artifacts(run_dir, trainer) for run_dir in run_dirs]
    epoch20_rows = []
    for run in artifacts:
        metrics = evaluate_epoch(run, args.epoch)
        epoch20_rows.append(
            {
                "run": run.run_name,
                "epoch": metrics.epoch,
                "temp": run.temp,
                "support_mass_mean": metrics.support_mass_mean,
                "support_acc": metrics.support_acc,
                "true_token_prob_mean": metrics.true_token_prob_mean,
                "exact_acc": metrics.exact_acc,
                "soft_ce": metrics.soft_ce,
                "entropy_bits_mean": metrics.entropy_bits_mean,
                "unseen_contexts": metrics.unseen_contexts,
            }
        )

    epoch20_rows.sort(
        key=lambda r: (
            float(r["support_mass_mean"]),
            float(r["support_acc"]),
            float(r["exact_acc"]),
        ),
        reverse=True,
    )
    top_rows = epoch20_rows[: int(args.top_k)]
    selected_runs = [str(r["run"]) for r in top_rows]
    selected_artifacts = [run for run in artifacts if run.run_name in set(selected_runs)]
    selected_artifacts.sort(key=lambda run: selected_runs.index(run.run_name))

    selection_summary = {
        "sweep_dir": str(args.sweep_dir.resolve()),
        "epoch": int(args.epoch),
        "ranking_metric": "support_mass_mean_epoch20",
        "population_size": len(epoch20_rows),
        "selected_runs": selected_runs,
        "selected_metrics": top_rows,
    }
    (out_dir / "selection_summary.json").write_text(json.dumps(selection_summary, indent=2))
    write_csv(
        out_dir / "epoch20_ranking_support_mass.csv",
        ["run", "epoch", "temp", "support_mass_mean", "support_acc", "true_token_prob_mean", "exact_acc", "soft_ce", "entropy_bits_mean", "unseen_contexts"],
        epoch20_rows,
    )

    top5_curves: Dict[str, List[EpochMetrics]] = {}
    family_rows_all: List[Dict[str, object]] = []
    low_rows_all: List[Dict[str, object]] = []
    extras_by_run: Dict[str, Mapping[str, float]] = {}
    for run in selected_artifacts:
        curve = []
        for summary_path in sorted(run.run_dir.glob("0_epoch_summary_epoch*.json")):
            epoch = int(summary_path.stem.split("epoch")[-1])
            vout_path = run.run_dir / f"0_vout_val_epoch{epoch}.npy"
            if not vout_path.exists():
                continue
            curve.append(evaluate_epoch(run, epoch))
        curve.sort(key=lambda m: m.epoch)
        top5_curves[run.run_name] = curve

        family_rows, low_rows, extras = summarize_run_failures(run, trainer, args.epoch)
        family_rows_all.extend(family_rows)
        low_rows_all.extend(low_rows)
        extras_by_run[run.run_name] = extras

    write_csv(
        out_dir / "top5_error_family_breakdown.csv",
        ["run", "epoch", "family", "count"],
        family_rows_all,
    )
    write_csv(
        out_dir / "top5_low_mass_contexts.csv",
        ["run", "epoch", "context", "mean_support_mass", "count"],
        low_rows_all,
    )

    plot_epoch20_ranking(out_dir / "figure1_epoch20_ranking.png", epoch20_rows)
    plot_metric_space(out_dir / "figure2_metric_space.png", epoch20_rows, selected_runs)
    plot_training_curves(out_dir / "figure3_top5_training_curves.png", top5_curves)
    plot_error_families(out_dir / "figure4_error_families.png", family_rows_all, selected_runs)
    plot_support_mass_histograms(out_dir / "figure5_support_mass_histograms.png", selected_artifacts, args.epoch)
    plot_low_mass_contexts(out_dir / "figure6_low_mass_contexts.png", low_rows_all, selected_runs)
    build_report(out_dir / "report.md", args.sweep_dir, epoch20_rows, top_rows, family_rows_all, extras_by_run)

    print(f"analysis_dir={out_dir}")
    print(f"selected_runs={selected_runs}")


if __name__ == "__main__":
    main()
