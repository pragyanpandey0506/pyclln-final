#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def load_trainer_module(script_path: Path):
    spec = importlib.util.spec_from_file_location("vocab16_onehot_trainer", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load trainer module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot support-accuracy-vs-epoch for all runs in a 16-vocab one-hot CE sweep."
    )
    p.add_argument(
        "--sweep-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "results_language_16_onehotce" / "sweeps" / "top60_from_softce_20260325-122210",
    )
    p.add_argument(
        "--trainer-script",
        type=Path,
        default=Path(__file__).resolve().parent / "clln_lang_trainer_onehot_ce.py",
    )
    p.add_argument(
        "--out-png",
        type=Path,
        default=None,
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=None,
    )
    return p.parse_args()


def get_run_dirs(sweep_dir: Path) -> List[Path]:
    return sorted([p for p in sweep_dir.iterdir() if p.is_dir()])


def get_run_epoch_files(run_dir: Path) -> List[Tuple[int, Path]]:
    out: List[Tuple[int, Path]] = []
    for p in run_dir.glob("0_vout_val_epoch*.npy"):
        epoch = int(p.stem.split("epoch")[-1])
        out.append((epoch, p))
    out.sort(key=lambda item: item[0])
    return out


def build_grammar_support_map(trainer, bit_v0: float, bit_v1: float) -> Dict[Tuple[int, ...], Set[int]]:
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


def decode_val_contexts(trainer, val_x: np.ndarray, bit_v0: float, bit_v1: float) -> List[Tuple[int, ...]]:
    out: List[Tuple[int, ...]] = []
    for x in val_x:
        words = trainer.decode_context_bits_to_words(x, bit_v0=bit_v0, bit_v1=bit_v1)
        out.append(tuple(int(trainer.WORD_TO_ID[w]) for w in words))
    return out


def support_accuracy_from_logits(
    logits: np.ndarray,
    val_ctx: Sequence[Tuple[int, ...]],
    support_map: Dict[Tuple[int, ...], Set[int]],
) -> Tuple[float, int]:
    pred = np.argmax(np.asarray(logits, dtype=float), axis=1)
    hits = np.zeros(pred.shape[0], dtype=float)
    unseen = 0

    for i, (ctx, p) in enumerate(zip(val_ctx, pred)):
        valid = support_map.get(tuple(ctx))
        if valid is None:
            unseen += 1
            hits[i] = 0.0
            continue
        hits[i] = float(int(p) in valid)

    return float(np.mean(hits)), int(unseen)


def main() -> None:
    args = parse_args()
    sweep_dir = args.sweep_dir.resolve()
    trainer_script = args.trainer_script.resolve()
    out_png = args.out_png.resolve() if args.out_png else sweep_dir / "support_acc_vs_epoch_all_runs.png"
    out_csv = args.out_csv.resolve() if args.out_csv else sweep_dir / "support_acc_vs_epoch_all_runs.csv"

    if not sweep_dir.exists():
        raise FileNotFoundError(f"Sweep dir not found: {sweep_dir}")
    if not trainer_script.exists():
        raise FileNotFoundError(f"Trainer script not found: {trainer_script}")

    run_dirs = get_run_dirs(sweep_dir)
    if not run_dirs:
        raise RuntimeError(f"No run directories found in {sweep_dir}")

    trainer = load_trainer_module(trainer_script)

    first_meta = json.loads((run_dirs[0] / "run_meta.json").read_text())
    dataset = first_meta["dataset"]
    bit_v0 = float(dataset["bit_v0"])
    bit_v1 = float(dataset["bit_v1"])
    support_map = build_grammar_support_map(trainer, bit_v0=bit_v0, bit_v1=bit_v1)

    rows: List[Dict[str, object]] = []
    series: Dict[str, List[Tuple[int, float]]] = {}

    for run_dir in run_dirs:
        run_name = run_dir.name
        val_x = np.load(run_dir / "val_x.npy")
        val_ctx = decode_val_contexts(trainer, val_x, bit_v0=bit_v0, bit_v1=bit_v1)
        points: List[Tuple[int, float]] = []
        for epoch, vout_path in get_run_epoch_files(run_dir):
            support_acc, unseen = support_accuracy_from_logits(
                logits=np.load(vout_path),
                val_ctx=val_ctx,
                support_map=support_map,
            )
            points.append((epoch, support_acc))
            rows.append(
                {
                    "run": run_name,
                    "epoch": epoch,
                    "support_acc": support_acc,
                    "unseen_contexts": unseen,
                }
            )
        if points:
            series[run_name] = points

    if not series:
        raise RuntimeError(f"No per-epoch validation logits found under {sweep_dir}")

    ranked = sorted(
        series.items(),
        key=lambda kv: max(v for _, v in kv[1]),
        reverse=True,
    )
    highlight = dict(ranked[:5])

    plt.figure(figsize=(14, 9))

    for run_name, points in ranked:
        x = [ep for ep, _ in points]
        y = [val for _, val in points]
        if run_name in highlight:
            plt.plot(x, y, linewidth=2.0, alpha=0.95, label=run_name)
        else:
            plt.plot(x, y, linewidth=0.9, alpha=0.35, color="#7f8c8d")

    plt.xlabel("Epoch")
    plt.ylabel("Support Accuracy")
    plt.title("16-Vocab One-Hot CE Sweep: Support Accuracy vs Epoch")
    plt.grid(True, alpha=0.25)
    plt.ylim(0.0, 1.0)
    plt.xlim(left=0)
    plt.legend(loc="lower right", fontsize=8, frameon=True, title="Top 5 by best support_acc")
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run", "epoch", "support_acc", "unseen_contexts"])
        writer.writeheader()
        writer.writerows(rows)

    best_run, best_points = ranked[0]
    best_epoch, best_support = max(best_points, key=lambda t: t[1])
    print(f"sweep_dir={sweep_dir}")
    print(f"out_png={out_png}")
    print(f"out_csv={out_csv}")
    print(f"runs={len(series)} support_contexts={len(support_map)}")
    print(f"best_run={best_run} best_epoch={best_epoch} best_support_acc={best_support:.10f}")


if __name__ == "__main__":
    main()
