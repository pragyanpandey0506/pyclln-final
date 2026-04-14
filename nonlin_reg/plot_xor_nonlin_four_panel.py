#!/usr/bin/env python3
"""
Build a single-column 2x2 figure that combines:
  - XOR accuracy for the best runs
  - XOR output MSE for the best runs
  - nonlinear-regression MSE
  - nonlinear-regression fit snapshots

Defaults point to the saved artifacts currently used for the paper figures.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_XOR_ROOT = ROOT / "xor" / "results" / "pnas_4x4_5runs_10000ep"
DEFAULT_NONLIN_RUN = Path(__file__).resolve().parent / "results" / "pnas_4x4_gamma0p4_truepts_25000ep"
DEFAULT_OUT_PREFIX = DEFAULT_NONLIN_RUN / "xor_nonlin_2x2_singlecol"
DEFAULT_SNAPSHOTS = [0, 100, 1000, 10000, 25000]
DEFAULT_XOR_INCLUDE_RUNS = ["seed-3"]


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Make a combined XOR/nonlinear-regression 2x2 figure")
    p.add_argument("--xor-root", type=Path, default=DEFAULT_XOR_ROOT)
    p.add_argument("--nonlin-run", type=Path, default=DEFAULT_NONLIN_RUN)
    p.add_argument("--out-prefix", type=Path, default=DEFAULT_OUT_PREFIX)
    p.add_argument("--top-k-xor", type=int, default=1)
    p.add_argument("--xor-include-runs", nargs="+", default=DEFAULT_XOR_INCLUDE_RUNS)
    p.add_argument("--snapshots", type=int, nargs="+", default=DEFAULT_SNAPSHOTS)
    p.add_argument("--width-in", type=float, default=3.35)
    p.add_argument("--height-in", type=float, default=5.0)
    return p.parse_args()


def _load_nonlin_dataset(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    meta = json.loads((run_dir / "run_meta.json").read_text())
    dataset = meta["dataset"]
    return np.asarray(dataset["vins"], dtype=float), np.asarray(dataset["vous"], dtype=float)


def _load_xor_targets(run_dir: Path) -> np.ndarray:
    meta = json.loads((run_dir / "run_meta.json").read_text())
    l0 = float(meta["dataset"]["L0"])
    return np.array([0.0, l0, l0, 0.0], dtype=float)


def _load_xor_runs(xor_root: Path) -> list[dict[str, object]]:
    runs: list[dict[str, object]] = []
    for run_dir in sorted(p for p in xor_root.iterdir() if p.is_dir() and (p / "run_meta.json").exists()):
        acc = np.load(run_dir / "0_acc.npy")
        outputs = np.load(run_dir / "0_outputs.npy", allow_pickle=True)
        preds = np.stack(outputs).astype(float)
        targets = _load_xor_targets(run_dir)
        mse = ((preds - targets.reshape(1, -1)) ** 2).mean(axis=1)

        final_acc = float(acc[-1])
        final_mse = float(mse[-1])
        first_perfect = next((idx for idx, value in enumerate(acc) if value >= 1.0), int(1e9))
        runs.append(
            {
                "run_dir": run_dir,
                "label": run_dir.name,
                "acc": acc,
                "mse": mse,
                "final_acc": final_acc,
                "final_mse": final_mse,
                "first_perfect": first_perfect,
            }
        )

    if not runs:
        raise FileNotFoundError(f"No XOR run directories with run_meta.json found under {xor_root}")

    runs.sort(
        key=lambda item: (
            -float(item["final_acc"]),
            int(item["first_perfect"]),
            float(item["final_mse"]),
            str(item["label"]),
        )
    )
    return runs


def _validate_snapshots(preds_hist: np.ndarray, snapshots: list[int]) -> list[int]:
    max_epoch = preds_hist.shape[0] - 1
    out: list[int] = []
    for epoch in snapshots:
        if epoch < 0 or epoch > max_epoch:
            raise ValueError(f"snapshot epoch {epoch} is out of range 0..{max_epoch}")
        out.append(int(epoch))
    return out


def _style_axes(ax: plt.Axes) -> None:
    ax.grid(True, alpha=0.25, linewidth=0.4)
    ax.tick_params(labelsize=6, width=0.6, length=2.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)


def _set_xor_xticks(ax: plt.Axes, ticks: list[int]) -> None:
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(int(x)) for x in ticks])
    labels = ax.get_xticklabels()
    if labels:
        labels[0].set_ha("left")
        labels[-1].set_ha("right")


def _annotate_panel(ax: plt.Axes, tag: str) -> None:
    ax.text(
        0.0,
        1.03,
        tag,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        va="bottom",
        ha="left",
        clip_on=False,
    )


def main() -> None:
    args = _parse_args()
    xor_root = args.xor_root.resolve()
    nonlin_run = args.nonlin_run.resolve()
    out_prefix = args.out_prefix.resolve()
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    allowed_xor_labels = set(str(x) for x in args.xor_include_runs)
    xor_runs = [run for run in _load_xor_runs(xor_root) if str(run["label"]) in allowed_xor_labels]
    xor_runs = xor_runs[: max(1, int(args.top_k_xor))]
    if not xor_runs:
        raise FileNotFoundError(f"No selected XOR runs found under {xor_root} for labels={sorted(allowed_xor_labels)}")
    nonlin_mse = np.load(nonlin_run / "mse_history.npy")
    nonlin_preds = np.load(nonlin_run / "preds_history.npy")
    nonlin_x, nonlin_y = _load_nonlin_dataset(nonlin_run)
    snapshots = _validate_snapshots(nonlin_preds, [int(x) for x in args.snapshots])

    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.titlesize": 7,
            "axes.labelsize": 7,
            "legend.fontsize": 5.5,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(float(args.width_in), float(args.height_in)))
    fig.subplots_adjust(left=0.16, right=0.985, bottom=0.11, top=0.97, wspace=0.42, hspace=0.30)

    xor_colors = plt.cm.tab10(np.linspace(0, 1, len(xor_runs), endpoint=False))
    nonlin_colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(snapshots)))
    xor_max_epoch = max(np.asarray(run["acc"]).size for run in xor_runs) - 1
    xor_xticks = [0, 5000, 10000] if xor_max_epoch >= 10000 else [0, xor_max_epoch // 2, xor_max_epoch]

    ax = axes[0, 0]
    for color, run in zip(xor_colors, xor_runs):
        mse = np.clip(np.asarray(run["mse"], dtype=float), np.finfo(float).tiny, None)
        ax.plot(np.arange(1, mse.size + 1), mse, color=color, linewidth=1.1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_yscale("log")
    ax.set_xlim(0, max(np.asarray(run["mse"]).size for run in xor_runs) + 350)
    _set_xor_xticks(ax, xor_xticks)
    _style_axes(ax)
    _annotate_panel(ax, "A")

    ax = axes[0, 1]
    for color, run in zip(xor_colors, xor_runs):
        acc = np.asarray(run["acc"], dtype=float)
        ax.plot(np.arange(acc.size), acc, color=color, linewidth=1.1, label=str(run["label"]))
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(0, xor_max_epoch + 350)
    _set_xor_xticks(ax, xor_xticks)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(frameon=False, loc="lower right", handlelength=1.8)
    _style_axes(ax)
    _annotate_panel(ax, "B")

    ax = axes[1, 0]
    ax.plot(
        np.arange(nonlin_mse.size),
        np.clip(nonlin_mse, np.finfo(float).tiny, None),
        color="black",
        linewidth=1.1,
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")
    ax.set_yscale("log")
    ax.set_xlim(0, nonlin_mse.size - 1)
    _style_axes(ax)
    _annotate_panel(ax, "C")

    ax = axes[1, 1]
    ax.scatter(nonlin_x, nonlin_y, s=14, facecolors="none", edgecolors="black", linewidths=0.7, label="Train data")
    for color, epoch in zip(nonlin_colors, snapshots):
        ax.plot(nonlin_x, nonlin_preds[epoch], color=color, linewidth=1.0, label=f"{epoch}")
    ax.set_xlabel("Input Voltage (V)")
    ax.set_ylabel("Output Voltage (V)")
    ax.legend(frameon=False, loc="lower right", title="Epoch", title_fontsize=5.5, handlelength=1.6)
    _style_axes(ax)
    _annotate_panel(ax, "D")

    png_path = out_prefix.with_suffix(".png")
    pdf_path = out_prefix.with_suffix(".pdf")
    fig.savefig(png_path, dpi=400)
    fig.savefig(pdf_path)
    plt.close(fig)

    print(f"xor_root={xor_root}")
    print(f"selected_xor_runs={[str(run['label']) for run in xor_runs]}")
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
