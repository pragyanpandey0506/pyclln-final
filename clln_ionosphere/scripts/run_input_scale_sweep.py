#!/usr/bin/env python3
"""
run_input_scale_sweep.py
------------------------
Sweeps input_scale for the best 5 hyperparameter configs to show how
accuracy degrades as input voltages shrink toward a linear regime.

Input scale S means ionosphere features in [-1,1] are mapped to [-S, S].
- Very small S (0.01): VGS barely changes → MOSFET acts linear → bad performance
- Optimal S (~5.0): full nonlinear MOSFET regime → best performance
- Large S: clips or saturates → eventually degrades

**Fair comparison mode (default)**: δ and margin scale linearly with
input_scale so the loss landscape stays proportional to the voltage regime.
Gamma stays fixed (it's a learning rate, not a voltage threshold).
Base hyperparameters were tuned at scale=5.0; at scale S they become:
    γ_S = γ_base  (unchanged)
    δ_S = δ_base × (S / 5.0)
    m_S = m_base × (S / 5.0)

Top-5 configs (at base scale=5.0):
  1. hinge   γ=1.0 δ=0.05 init=7  margin=0.02
  2. hinge   γ=3.0 δ=0.05 init=7  margin=0.02
  3. hinge   γ=2.0 δ=0.05 init=7  margin=0.02
  4. hinge   γ=1.5 δ=0.05 init=7  margin=0.02
  5. sq_hinge γ=0.3 δ=0.05 init=6  margin=0.02

Usage:
  python run_input_scale_sweep.py                  # run everything + plot
  python run_input_scale_sweep.py --plot-only DIR  # just regenerate plot
  python run_input_scale_sweep.py --scales 0.01 0.1 0.5 1.0 5.0
  python run_input_scale_sweep.py --no-scale-hparams  # old mode: fixed γ/δ/m
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR       = Path(__file__).resolve().parent
PROJECT_DIR      = SCRIPT_DIR.parent
TRAIN_SCRIPT     = SCRIPT_DIR / "train_iono.py"
DEVICE_LIB_NMOS  = str(PROJECT_DIR / "device_model" / "nmos_lvl1_ald1106.lib")
IONO_DIR         = str(PROJECT_DIR / "data" / "ionosphere")

# ── Default input scales to sweep ────────────────────────────────────────────
# Fine steps at the small end; the "linear floor" appears around 0.01–0.05.
DEFAULT_SCALES = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]

# ── True best 5 configs across ALL sweeps at BASE_SCALE ──────────────────────
# All: gate=source, body=floating
# γ, δ, margin values below are the base values tuned at BASE_SCALE.
BASE_SCALE = 5.0

TOP5_CONFIGS = [
    {
        "name":     "hinge_g1.0_d0.05_i7",
        "label":    "hinge γ=1.0 δ=0.05 init=7",
        "loss":     "hinge",    "gamma": 1.0, "delta": 0.05, "margin": 0.02, "init": "7.0",
    },
    {
        "name":     "hinge_g3.0_d0.05_i7",
        "label":    "hinge γ=3.0 δ=0.05 init=7",
        "loss":     "hinge",    "gamma": 3.0, "delta": 0.05, "margin": 0.02, "init": "7.0",
    },
    {
        "name":     "hinge_g2.0_d0.05_i7",
        "label":    "hinge γ=2.0 δ=0.05 init=7",
        "loss":     "hinge",    "gamma": 2.0, "delta": 0.05, "margin": 0.02, "init": "7.0",
    },
    {
        "name":     "hinge_g1.5_d0.05_i7",
        "label":    "hinge γ=1.5 δ=0.05 init=7",
        "loss":     "hinge",    "gamma": 1.5, "delta": 0.05, "margin": 0.02, "init": "7.0",
    },
    {
        "name":     "sqh_g0.3_d0.05_i6",
        "label":    "sq_hinge γ=0.3 δ=0.05 init=6",
        "loss":     "sq_hinge", "gamma": 0.3, "delta": 0.05, "margin": 0.02, "init": "6.0",
    },
]

COLORS = ["#2563EB", "#16A34A", "#EA580C", "#9333EA", "#DC2626"]
MARKERS = ["o", "s", "^", "D", "v"]


# ── Runner ────────────────────────────────────────────────────────────────────

def run_one(cfg: dict, scale: float, seed: int, sweep_dir: Path,
            epochs: int = 100, scale_hparams: bool = True) -> dict:
    run_name = f"{cfg['name']}_scale{scale:.4g}_seed{seed}"
    run_dir  = sweep_dir / "runs" / run_name

    # Skip if already done (only skip if run has all epochs)
    va_file = run_dir / "0_val_acc.npy"
    if va_file.exists():
        va = np.load(va_file)
        if len(va) > 0 and len(va) >= epochs:
            print(f"  [skip] {run_name} already exists (final={float(va[-1]):.4f})", flush=True)
            return _load_result(cfg, scale, seed, run_dir, va)

    # Scale δ and margin linearly with input_scale so the loss landscape
    # stays proportional to the voltage regime. Gamma stays fixed.
    ratio = scale / BASE_SCALE
    gamma = cfg["gamma"]
    if scale_hparams:
        delta  = cfg["delta"]  * ratio
        margin = cfg["margin"] * ratio
    else:
        delta  = cfg["delta"]
        margin = cfg["margin"]

    cmd = [
        sys.executable, str(TRAIN_SCRIPT), str(seed),
        "--epochs",        str(epochs),
        "--dataset",       "ionosphere",
        "--ionosphere-dir", IONO_DIR,
        "--device-mode",   "subckt",
        "--device-lib",    DEVICE_LIB_NMOS,
        "--device-model",  "ncg",
        "--body-tie",      "floating",
        "--gate-ref",      "source",
        "--loss",          cfg["loss"],
        "--gamma",         str(gamma),
        "--delta",         str(delta),
        "--margin",        str(margin),
        "--input-scale",   str(scale),
        "--vg-init",       "fixed",
        "--vg-init-fixed", cfg["init"],
        "--vg-clip-lo",    "0.4",
        "--vg-clip-hi",    "8.0",
        "--vplus",         "0.45",
        "--vminus",        "0.0",
    ]

    env = {**os.environ, "RUN_DIR": str(run_dir)}

    print(f"\n{'─'*65}", flush=True)
    print(f"  Config : {cfg['label']}  |  scale={scale}", flush=True)
    print(f"  Scaled : γ={gamma:.4g}  δ={delta:.4g}  m={margin:.4g}"
          f"  (ratio={ratio:.4g})", flush=True)
    print(f"  Run    : {run_name}", flush=True)
    print(f"  Cmd    : {' '.join(cmd[2:])}", flush=True)

    t0 = time.time()
    proc = subprocess.run(cmd, env=env)
    elapsed = time.time() - t0

    if va_file.exists():
        va = np.load(va_file)
        return _load_result(cfg, scale, seed, run_dir, va, elapsed=elapsed,
                            returncode=proc.returncode)
    else:
        print(f"  [warn] val_acc not found after run (rc={proc.returncode})", flush=True)
        return {
            "config": cfg["name"], "label": cfg["label"],
            "scale": scale, "seed": seed, "run_dir": str(run_dir),
            "best_val_acc": float("nan"), "final_val_acc": float("nan"),
            "val_acc_curve": [], "elapsed_s": round(elapsed, 1),
            "returncode": proc.returncode,
        }


def _load_result(cfg, scale, seed, run_dir, va, elapsed=0.0, returncode=0):
    run_dir = Path(run_dir)
    # Power arrays (µW per test sample, one value per epoch)
    p_supply = _try_load(run_dir / "0_power_supply_uw.npy")
    p_total  = _try_load(run_dir / "0_power_total_uw.npy")
    return {
        "config":              cfg["name"],
        "label":               cfg["label"],
        "scale":               scale,
        "seed":                seed,
        "run_dir":             str(run_dir),
        "best_val_acc":        float(va.max()),
        "final_val_acc":       float(va[-1]),
        "val_acc_curve":       va.tolist(),
        # Power at final epoch (µW); NaN if not logged
        "power_supply_uw":     float(p_supply[-1]) if p_supply is not None and len(p_supply) > 0 else float("nan"),
        "power_total_uw":      float(p_total[-1])  if p_total  is not None and len(p_total)  > 0 else float("nan"),
        "power_supply_uw_curve": p_supply.tolist() if p_supply is not None else [],
        "power_total_uw_curve":  p_total.tolist()  if p_total  is not None else [],
        "elapsed_s":           round(elapsed, 1),
        "returncode":          returncode,
    }


def _try_load(path):
    try:
        arr = np.load(path)
        return arr if len(arr) > 0 else None
    except Exception:
        return None


# ── Plotting ──────────────────────────────────────────────────────────────────

def make_plot(results: list, sweep_dir: Path):
    """Generates three figures:
       1. input_scale_accuracy_power.png  — accuracy + power vs scale (shared x-axis)
       2. input_scale_learning_curves.png — small multiples grid
       3. input_scale_power_vs_acc.png    — power vs accuracy trade-off scatter
    """
    scales_all = sorted({r["scale"] for r in results})
    configs = TOP5_CONFIGS

    # ── Figure 1: accuracy + power vs scale ─────────────────────────────────
    fig, (ax_acc, ax_pow) = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    has_power = any(np.isfinite(r.get("power_total_uw", float("nan"))) for r in results)

    for i, cfg in enumerate(configs):
        cfg_results = sorted([r for r in results if r["config"] == cfg["name"]],
                             key=lambda r: r["scale"])
        xs      = [r["scale"]       for r in cfg_results]
        ys_best = [r["best_val_acc"] for r in cfg_results]
        ys_fin  = [r["final_val_acc"] for r in cfg_results]
        ps      = [r.get("power_total_uw", float("nan")) for r in cfg_results]

        # Accuracy panel
        valid_acc = [(x, yb, yf) for x, yb, yf in zip(xs, ys_best, ys_fin)
                     if np.isfinite(yb) and np.isfinite(yf)]
        if valid_acc:
            xv, yb_v, yf_v = zip(*valid_acc)
            ax_acc.plot(xv, yb_v, color=COLORS[i], marker=MARKERS[i],
                        markersize=6, lw=1.8, label=cfg["label"])
            ax_acc.plot(xv, yf_v, color=COLORS[i], marker=MARKERS[i],
                        markersize=4, lw=0.8, ls="--", alpha=0.4)

        # Power panel
        valid_pow = [(x, p) for x, p in zip(xs, ps) if np.isfinite(p)]
        if valid_pow and has_power:
            xp, yp = zip(*valid_pow)
            ax_pow.plot(xp, yp, color=COLORS[i], marker=MARKERS[i],
                        markersize=6, lw=1.8, label=cfg["label"])

    ax_acc.set_xscale("log")
    ax_acc.set_ylabel("Validation accuracy", fontsize=10)
    ax_acc.set_title("Accuracy vs input scale\n(solid=best epoch, dashed=final)", fontsize=9)
    ax_acc.set_ylim(0.45, 1.02)
    ax_acc.axhline(0.9718, ls=":", lw=0.9, color="#6B7280", label="97.18% ref")
    ax_acc.axhline(0.9577, ls=":", lw=0.6, color="#9CA3AF")
    ax_acc.legend(fontsize=7, loc="lower right")
    ax_acc.tick_params(labelsize=8)
    for sp in ["top", "right"]: ax_acc.spines[sp].set_visible(False)

    # Linear regime annotation
    min_scale = min(scales_all)
    if min_scale <= 0.02:
        lo, hi = min_scale * 0.7, min_scale * 2.5
        ax_acc.axvspan(lo, hi, alpha=0.07, color="#F59E0B")
        ax_acc.text(min_scale * 1.2, 0.49, "~linear\nregime",
                    fontsize=6, color="#B45309", ha="center", va="bottom")

    if has_power:
        ax_pow.set_xscale("log")
        ax_pow.set_xlabel("Input scale  (inputs ∈ [−S, S])", fontsize=10)
        ax_pow.set_ylabel("Inference power (µW / sample)", fontsize=10)
        ax_pow.set_title("Supply + input power vs input scale", fontsize=9)
        ax_pow.legend(fontsize=7, loc="upper left")
        ax_pow.tick_params(labelsize=8)
        for sp in ["top", "right"]: ax_pow.spines[sp].set_visible(False)
    else:
        ax_pow.text(0.5, 0.5,
                    "Power not logged in this sweep.\n"
                    "Reruns with the updated train_iono.py will populate this panel.",
                    ha="center", va="center", transform=ax_pow.transAxes,
                    fontsize=9, color="#6B7280",
                    bbox=dict(boxstyle="round,pad=0.5", fc="#F9FAFB", ec="#D1D5DB"))
        ax_pow.set_xlabel("Input scale  (inputs ∈ [−S, S])", fontsize=10)
        ax_pow.tick_params(labelsize=8)

    fig.tight_layout()
    main_path = sweep_dir / "input_scale_accuracy_power.png"
    fig.savefig(main_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] accuracy + power → {main_path}", flush=True)

    # ── Figure 2: power vs accuracy scatter ──────────────────────────────────
    if has_power:
        fig3, ax3 = plt.subplots(figsize=(7, 5))
        for i, cfg in enumerate(configs):
            cfg_results = [r for r in results if r["config"] == cfg["name"]]
            pts = [(r["best_val_acc"], r.get("power_total_uw", float("nan")), r["scale"])
                   for r in cfg_results
                   if np.isfinite(r["best_val_acc"]) and np.isfinite(r.get("power_total_uw", float("nan")))]
            if not pts: continue
            accs, pows, scales = zip(*pts)
            sc = ax3.scatter(pows, accs, c=np.log10(scales), cmap="plasma",
                             marker=MARKERS[i], s=60, label=cfg["label"],
                             vmin=np.log10(min(scales_all)), vmax=np.log10(max(scales_all)))
            # Connect dots in scale order
            order = np.argsort(scales)
            ax3.plot([pows[j] for j in order], [accs[j] for j in order],
                     color=COLORS[i], lw=0.7, alpha=0.5)
        cbar = fig3.colorbar(sc, ax=ax3)
        cbar.set_label("log₁₀(input scale)", fontsize=8)
        ax3.set_xlabel("Inference power (µW / sample)", fontsize=10)
        ax3.set_ylabel("Best val accuracy", fontsize=10)
        ax3.set_title("Power–accuracy trade-off by input scale\n(color = log₁₀ scale, shape = config)", fontsize=9)
        ax3.legend(fontsize=7, loc="lower right")
        ax3.tick_params(labelsize=8)
        for sp in ["top", "right"]: ax3.spines[sp].set_visible(False)
        fig3.tight_layout()
        scatter_path = sweep_dir / "input_scale_power_vs_acc.png"
        fig3.savefig(scatter_path, dpi=160, bbox_inches="tight")
        plt.close(fig3)
        print(f"[plot] power vs accuracy → {scatter_path}", flush=True)

    # ── Figure 3: small-multiple learning curves ──────────────────────────────
    n_scales  = len(scales_all)
    n_configs = len(configs)
    fig2, axes2 = plt.subplots(
        n_configs, n_scales,
        figsize=(max(8, 2.2 * n_scales), max(5, 1.6 * n_configs)),
        squeeze=False,
    )
    fig2.suptitle("Learning curves — rows=config, cols=input scale", fontsize=9)

    for row, cfg in enumerate(configs):
        for col, scale in enumerate(scales_all):
            axi = axes2[row][col]
            match = [r for r in results
                     if r["config"] == cfg["name"] and r["scale"] == scale]
            if not match or not match[0]["val_acc_curve"]:
                axi.text(0.5, 0.5, "N/A", ha="center", va="center",
                         transform=axi.transAxes, fontsize=6, color="#9CA3AF")
                axi.set_xticks([]); axi.set_yticks([])
            else:
                va   = np.array(match[0]["val_acc_curve"])
                best = match[0]["best_val_acc"]
                pw   = match[0].get("power_total_uw", float("nan"))
                color = COLORS[row]
                axi.plot(np.arange(1, len(va)+1), va, color=color, lw=0.9)
                axi.axhline(0.9718, ls="--", lw=0.4, color="#9CA3AF")
                axi.set_ylim(0.40, 1.02)
                axi.set_xlim(1, len(va))
                label = f"{best:.3f}"
                if np.isfinite(pw):
                    label += f"\n{pw:.1f}µW"
                axi.text(0.97, 0.04, label, transform=axi.transAxes,
                         ha="right", va="bottom", fontsize=4.5,
                         color=color, fontweight="bold")
            if row == 0:
                axi.set_title(f"S={scale:.3g}", fontsize=6, pad=2)
            if col == 0:
                axi.set_ylabel(cfg["label"][:18], fontsize=5, labelpad=2)
            axi.tick_params(labelsize=4, length=2, pad=1)
            for sp in ["top", "right"]: axi.spines[sp].set_visible(False)

    fig2.tight_layout(rect=[0, 0, 1, 0.96])
    curves_path = sweep_dir / "input_scale_learning_curves.png"
    fig2.savefig(curves_path, dpi=160, bbox_inches="tight")
    plt.close(fig2)
    print(f"[plot] learning curves → {curves_path}", flush=True)

    return main_path, curves_path


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--plot-only", type=str, default=None,
                   metavar="SWEEP_DIR",
                   help="Skip training; just regenerate plots from an existing sweep dir.")
    p.add_argument("--scales", type=float, nargs="+", default=DEFAULT_SCALES,
                   help="Input scales to sweep (default: 0.01 0.05 0.1 0.2 0.5 1.0 2.0 5.0 10.0)")
    p.add_argument("--configs", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                   help="Which of the top-5 configs to run (0-indexed, default all)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--no-scale-hparams", action="store_true",
                   help="Disable linear scaling of γ/δ/margin with input_scale (old behavior)")
    return p.parse_args()


def main():
    args = parse_args()

    if args.plot_only:
        sweep_dir = Path(args.plot_only)
        results_file = sweep_dir / "results.json"
        if not results_file.exists():
            print(f"ERROR: {results_file} not found", file=sys.stderr)
            sys.exit(1)
        results = json.loads(results_file.read_text())
        make_plot(results, sweep_dir)
        return

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    sweep_dir = SCRIPT_DIR / "results" / "sweeps" / f"input_scale_sweep_{ts}"
    (sweep_dir / "runs").mkdir(parents=True, exist_ok=True)

    scales  = sorted(args.scales)
    configs = [TOP5_CONFIGS[i] for i in args.configs]
    scale_hparams = not args.no_scale_hparams

    print(f"\n{'='*70}", flush=True)
    print(f"  Input Scale Sweep", flush=True)
    print(f"  Sweep dir : {sweep_dir}", flush=True)
    print(f"  Scales    : {scales}", flush=True)
    print(f"  Configs   : {[c['name'] for c in configs]}", flush=True)
    print(f"  Scale γ/δ/m: {scale_hparams}  (base_scale={BASE_SCALE})", flush=True)
    print(f"  Total runs: {len(scales) * len(configs)}", flush=True)
    print(f"{'='*70}", flush=True)

    all_results = []
    results_file = sweep_dir / "results.json"

    for scale in scales:
        for cfg in configs:
            result = run_one(cfg, scale, args.seed, sweep_dir,
                             epochs=args.epochs, scale_hparams=scale_hparams)
            all_results.append(result)
            # Save after every run so we can plot partial results
            results_file.write_text(json.dumps(all_results, indent=2))
            va = result.get("final_val_acc", float("nan"))
            best = result.get("best_val_acc", float("nan"))
            print(f"  → scale={scale:<6.3g}  {cfg['name']:25s}  "
                  f"best={best:.4f}  final={va:.4f}", flush=True)

    print(f"\n[done] {len(all_results)} runs completed", flush=True)
    make_plot(all_results, sweep_dir)
    print(f"\nResults: {results_file}", flush=True)


if __name__ == "__main__":
    main()
