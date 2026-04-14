#!/usr/bin/env python3
"""
run_hidden5_from_pruned.py
--------------------------
Extract pruned topologies from hidden5 sweep, retrain without pruning.

Multiple strategies:
A) Structured pruning: keep ALL direct inp→out edges (68) + hid→out (10) +
   out↔out (2) = 80 fixed, then select inp→hid edges to reach 125 total (50% of 250).
   Select inp→hid using VG values from baseline or pruning runs.
B) Dynamic pruning extractions: use stable topologies from rg20 runs.
C) Peak-epoch extractions: use topologies at accuracy peak from pruning runs.
"""
import argparse
import subprocess, sys, os, numpy as np, json
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
TOPO_DIR = PROJECT_DIR / "topologies"
DEVICE_DIR = PROJECT_DIR / "device_model"

HIDDEN5_TOPO = np.load(TOPO_DIR / "ionosphere_34_hidden5_maxconn.npz", allow_pickle=True)
SWEEP_RUNS = PROJECT_DIR / "results" / "sweeps" / "hidden5_prune_20260413-004513" / "runs"


def parse_args():
    p = argparse.ArgumentParser(
        description="Retrain hidden5 extracted/pruned topologies over a hyperparameter grid."
    )
    p.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Only run the specified topology labels (for example: struct_g3.0_ep39).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Maximum number of training runs to execute in parallel (default 2).",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=150,
        help="Epochs per training run (default 150).",
    )
    return p.parse_args()


args = parse_args()

ts = datetime.now().strftime("%Y%m%d-%H%M%S")
OUT = PROJECT_DIR / "results" / "sweeps" / f"hidden5_retrain_{ts}"
RUNS_DIR = OUT / "runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)

edges_D = HIDDEN5_TOPO["edges_D"]
edges_S = HIDDEN5_TOPO["edges_S"]
inputs = set(HIDDEN5_TOPO["input_nodes"].tolist())
outputs = set(HIDDEN5_TOPO["out_nodes"].tolist())
hidden = set(range(300, 305))

# Classify each edge
inp_hid_mask = np.array([(d in inputs and s in hidden) or (s in inputs and d in hidden) for d, s in zip(edges_D, edges_S)])
hid_out_mask = np.array([(d in hidden and s in outputs) or (s in hidden and d in outputs) for d, s in zip(edges_D, edges_S)])
inp_out_mask = np.array([(d in inputs and s in outputs) or (s in inputs and d in outputs) for d, s in zip(edges_D, edges_S)])
out_out_mask = np.array([d in outputs and s in outputs for d, s in zip(edges_D, edges_S)])

print(f"Edge breakdown: inp→hid={inp_hid_mask.sum()}, hid→out={hid_out_mask.sum()}, "
      f"inp→out={inp_out_mask.sum()}, out↔out={out_out_mask.sum()}")


def save_topo(active_mask, out_npz, label, meta_extra=None):
    """Save topology from edge mask. Returns edge count."""
    n = int(active_mask.sum())
    meta = {"note": "hidden5_pruned_retrain", "label": label, "active_edges": n}
    if meta_extra:
        meta.update(meta_extra)
    np.savez(
        out_npz,
        Nin=HIDDEN5_TOPO["Nin"], K=HIDDEN5_TOPO["K"],
        negref=HIDDEN5_TOPO["negref"], posref=HIDDEN5_TOPO["posref"],
        input_nodes=HIDDEN5_TOPO["input_nodes"],
        out_nodes=HIDDEN5_TOPO["out_nodes"],
        edges_D=edges_D[active_mask],
        edges_S=edges_S[active_mask],
        num_edges=np.array(n),
        meta=np.array(meta, dtype=object),
    )
    return n


def build_structured_topo(vg_source, label, n_inp_hid=45):
    """
    Structured pruning: keep all inp→out, hid→out, out↔out edges.
    Select top n_inp_hid inp→hid edges by VG value.
    """
    # Fixed edges: all non-inp_hid edges
    keep = hid_out_mask | inp_out_mask | out_out_mask  # 80 edges

    # Select top inp_hid edges by VG
    inp_hid_indices = np.where(inp_hid_mask)[0]
    inp_hid_vg = vg_source[inp_hid_indices]
    top_indices = inp_hid_indices[np.argsort(-inp_hid_vg)[:n_inp_hid]]

    active = keep.copy()
    active[top_indices] = True

    # Ensure each hidden node has at least some connections
    for h in hidden:
        h_edges = [i for i in inp_hid_indices if (edges_D[i] == h or edges_S[i] == h)]
        if not any(active[i] for i in h_edges) and h_edges:
            # Force at least 1 connection for this hidden node
            best = max(h_edges, key=lambda i: vg_source[i])
            active[best] = True

    topo_path = OUT / f"topo_{label}.npz"
    n = save_topo(active, topo_path, label, {"strategy": "structured", "n_inp_hid": int(active[inp_hid_mask].sum())})
    return topo_path, n


def build_extraction_topo(src_dir, epoch, label):
    """Extract topology from VG at given epoch."""
    vg = np.load(src_dir / f"0_vg_unique_epoch{epoch}.npy")
    active = vg > 0.41
    topo_path = OUT / f"topo_{label}.npz"
    n = save_topo(active, topo_path, label, {"source_run": src_dir.name, "source_epoch": epoch})
    return topo_path, n


# ── Strategy A: Structured pruning using baseline VG values ──────────────
print("\n=== Strategy A: Structured pruning ===")

# Use multiple baselines for VG-guided selection
baseline_vg_sources = [
    ("A_baseline_g3.0_ep100_pf0.0_ps9999_pi9999_rg0_losshinge_f7_s0", 38),   # peak=97.18%
    ("A_baseline_g1.0_ep100_pf0.0_ps9999_pi9999_rg0_losshinge_f7_s0", 52),   # peak=95.77%
]

topo_info = {}

for bname, peak_ep in baseline_vg_sources:
    d = SWEEP_RUNS / bname
    vg = np.load(d / f"0_vg_unique_epoch{peak_ep}.npy")
    label = f"struct_{bname.split('_')[2]}_ep{peak_ep+1}"
    topo_path, n = build_structured_topo(vg, label, n_inp_hid=45)
    topo_info[label] = {"path": topo_path, "n_active": n}
    print(f"  {label}: {n}/250 edges")

# Also try random inp→hid selection (3 random seeds)
for rseed in [42, 123, 7]:
    rng = np.random.RandomState(rseed)
    vg_random = np.ones(250) * 7.0  # uniform
    inp_hid_idx = np.where(inp_hid_mask)[0]
    vg_random[inp_hid_idx] = rng.random(len(inp_hid_idx))  # random ranking
    label = f"struct_rand_{rseed}"
    topo_path, n = build_structured_topo(vg_random, label, n_inp_hid=45)
    topo_info[label] = {"path": topo_path, "n_active": n}
    print(f"  {label}: {n}/250 edges")


# ── Strategy B: Stable dynamic pruning extractions ───────────────────────
print("\n=== Strategy B: Dynamic pruning (stable final) ===")

dyn_sources = [
    ("D_dynamic_g0.5_ep150_pf0.2_ps5_pi5_rg20_losshinge_f7_s2", 149),  # 126 edges, 87% final
    ("E_sqhinge_g0.3_ep150_pf0.2_ps5_pi5_rg30_losssq_hinge_f7_s2", 47),  # 112 edges, peak 95.77%
]

for rname, ep in dyn_sources:
    d = SWEEP_RUNS / rname
    if not d.exists():
        print(f"  [SKIP] {rname}: not found")
        continue
    tag = rname.split("_")[0:3]
    label = f"dyn_{'_'.join(tag)}_ep{ep+1}"
    topo_path, n = build_extraction_topo(d, ep, label)
    topo_info[label] = {"path": topo_path, "n_active": n}
    print(f"  {label}: {n}/250 edges")


# ── Build retrain configs ────────────────────────────────────────────────────
INPUT_VMIN = -1.0
INPUT_VMAX = 1.0
INPUT_SCALE = 1.0
GAMMAS = [0.003, 0.03, 0.3, 3.0]
DELTAS = [0.01, 0.05, 0.1]
MARGINS = [0.01, 0.05, 0.1]
LOSS = "hinge"
SEEDS = [0]
BODY_TIES = ["floating", "source", "ground"]  # interpret "body" as body tied to ground
GATE_REFS = ["ground", "source", "drain"]
VG_INITS = [7.0, 4.0, 1.0]
EPOCHS = int(args.epochs)

if args.labels:
    requested = set(args.labels)
    available = set(topo_info.keys())
    missing = sorted(requested - available)
    if missing:
        raise ValueError(f"Unknown topology labels: {missing}. Available: {sorted(available)}")
    topo_info = {k: v for k, v in topo_info.items() if k in requested}

configs = []
for label, info in topo_info.items():
    for gamma in GAMMAS:
        for delta in DELTAS:
            for margin in MARGINS:
                for body_tie in BODY_TIES:
                    for gate_ref in GATE_REFS:
                        for vg_init in VG_INITS:
                            for seed in SEEDS:
                                configs.append({
                                    "label": label,
                                    "topo_path": str(info["path"]),
                                    "n_edges": info["n_active"],
                                    "gamma": gamma,
                                    "delta": delta,
                                    "margin": margin,
                                    "loss": LOSS,
                                    "body_tie": body_tie,
                                    "gate_ref": gate_ref,
                                    "vg_init": vg_init,
                                    "seed": seed,
                                })

print(f"\nTotal retrain configs: {len(configs)}")
print(f"Epochs per run: {EPOCHS}")
print(f"Input map: [{INPUT_VMIN}, {INPUT_VMAX}] V, scale={INPUT_SCALE}")
print(f"Topology labels: {sorted(topo_info.keys())}")
print(f"Running max {max(1, int(args.batch_size))} parallel\n")

# ── Run ──────────────────────────────────────────────────────────────────────
BATCH_SIZE = max(1, int(args.batch_size))
results = []

for batch_start in range(0, len(configs), BATCH_SIZE):
    batch = configs[batch_start:batch_start + BATCH_SIZE]
    procs = []

    for cfg in batch:
        name = (
            f"{cfg['label']}_g{cfg['gamma']:g}_d{cfg['delta']:g}_m{cfg['margin']:g}_"
            f"{cfg['loss']}_b{cfg['body_tie']}_gr{cfg['gate_ref']}_"
            f"vg{cfg['vg_init']:g}_s{cfg['seed']}"
        )
        run_dir = RUNS_DIR / name

        if run_dir.exists() and (run_dir / "0_val_acc.npy").exists():
            acc = np.load(run_dir / "0_val_acc.npy")
            if len(acc) >= EPOCHS:
                results.append({
                    "label": cfg["label"], "n_edges": cfg["n_edges"],
                    "gamma": cfg["gamma"], "delta": cfg["delta"], "margin": cfg["margin"],
                    "loss": cfg["loss"], "body_tie": cfg["body_tie"], "gate_ref": cfg["gate_ref"],
                    "vg_init": cfg["vg_init"], "seed": cfg["seed"],
                    "peak": float(acc.max()), "peak_ep": int(acc.argmax()) + 1,
                    "final": float(acc[-1]), "run_name": name,
                })
                print(f"  [skip] {name} peak={acc.max():.4f}")
                continue

        run_dir.mkdir(exist_ok=True)
        env = {**os.environ, "RUN_DIR": str(run_dir)}
        cmd = [
            sys.executable, str(SCRIPT_DIR / "train_iono.py"),
            "--topology", cfg["topo_path"],
            "--epochs", str(EPOCHS),
            "--gamma", str(cfg["gamma"]),
            "--loss", cfg["loss"],
            "--gate-ref", cfg["gate_ref"],
            "--body-tie", cfg["body_tie"],
            "--vg-init", "fixed",
            "--vg-init-fixed", str(cfg["vg_init"]),
            "--vg-clip-lo", "0.4",
            "--vg-clip-hi", "8.0",
            "--input-vmin", str(INPUT_VMIN),
            "--input-vmax", str(INPUT_VMAX),
            "--input-scale", str(INPUT_SCALE),
            "--delta", str(cfg["delta"]),
            "--margin", str(cfg["margin"]),
            "--vplus", "0.45",
            "--vminus", "0.0",
            "--device-mode", "subckt",
            "--device-lib", str(DEVICE_DIR / "nmos_lvl1_ald1106.lib"),
            "--device-model", "ncg",
            str(cfg["seed"]),
        ]

        log_path = run_dir / "train.log"
        with open(log_path, "w") as log:
            p = subprocess.Popen(cmd, env=env, stdout=log, stderr=log)
        procs.append((p, cfg, name, run_dir))

    for p, cfg, name, run_dir in procs:
        p.wait()
        va_path = run_dir / "0_val_acc.npy"
        if va_path.exists():
            acc = np.load(va_path)
            peak = float(acc.max())
            peak_ep = int(acc.argmax()) + 1
            final = float(acc[-1])
            results.append({
                "label": cfg["label"], "n_edges": cfg["n_edges"],
                "gamma": cfg["gamma"], "delta": cfg["delta"], "margin": cfg["margin"],
                "loss": cfg["loss"], "body_tie": cfg["body_tie"], "gate_ref": cfg["gate_ref"],
                "vg_init": cfg["vg_init"], "seed": cfg["seed"],
                "peak": peak, "peak_ep": peak_ep, "final": final,
                "run_name": name,
            })
            flag = " *** 98%!" if peak >= 0.9859 else (" ** 97%" if peak >= 0.9718 else (" * 96%" if peak >= 0.9577 else ""))
            print(f"  [{len(results):3d}/{len(configs)}] {name:55s} "
                  f"peak={peak:.4f}@ep{peak_ep:<3d} final={final:.4f}{flag}",
                  flush=True)
        else:
            print(f"  [FAILED] {name}", flush=True)

    # Save incrementally
    ranked = sorted(results, key=lambda x: (-x["peak"], -x["final"]))
    with open(OUT / "summary.json", "w") as f:
        json.dump(ranked, f, indent=2)
    with open(OUT / "summary.md", "w") as f:
        f.write(f"# Hidden5 Retrain — {OUT.name}\n\n")
        f.write(f"Completed: {len(results)}/{len(configs)}\n\n")
        f.write("| # | Peak | @ep | Final | Edges | gamma | delta | margin | Body | Gate | VG init | Seed | Source |\n")
        f.write("|---|------|-----|-------|-------|-------|-------|--------|------|------|---------|------|--------|\n")
        for j, r in enumerate(ranked[:40], 1):
            flag = " ***" if r["peak"] >= 0.9859 else (" **" if r["peak"] >= 0.9718 else (" *" if r["peak"] >= 0.9577 else ""))
            f.write(f"| {j} | {r['peak']:.4f}{flag} | {r['peak_ep']} "
                    f"| {r['final']:.4f} | {r['n_edges']} | {r['gamma']} "
                    f"| {r['delta']} | {r['margin']} | {r['body_tie']} | {r['gate_ref']} "
                    f"| {r['vg_init']} | {r['seed']} | {r['label']} |\n")

# ── Final report ─────────────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  RETRAIN COMPLETE — {len(results)}/{len(configs)} runs")
print(f"{'='*70}")

ranked = sorted(results, key=lambda x: (-x["peak"], -x["final"]))
print("\n  TOP 20:")
for j, r in enumerate(ranked[:20], 1):
    flag = " ***" if r["peak"] >= 0.9859 else (" **" if r["peak"] >= 0.9718 else (" *" if r["peak"] >= 0.9577 else ""))
    print(f"  #{j:2d}: peak={r['peak']:.4f}@ep{r['peak_ep']:<3d} final={r['final']:.4f} "
          f"| {r['n_edges']}e g={r['gamma']} d={r['delta']} m={r['margin']} "
          f"b={r['body_tie']} gr={r['gate_ref']} vg={r['vg_init']} s={r['seed']} "
          f"| {r['label']}{flag}")

best = ranked[0] if ranked else None
if best and best["peak"] >= 0.9577:
    print(f"\n  SUCCESS: {best['peak']:.4f} ({best['n_edges']} edges)")
else:
    print(f"\n  Did not reach 96%+ target")

print(f"\nResults: {OUT / 'summary.md'}")
