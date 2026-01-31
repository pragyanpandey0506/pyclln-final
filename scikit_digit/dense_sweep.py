#!/usr/bin/env python3
"""
Dense trainer sweep launcher (10-way parallel)

User-specified sweep (same grid as auto_prune_sweep.py):
- epochs: 100
- gamma:  (0.03, 0.3, 3)
- delta:  (0.02, 0.05, 0.1)
- margin: (0.02, 0.05, 0.1)
- body-tie: (ground, source, floating)   # "body" == "ground"
- vg-init:
    fixed: 0.75, 2.0, 4.0
    random: [0.75,3.0], [1.0,6.0], [1.5,2.5]
- solver: klu only
- seed: 0,1,2
- parallel: 10
- output root: /home/ma-lab/Desktop/pyclln-final/scikit_digit/results/dense_sweep_11jan2026

This script assumes the trainer honors RUN_DIR env var (as your trainers do) to place
all artifacts in that directory.
"""

from __future__ import annotations

import itertools
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# -----------------------------
# USER CONFIG
# -----------------------------
TRAINER = Path("/home/ma-lab/Desktop/pyclln-final/scikit_digit/dense_trainer.py")
SWEEP_ROOT = Path("/home/ma-lab/Desktop/pyclln-final/scikit_digit/results/dense_sweep_11jan2026")

MAX_PARALLEL = 10
PYTHON = sys.executable  # use current python

EPOCHS = 100
SOLVER = "klu"

GAMMAS = [0.03, 0.3, 3.0]
DELTAS = [0.02, 0.05, 0.1]
MARGINS = [0.02, 0.05, 0.1]

BODY_TIES = ["ground", "source", "floating"]  # "body" == "ground"
SEEDS = [0, 1, 2]

VG_FIXED = [0.75, 2.0, 4.0]
VG_RANDOM_RANGES = [(0.75, 3.0), (1.0, 6.0), (1.5, 2.5)]

# If you want deterministic ordering, set SHUFFLE=False
SHUFFLE = True
RNG_SEED_FOR_SHUFFLE = 11

# Skip runs that already have DONE marker
SKIP_DONE = True

# -----------------------------
# Helpers
# -----------------------------
def sanitize(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[^a-zA-Z0-9._=-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s)
    return s[:180]

def fmt_f(x: float) -> str:
    # compact but stable
    return f"{x:.6g}"

@dataclass(frozen=True)
class RunSpec:
    seed: int
    gamma: float
    delta: float
    margin: float
    body_tie: str  # ground/source/floating
    vg_init_mode: str  # fixed/random
    vg_fixed: Optional[float] = None
    vg_lo: Optional[float] = None
    vg_hi: Optional[float] = None

    def short_name(self) -> str:
        bt = {"ground": "body", "source": "source", "floating": "floating"}.get(self.body_tie, self.body_tie)
        if self.vg_init_mode == "fixed":
            assert self.vg_fixed is not None
            init = f"vgfix{fmt_f(self.vg_fixed)}"
        else:
            assert self.vg_lo is not None and self.vg_hi is not None
            init = f"vgrand{fmt_f(self.vg_lo)}to{fmt_f(self.vg_hi)}"
        parts = [
            f"seed{self.seed}",
            f"g{fmt_f(self.gamma)}",
            f"d{fmt_f(self.delta)}",
            f"m{fmt_f(self.margin)}",
            f"bt{bt}",
            init,
        ]
        return sanitize("_".join(parts))

    def to_dict(self) -> Dict:
        return {
            "seed": int(self.seed),
            "epochs": int(EPOCHS),
            "gamma": float(self.gamma),
            "delta": float(self.delta),
            "margin": float(self.margin),
            "body_tie": str(self.body_tie),
            "vg_init_mode": str(self.vg_init_mode),
            "vg_fixed": None if self.vg_fixed is None else float(self.vg_fixed),
            "vg_lo": None if self.vg_lo is None else float(self.vg_lo),
            "vg_hi": None if self.vg_hi is None else float(self.vg_hi),
            "solver": str(SOLVER),
        }

def build_cmd(spec: RunSpec) -> List[str]:
    cmd = [
        PYTHON, str(TRAINER),
        str(spec.seed),
        "--epochs", str(EPOCHS),
        "--solver", SOLVER,
        "--gamma", fmt_f(spec.gamma),
        "--delta", fmt_f(spec.delta),
        "--margin", fmt_f(spec.margin),
        "--body-tie", spec.body_tie,
    ]
    if spec.vg_init_mode == "fixed":
        cmd += ["--vg-init", "fixed", "--vg-init-fixed", fmt_f(float(spec.vg_fixed))]
    else:
        cmd += ["--vg-init", "random", "--vg-init-lo", fmt_f(float(spec.vg_lo)), "--vg-init-hi", fmt_f(float(spec.vg_hi))]
    return cmd

def write_json(path: Path, obj: Dict) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=False) + "\n")

def tail_file(path: Path, n_lines: int = 80) -> str:
    if not path.exists():
        return ""
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            # read last ~64KB
            f.seek(max(0, size - 65536), os.SEEK_SET)
            data = f.read().decode("utf-8", errors="replace")
        lines = data.splitlines()[-n_lines:]
        return "\n".join(lines)
    except Exception:
        return ""

# -----------------------------
# Main launcher
# -----------------------------
def main() -> int:
    if not TRAINER.exists():
        print(f"ERROR: trainer not found: {TRAINER}", file=sys.stderr)
        return 2

    SWEEP_ROOT.mkdir(parents=True, exist_ok=True)

    # Enumerate runs
    runs: List[RunSpec] = []
    for seed, gamma, delta, margin, body_tie in itertools.product(SEEDS, GAMMAS, DELTAS, MARGINS, BODY_TIES):
        # fixed init
        for vgf in VG_FIXED:
            runs.append(RunSpec(seed=seed, gamma=gamma, delta=delta, margin=margin,
                                body_tie=body_tie, vg_init_mode="fixed", vg_fixed=vgf))
        # random init
        for lo, hi in VG_RANDOM_RANGES:
            runs.append(RunSpec(seed=seed, gamma=gamma, delta=delta, margin=margin,
                                body_tie=body_tie, vg_init_mode="random", vg_lo=lo, vg_hi=hi))

    # Shuffle (optional)
    if SHUFFLE:
        random.seed(RNG_SEED_FOR_SHUFFLE)
        random.shuffle(runs)

    # Manifest / summary
    manifest_path = SWEEP_ROOT / "manifest.json"
    manifest = {
        "trainer": str(TRAINER),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "max_parallel": int(MAX_PARALLEL),
        "total_runs": int(len(runs)),
        "sweep": {
            "epochs": EPOCHS,
            "solver": SOLVER,
            "gamma": GAMMAS,
            "delta": DELTAS,
            "margin": MARGINS,
            "body_tie": BODY_TIES,
            "seeds": SEEDS,
            "vg_fixed": VG_FIXED,
            "vg_random_ranges": VG_RANDOM_RANGES,
        },
    }
    write_json(manifest_path, manifest)

    # Pre-write all specs list for reproducibility
    specs_path = SWEEP_ROOT / "runspecs.jsonl"
    with specs_path.open("w") as f:
        for i, spec in enumerate(runs):
            rec = {"idx": i, "name": spec.short_name(), **spec.to_dict()}
            f.write(json.dumps(rec) + "\n")

    print(f"[sweep] root={SWEEP_ROOT}")
    print(f"[sweep] total runs = {len(runs)} | parallel = {MAX_PARALLEL}")
    print(f"[sweep] manifest = {manifest_path}")
    print(f"[sweep] runspecs = {specs_path}")

    # Running pool
    running: List[Tuple[subprocess.Popen, RunSpec, Path, float]] = []
    completed = 0
    failed = 0
    skipped = 0

    status_path = SWEEP_ROOT / "status.jsonl"
    status_f = status_path.open("a", buffering=1)

    def log_status(event: Dict) -> None:
        status_f.write(json.dumps(event) + "\n")
        status_f.flush()

    def launch(spec: RunSpec, idx: int) -> Optional[Tuple[subprocess.Popen, RunSpec, Path, float]]:
        # Unique run dir name; include idx to avoid collisions
        run_name = f"run_{idx:04d}_{spec.short_name()}"
        run_dir = SWEEP_ROOT / run_name
        done_marker = run_dir / "DONE"
        fail_marker = run_dir / "FAILED"

        if SKIP_DONE and done_marker.exists():
            nonlocal skipped
            skipped += 1
            log_status({"t": time.time(), "event": "skip_done", "idx": idx, "run_dir": str(run_dir)})
            return None

        run_dir.mkdir(parents=True, exist_ok=True)

        # Save spec
        write_json(run_dir / "run_spec.json", spec.to_dict())

        cmd = build_cmd(spec)

        # stdout/stderr to files (trainer also tees internally; this is extra safety)
        out_f = (run_dir / "stdout.txt").open("wb", buffering=0)
        err_f = (run_dir / "stderr.txt").open("wb", buffering=0)

        env = os.environ.copy()
        env["RUN_DIR"] = str(run_dir)
        # Keep ngspice quieter unless you want everything:
        # env["NGSPICE_BATCH"] = "1"

        log_status({"t": time.time(), "event": "start", "idx": idx, "run_dir": str(run_dir), "cmd": cmd})

        p = subprocess.Popen(
            cmd,
            cwd=str(TRAINER.parent),
            stdout=out_f,
            stderr=err_f,
            env=env,
        )
        return (p, spec, run_dir, time.time())

    def finalize(p: subprocess.Popen, spec: RunSpec, run_dir: Path, t_start: float, idx: int) -> None:
        nonlocal completed, failed
        rc = p.returncode
        dt = time.time() - t_start

        # Close file descriptors if any are still open (they are owned by Popen; safe to ignore here)
        if rc == 0:
            (run_dir / "DONE").write_text(f"ok dt_s={dt:.3f}\n")
            completed += 1
            log_status({"t": time.time(), "event": "done", "idx": idx, "run_dir": str(run_dir), "rc": rc, "dt_s": dt})
        else:
            # Put some context into FAILED marker
            stderr_tail = tail_file(run_dir / "stderr.txt", n_lines=120)
            stdout_tail = tail_file(run_dir / "stdout.txt", n_lines=120)
            (run_dir / "FAILED").write_text(
                f"rc={rc} dt_s={dt:.3f}\n\n"
                f"--- stderr tail ---\n{stderr_tail}\n\n"
                f"--- stdout tail ---\n{stdout_tail}\n"
            )
            failed += 1
            log_status({"t": time.time(), "event": "fail", "idx": idx, "run_dir": str(run_dir), "rc": rc, "dt_s": dt})

    # Scheduler loop
    i = 0
    total = len(runs)

    try:
        while completed + failed + skipped < total:
            # Launch up to MAX_PARALLEL
            while len(running) < MAX_PARALLEL and i < total:
                spec = runs[i]
                launched = launch(spec, i)
                if launched is not None:
                    running.append(launched)
                i += 1

            # Poll running processes
            still_running: List[Tuple[subprocess.Popen, RunSpec, Path, float]] = []
            for (p, spec, run_dir, t0) in running:
                rc = p.poll()
                if rc is None:
                    still_running.append((p, spec, run_dir, t0))
                else:
                    # Find original idx from directory name prefix run_XXXX_
                    m = re.match(r"run_(\d{4})_", run_dir.name)
                    idx = int(m.group(1)) if m else -1
                    finalize(p, spec, run_dir, t0, idx)

            running = still_running

            # Progress line
            done_now = completed + failed + skipped
            print(
                f"[progress] done={done_now}/{total} (ok={completed} fail={failed} skip={skipped}) "
                f"running={len(running)} queued={total - done_now - len(running)}",
                flush=True,
            )

            if running:
                time.sleep(1.0)
            else:
                time.sleep(0.2)

    except KeyboardInterrupt:
        print("\n[interrupt] received Ctrl+C; terminating running jobs...", flush=True)
        for (p, _, run_dir, _) in running:
            try:
                p.terminate()
            except Exception:
                pass
            (run_dir / "FAILED").write_text("terminated by user (KeyboardInterrupt)\n")
        return 130
    finally:
        try:
            status_f.close()
        except Exception:
            pass

    # Final summary
    summary = {
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total": total,
        "completed": completed,
        "failed": failed,
        "skipped_done": skipped,
        "max_parallel": MAX_PARALLEL,
    }
    write_json(SWEEP_ROOT / "summary.json", summary)
    print(f"[done] summary: {summary}")
    print(f"[done] status log: {status_path}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
