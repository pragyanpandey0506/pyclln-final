#!/usr/bin/env python3
"""
Cross-entropy trainer sweep launcher (parallel).

Sweeps:
  - epochs: 20
  - gamma:  (0.03, 0.3, 3)
  - delta:  (0.25, 0.5, 0.75)
  - softmax-temp: (0.5, 1.0, 2.0)
  - body-tie: (ground, source, floating)
  - vg-init:
      fixed: 0.75, 2.0, 4.0
      random: [0.75,3.0], [1.0,6.0], [1.5,2.5]
  - solver: klu only
  - seed: 0 (fixed)
  - parallel: 20
  - output root: scikit_digit/results/xent_sweep_11jan2026

This script uses RUN_DIR to place each run's artifacts under the sweep root.
"""

from __future__ import annotations

import itertools
import json
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# -----------------------------
# USER CONFIG
# -----------------------------
TRAINER = Path("/home/ma-lab/Desktop/pyclln-final/scikit_digit/dense_trainer_cross_entropy.py")
SWEEP_ROOT = Path("/home/ma-lab/Desktop/pyclln-final/scikit_digit/results/xent_sweep_11jan2026")

MAX_PARALLEL = 10
PYTHON = sys.executable  # use current python

EPOCHS = 20
SOLVER = "klu"
SEED = 0

GAMMAS = [0.03, 0.3, 3.0]
DELTAS = [0.25, 0.5, 0.75]
SOFTMAX_TEMPS = [0.0005, 0.001, 0.005, 0.01, 0.05]

BODY_TIES = ["ground", "source", "floating"]

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
    return f"{x:.6g}"


@dataclass(frozen=True)
class RunSpec:
    gamma: float
    delta: float
    softmax_temp: float
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
            f"seed{SEED}",
            f"g{fmt_f(self.gamma)}",
            f"d{fmt_f(self.delta)}",
            f"T{fmt_f(self.softmax_temp)}",
            f"bt{bt}",
            init,
        ]
        return sanitize("_".join(parts))

    def to_dict(self) -> Dict:
        return {
            "seed": int(SEED),
            "epochs": int(EPOCHS),
            "gamma": float(self.gamma),
            "delta": float(self.delta),
            "softmax_temp": float(self.softmax_temp),
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
        str(SEED),
        "--epochs", str(EPOCHS),
        "--solver", SOLVER,
        "--gamma", fmt_f(spec.gamma),
        "--delta", fmt_f(spec.delta),
        "--softmax-temp", fmt_f(spec.softmax_temp),
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
    for gamma, delta, temp, body_tie in itertools.product(GAMMAS, DELTAS, SOFTMAX_TEMPS, BODY_TIES):
        for vgf in VG_FIXED:
            runs.append(
                RunSpec(
                    gamma=gamma,
                    delta=delta,
                    softmax_temp=temp,
                    body_tie=body_tie,
                    vg_init_mode="fixed",
                    vg_fixed=vgf,
                )
            )
        for lo, hi in VG_RANDOM_RANGES:
            runs.append(
                RunSpec(
                    gamma=gamma,
                    delta=delta,
                    softmax_temp=temp,
                    body_tie=body_tie,
                    vg_init_mode="random",
                    vg_lo=lo,
                    vg_hi=hi,
                )
            )

    if SHUFFLE:
        random.seed(RNG_SEED_FOR_SHUFFLE)
        random.shuffle(runs)

    manifest_path = SWEEP_ROOT / "manifest.json"
    manifest = {
        "trainer": str(TRAINER),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "max_parallel": int(MAX_PARALLEL),
        "total_runs": int(len(runs)),
        "sweep": {
            "epochs": EPOCHS,
            "solver": SOLVER,
            "seed": SEED,
            "gamma": GAMMAS,
            "delta": DELTAS,
            "softmax_temp": SOFTMAX_TEMPS,
            "body_tie": BODY_TIES,
            "vg_fixed": VG_FIXED,
            "vg_random_ranges": VG_RANDOM_RANGES,
        },
    }
    write_json(manifest_path, manifest)

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
        write_json(run_dir / "run_spec.json", spec.to_dict())

        cmd = build_cmd(spec)

        out_f = (run_dir / "stdout.txt").open("wb", buffering=0)
        err_f = (run_dir / "stderr.txt").open("wb", buffering=0)

        env = os.environ.copy()
        env["RUN_DIR"] = str(run_dir)

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
        if rc == 0:
            (run_dir / "DONE").write_text(f"ok dt_s={dt:.3f}\n")
            completed += 1
            log_status({"t": time.time(), "event": "done", "idx": idx, "run_dir": str(run_dir), "rc": rc, "dt_s": dt})
        else:
            stderr_tail = tail_file(run_dir / "stderr.txt", n_lines=120)
            stdout_tail = tail_file(run_dir / "stdout.txt", n_lines=120)
            (run_dir / "FAILED").write_text(
                f"rc={rc} dt_s={dt:.3f}\n\n"
                f"--- stderr tail ---\n{stderr_tail}\n\n"
                f"--- stdout tail ---\n{stdout_tail}\n"
            )
            failed += 1
            log_status({"t": time.time(), "event": "fail", "idx": idx, "run_dir": str(run_dir), "rc": rc, "dt_s": dt})

    i = 0
    total = len(runs)

    try:
        while completed + failed + skipped < total:
            while len(running) < MAX_PARALLEL and i < total:
                spec = runs[i]
                launched = launch(spec, i)
                if launched is not None:
                    running.append(launched)
                i += 1

            still_running: List[Tuple[subprocess.Popen, RunSpec, Path, float]] = []
            for p, spec, run_dir, t_start in running:
                rc = p.poll()
                if rc is None:
                    still_running.append((p, spec, run_dir, t_start))
                else:
                    finalize(p, spec, run_dir, t_start, idx=0)

            running = still_running

            if (completed + failed + skipped) % 5 == 0:
                done_now = completed + failed + skipped
                print(
                    f"[progress] done={done_now}/{total} (ok={completed} fail={failed} skip={skipped}) "
                    f"running={len(running)} queued={total - done_now - len(running)}",
                    flush=True,
                )
            time.sleep(0.5)
    finally:
        status_f.flush()
        status_f.close()

    summary = {
        "completed": completed,
        "failed": failed,
        "skipped_done": skipped,
        "total": total,
    }
    print(f"[done] summary: {summary}")
    print(f"[done] status log: {status_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
