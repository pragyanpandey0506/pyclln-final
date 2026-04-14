#!/usr/bin/env python3
"""Compare a split-backend digit run against a reference dense run."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCE_RUN = (
    REPO_ROOT
    / "scikit_digit"
    / "results"
    / "dense_sweep_11jan2026"
    / "run_0489_seed1_g0.3_d0.05_m0.02_btfloating_vgfix4"
)
DEFAULT_CANDIDATE_RUN = REPO_ROOT / "scikit_digit" / "results" / "circuit_split" / "latest"

CORE_ARRAYS = [
    "0_train_acc.npy",
    "0_train_hinge.npy",
    "0_val_acc.npy",
    "0_val_hinge.npy",
    "0_hinge_active_frac.npy",
    "0_reload_free.npy",
    "0_reload_clamp.npy",
    "0_nonfinite_free.npy",
    "0_nonfinite_clamp.npy",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare dense and split scikit-digit hinge runs.")
    p.add_argument("--reference-run", type=Path, default=DEFAULT_REFERENCE_RUN)
    p.add_argument("--candidate-run", type=Path, default=DEFAULT_CANDIDATE_RUN)
    p.add_argument("--atol", type=float, default=0.0, help="Absolute tolerance for allclose fallback.")
    p.add_argument("--rtol", type=float, default=0.0, help="Relative tolerance for allclose fallback.")
    return p.parse_args()


def compare_array(path_a: Path, path_b: Path, atol: float, rtol: float) -> tuple[bool, str]:
    arr_a = np.load(path_a)
    arr_b = np.load(path_b)
    if arr_a.shape != arr_b.shape:
        return False, f"shape mismatch {arr_a.shape} != {arr_b.shape}"
    if np.array_equal(arr_a, arr_b, equal_nan=True):
        return True, "exact"
    if np.allclose(arr_a, arr_b, atol=atol, rtol=rtol, equal_nan=True):
        max_abs = float(np.nanmax(np.abs(arr_a - arr_b)))
        return True, f"allclose max_abs={max_abs:.3e}"
    max_abs = float(np.nanmax(np.abs(arr_a - arr_b)))
    return False, f"diff max_abs={max_abs:.3e}"


def compare_named_files(
    reference_run: Path,
    candidate_run: Path,
    names: Iterable[str],
    atol: float,
    rtol: float,
) -> List[str]:
    lines: List[str] = []
    for name in names:
        ref_path = reference_run / name
        cand_path = candidate_run / name
        if not ref_path.exists() or not cand_path.exists():
            lines.append(f"FAIL {name}: missing file")
            continue
        ok, detail = compare_array(ref_path, cand_path, atol=atol, rtol=rtol)
        lines.append(f"{'OK' if ok else 'FAIL'} {name}: {detail}")
    return lines


def compare_globbed_files(
    reference_run: Path,
    candidate_run: Path,
    pattern: str,
    atol: float,
    rtol: float,
) -> List[str]:
    ref_files = sorted(p.name for p in reference_run.glob(pattern))
    cand_files = sorted(p.name for p in candidate_run.glob(pattern))
    if ref_files != cand_files:
        return [
            f"FAIL {pattern}: file list mismatch",
            f"  ref count={len(ref_files)} candidate count={len(cand_files)}",
        ]
    return compare_named_files(reference_run, candidate_run, ref_files, atol=atol, rtol=rtol)


def main() -> None:
    args = parse_args()
    reference_run = args.reference_run.resolve()
    candidate_run = args.candidate_run.resolve()

    if not reference_run.exists():
        raise FileNotFoundError(f"Reference run not found: {reference_run}")
    if not candidate_run.exists():
        raise FileNotFoundError(f"Candidate run not found: {candidate_run}")

    lines: List[str] = []
    lines.append(f"reference={reference_run}")
    lines.append(f"candidate={candidate_run}")
    lines.extend(compare_named_files(reference_run, candidate_run, CORE_ARRAYS, args.atol, args.rtol))
    lines.extend(compare_globbed_files(reference_run, candidate_run, "0_vg_unique_epoch*.npy", args.atol, args.rtol))
    lines.extend(compare_globbed_files(reference_run, candidate_run, "0_val_confusion_epoch*.npy", args.atol, args.rtol))
    lines.extend(compare_globbed_files(reference_run, candidate_run, "0_vout_test_epoch*.npy", args.atol, args.rtol))

    failures = [line for line in lines if line.startswith("FAIL ")]
    print("\n".join(lines))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
