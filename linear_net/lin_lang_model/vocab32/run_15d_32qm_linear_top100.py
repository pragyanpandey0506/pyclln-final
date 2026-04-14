#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


ROOT = Path(__file__).resolve().parent
TRAINER = ROOT / "15d_32qm_linear.py"
HIST_ROOT = ROOT / "results_language_32_embed4_linear_resistor_onehotce"
DEFAULT_OUT_ROOT = ROOT / "results_language_qm_embed15_linear" / "sweeps"


@dataclass(frozen=True)
class HistConfig:
    gamma: float
    delta: float
    temp: float
    init_mode: str
    init_fixed: float
    init_lo: float
    init_hi: float
    qmass: float
    exact: float
    source_run: str


@dataclass(frozen=True)
class Candidate:
    gamma: float
    delta: float
    temp: float
    init_mode: str
    init_fixed: float
    init_lo: float
    init_hi: float
    prior_score: float
    matched_source_run: str

    def key(self) -> Tuple[float, float, float, str, float, float, float]:
        return (
            float(self.gamma),
            float(self.delta),
            float(self.temp),
            str(self.init_mode),
            float(self.init_fixed),
            float(self.init_lo),
            float(self.init_hi),
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run top-100 selected hyperparameters for the QM 15D vocab32 linear trainer.")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--top-k", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-sentences", type=int, default=16000)
    p.add_argument("--min-target-count", type=int, default=120)
    p.add_argument("--max-sentence-words", type=int, default=9)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--max-train", type=int, default=0)
    p.add_argument("--max-val", type=int, default=0)
    p.add_argument("--template-mode", type=str, choices=["balanced", "broad"], default="broad")
    p.add_argument("--sample-prompts", type=int, default=0)
    p.add_argument("--sample-max-len", type=int, default=12)
    p.add_argument("--gen-temp", type=float, default=0.10)
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    return p.parse_args()


def fmt_float(x: float) -> str:
    s = f"{float(x):g}"
    return s.replace("-", "m").replace(".", "p")


def load_historical_configs() -> List[HistConfig]:
    best_by_key: Dict[Tuple[float, float, float, str, float, float, float], HistConfig] = {}
    for meta_p in HIST_ROOT.rglob("run_meta.json"):
        run_dir = meta_p.parent
        try:
            meta = json.loads(meta_p.read_text())
        except Exception:
            continue
        backend = meta.get("backend")
        backend_name = backend.get("name") if isinstance(backend, dict) else backend
        if backend_name != "linear_resistor_kcl":
            continue
        summaries = sorted(run_dir.glob("0_epoch_summary_epoch*.json"), key=lambda p: int(p.stem.split("epoch")[-1]))
        if not summaries:
            continue
        try:
            summ = json.loads(summaries[-1].read_text())
        except Exception:
            continue
        val = summ.get("val", {})
        qmass = val.get("qmass_mean", val.get("val_qmass_mean"))
        exact = val.get("exact_acc")
        if qmass is None or exact is None:
            continue
        cfg = HistConfig(
            gamma=float(meta["gamma"]),
            delta=float(meta["delta"]),
            temp=float(meta["softmax_temp"]),
            init_mode=str(meta["vg_init"]["mode"]),
            init_fixed=float(meta["vg_init"]["fixed"]),
            init_lo=float(meta["vg_init"]["lo"]),
            init_hi=float(meta["vg_init"]["hi"]),
            qmass=float(qmass),
            exact=float(exact),
            source_run=str(run_dir.relative_to(HIST_ROOT)),
        )
        key = (cfg.gamma, cfg.delta, cfg.temp, cfg.init_mode, cfg.init_fixed, cfg.init_lo, cfg.init_hi)
        prev = best_by_key.get(key)
        if prev is None or (cfg.qmass, cfg.exact) > (prev.qmass, prev.exact):
            best_by_key[key] = cfg
    ranked = sorted(best_by_key.values(), key=lambda r: (r.qmass, r.exact), reverse=True)
    if not ranked:
        raise RuntimeError(f"No historical configs found under {HIST_ROOT}")
    return ranked


def init_penalty(mode: str, fixed: float, lo: float, hi: float, ref: HistConfig) -> float:
    if mode == ref.init_mode:
        if mode == "fixed":
            return 0.04 * abs(fixed - ref.init_fixed)
        return 0.02 * (abs(lo - ref.init_lo) + abs(hi - ref.init_hi))
    return 0.04


def log10_dist(a: float, b: float) -> float:
    return abs(math.log10(float(a)) - math.log10(float(b)))


def score_candidate(
    gamma: float,
    delta: float,
    temp: float,
    init_mode: str,
    init_fixed: float,
    init_lo: float,
    init_hi: float,
    history: Sequence[HistConfig],
) -> Tuple[float, str]:
    best_score = -1e30
    best_source = ""
    for ref in history:
        score = float(ref.qmass)
        score -= 0.08 * log10_dist(gamma, ref.gamma)
        score -= 0.05 * log10_dist(delta, ref.delta)
        score -= 0.05 * log10_dist(temp, ref.temp)
        score -= init_penalty(init_mode, init_fixed, init_lo, init_hi, ref)
        score += 0.02 * float(ref.exact)
        if score > best_score:
            best_score = score
            best_source = ref.source_run
    return best_score, best_source


def build_candidates(history: Sequence[HistConfig]) -> List[Candidate]:
    gamma_vals = [0.02, 0.03, 0.1, 0.3, 1.0, 3.0]
    delta_vals = [0.05, 0.075, 0.1, 0.15, 0.2]
    temp_vals = [0.001, 0.005, 0.01, 0.05, 0.1]
    init_opts = [
        ("random", 2.0, 1.0, 3.0),
        ("fixed", 1.0, 1.0, 3.0),
        ("fixed", 1.5, 1.0, 3.0),
        ("fixed", 2.0, 1.0, 3.0),
    ]

    out: List[Candidate] = []
    for gamma in gamma_vals:
        for delta in delta_vals:
            for temp in temp_vals:
                for init_mode, init_fixed, init_lo, init_hi in init_opts:
                    prior_score, matched_source = score_candidate(
                        gamma=gamma,
                        delta=delta,
                        temp=temp,
                        init_mode=init_mode,
                        init_fixed=init_fixed,
                        init_lo=init_lo,
                        init_hi=init_hi,
                        history=history,
                    )
                    out.append(
                        Candidate(
                            gamma=gamma,
                            delta=delta,
                            temp=temp,
                            init_mode=init_mode,
                            init_fixed=init_fixed,
                            init_lo=init_lo,
                            init_hi=init_hi,
                            prior_score=prior_score,
                            matched_source_run=matched_source,
                        )
                    )
    out.sort(key=lambda c: (c.prior_score, -c.temp, -c.delta), reverse=True)
    return out


def select_top_candidates(history: Sequence[HistConfig], top_k: int) -> List[Candidate]:
    ranked = build_candidates(history)
    out: List[Candidate] = []
    seen = set()
    for cand in ranked:
        key = cand.key()
        if key in seen:
            continue
        seen.add(key)
        out.append(cand)
        if len(out) >= top_k:
            break
    return out


def candidate_name(rank: int, cand: Candidate) -> str:
    init_tag = f"fixed{fmt_float(cand.init_fixed)}" if cand.init_mode == "fixed" else f"rand{fmt_float(cand.init_lo)}to{fmt_float(cand.init_hi)}"
    return f"rank{rank:03d}_g{fmt_float(cand.gamma)}_d{fmt_float(cand.delta)}_t{fmt_float(cand.temp)}_{init_tag}"


def run_one(cand: Candidate, rank: int, run_root: Path, args: argparse.Namespace) -> Dict[str, object]:
    run_name = candidate_name(rank, cand)
    run_dir = run_root / run_name
    log_path = run_dir / "launcher.log"
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(TRAINER),
        str(int(args.seed)),
        "--epochs", str(int(args.epochs)),
        "--gamma", str(float(cand.gamma)),
        "--delta", str(float(cand.delta)),
        "--softmax-temp", str(float(cand.temp)),
        "--num-sentences", str(int(args.num_sentences)),
        "--min-target-count", str(int(args.min_target_count)),
        "--max-sentence-words", str(int(args.max_sentence_words)),
        "--val-frac", str(float(args.val_frac)),
        "--max-train", str(int(args.max_train)),
        "--max-val", str(int(args.max_val)),
        "--template-mode", str(args.template_mode),
        "--vg-init", str(cand.init_mode),
        "--vg-init-lo", str(float(cand.init_lo)),
        "--vg-init-hi", str(float(cand.init_hi)),
        "--vg-init-fixed", str(float(cand.init_fixed)),
        "--sample-prompts", str(int(args.sample_prompts)),
        "--sample-max-len", str(int(args.sample_max_len)),
        "--gen-temp", str(float(args.gen_temp)),
        "--results-dir", str(run_root),
    ]

    env = os.environ.copy()
    env["RUN_DIR"] = str(run_dir)
    with log_path.open("w") as log_f:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT.parent.parent.parent),
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    result: Dict[str, object] = {
        "rank": rank,
        "run_name": run_name,
        **asdict(cand),
        "returncode": int(proc.returncode),
    }
    if proc.returncode == 0:
        val_hist_p = run_dir / "val_history.json"
        if val_hist_p.exists():
            val_hist = json.loads(val_hist_p.read_text())
            if val_hist:
                final = val_hist[-1]
                best_support = max(val_hist, key=lambda r: float(r.get("support_acc", float("-inf"))))
                best_qmass = max(val_hist, key=lambda r: float(r.get("qmass_mean", float("-inf"))))
                result.update({
                    "final_epoch": int(final["epoch"]),
                    "final_val_exact_acc": float(final["exact_acc"]),
                    "final_val_support_acc": float(final["support_acc"]),
                    "final_val_qmass_mean": float(final["qmass_mean"]),
                    "final_val_soft_ce": float(final["soft_ce"]),
                    "best_support_epoch": int(best_support["epoch"]),
                    "best_val_support_acc": float(best_support["support_acc"]),
                    "best_support_qmass_mean": float(best_support["qmass_mean"]),
                    "best_qmass_epoch": int(best_qmass["epoch"]),
                    "best_val_qmass_mean": float(best_qmass["qmass_mean"]),
                    "best_qmass_support_acc": float(best_qmass["support_acc"]),
                })
    return result


def write_summary(run_root: Path, rows: Sequence[Dict[str, object]]) -> None:
    summary_path = run_root / "sweep_summary.json"
    csv_path = run_root / "leaderboard_final_qmass.csv"
    rows_sorted = sorted(
        rows,
        key=lambda r: (
            float(r.get("final_val_qmass_mean", float("-inf"))),
            float(r.get("final_val_support_acc", float("-inf"))),
            float(r.get("final_val_exact_acc", float("-inf"))),
        ),
        reverse=True,
    )
    summary = {
        "num_runs": len(rows),
        "num_success": int(sum(int(r.get("returncode", 1) == 0) for r in rows)),
        "top_final_qmass": rows_sorted[:10],
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    if rows_sorted:
        fieldnames = list(rows_sorted[0].keys())
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows_sorted)


def main() -> None:
    args = parse_args()
    history = load_historical_configs()
    selected = select_top_candidates(history, int(args.top_k))
    run_root = args.out_root / f"top{len(selected)}_from_embed4_prior_50ep_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_root.mkdir(parents=True, exist_ok=True)

    (run_root / "selection_history_top55.json").write_text(json.dumps([asdict(h) for h in history], indent=2))
    (run_root / "selected_candidates.json").write_text(json.dumps([asdict(c) for c in selected], indent=2))

    results: List[Dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=int(args.workers)) as ex:
        fut_map = {
            ex.submit(run_one, cand, idx + 1, run_root, args): (idx + 1, cand)
            for idx, cand in enumerate(selected)
        }
        for fut in as_completed(fut_map):
            rank, cand = fut_map[fut]
            result = fut.result()
            results.append(result)
            print(
                json.dumps(
                    {
                        "rank": rank,
                        "gamma": cand.gamma,
                        "delta": cand.delta,
                        "temp": cand.temp,
                        "init_mode": cand.init_mode,
                        "returncode": result.get("returncode"),
                        "final_val_qmass_mean": result.get("final_val_qmass_mean"),
                        "final_val_support_acc": result.get("final_val_support_acc"),
                    }
                ),
                flush=True,
            )
            write_summary(run_root, results)

    write_summary(run_root, results)
    print(f"saved sweep to {run_root}", flush=True)


if __name__ == "__main__":
    main()
