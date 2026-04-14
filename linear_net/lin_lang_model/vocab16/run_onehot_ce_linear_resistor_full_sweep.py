#!/usr/bin/env python3
"""Full vocab16 linear-resistor one-hot CE hyperparameter sweep with 5-way concurrency."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable


TEMPS = ("0.001", "0.005", "0.01", "0.05", "0.1")
GAMMAS = ("0.03", "0.3", "3")
DELTAS = ("0.05", "0.1", "0.3")
INITS = (
    ("fixed_low", ["--vg-init", "fixed", "--vg-init-fixed", "1.0"]),
    ("fixed_high", ["--vg-init", "fixed", "--vg-init-fixed", "3.0"]),
    ("rand_1to3", ["--vg-init", "random", "--vg-init-lo", "1.0", "--vg-init-hi", "3.0"]),
)


@dataclass(frozen=True)
class Combo:
    gamma: str
    delta: str
    temp: str
    init_name: str
    init_args: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sweep vocab16 linear-resistor one-hot CE trainer over gamma/delta/temp/init."
    )
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-workers", type=int, default=5)
    p.add_argument("--num-sentences", type=int, default=1000)
    p.add_argument("--template-mode", type=str, default="balanced", choices=["tiny", "balanced"])
    p.add_argument("--max-sentence-words", type=int, default=7)
    p.add_argument("--limit", type=int, default=0, help="If >0, only run the first N combos for testing.")
    return p.parse_args()


def make_combos() -> list[Combo]:
    combos: list[Combo] = []
    for gamma, delta, temp, (init_name, init_args) in itertools.product(GAMMAS, DELTAS, TEMPS, INITS):
        combos.append(Combo(gamma=gamma, delta=delta, temp=temp, init_name=init_name, init_args=tuple(init_args)))
    return combos


def format_float_tag(value: str) -> str:
    return value.replace(".", "p")


def run_name(combo: Combo) -> str:
    return (
        f"g{format_float_tag(combo.gamma)}"
        f"_d{format_float_tag(combo.delta)}"
        f"_t{format_float_tag(combo.temp)}"
        f"_init-{combo.init_name}"
    )


def run_one(repo: Path, sweep_root: Path, args: argparse.Namespace, combo: Combo) -> dict[str, object]:
    name = run_name(combo)
    run_dir = sweep_root / name
    env = os.environ.copy()
    env["RUN_DIR"] = str(run_dir)
    cmd = [
        sys.executable,
        str(repo / "linear_net" / "lin_lang_model" / "vocab16" / "clln_lang_trainer_onehot_ce_linear_resistor.py"),
        str(int(args.seed)),
        "--epochs",
        str(int(args.epochs)),
        "--num-sentences",
        str(int(args.num_sentences)),
        "--template-mode",
        str(args.template_mode),
        "--max-sentence-words",
        str(int(args.max_sentence_words)),
        "--gamma",
        combo.gamma,
        "--delta",
        combo.delta,
        "--softmax-temp",
        combo.temp,
        "--process-mode",
        "in_process",
        "--eval-every",
        "1",
        "--sample-every",
        "0",
        "--plot-every",
        "0",
        "--sample-prompts",
        "0",
    ] + list(combo.init_args)

    t0 = time.time()
    proc = subprocess.run(cmd, cwd=repo, env=env, capture_output=True, text=True)
    return {
        "combo": combo,
        "run_name": name,
        "returncode": proc.returncode,
        "seconds": time.time() - t0,
        "stdout_tail": "\n".join(proc.stdout.splitlines()[-8:]),
        "stderr_tail": "\n".join(proc.stderr.splitlines()[-20:]),
    }


def load_epoch_summaries(run_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for path in sorted(run_dir.glob("0_epoch_summary_epoch*.json")):
        rows.append(json.loads(path.read_text()))
    return rows


def summarize_run(run_dir: Path) -> dict[str, object]:
    rows = load_epoch_summaries(run_dir)
    if not rows:
        raise FileNotFoundError(f"No epoch summaries found in {run_dir}")

    final = max(rows, key=lambda row: int(row["epoch"]))
    peak_support_acc = max(rows, key=lambda row: float(row["val"]["support_acc"]))
    peak_support_mass = max(rows, key=lambda row: float(row["val"]["support_mass"]))
    cfg = final["config"]
    return {
        "run": run_dir.name,
        "gamma": float(cfg["gamma"]),
        "delta": float(cfg["delta"]),
        "softmax_temp": float(cfg["softmax_temp"]),
        "vg_init_mode": str(cfg["vg_init"]["mode"]),
        "vg_init_fixed": float(cfg["vg_init"]["fixed"]),
        "vg_init_lo": float(cfg["vg_init"]["lo"]),
        "vg_init_hi": float(cfg["vg_init"]["hi"]),
        "final_epoch": int(final["epoch"]),
        "final_val_support_acc": float(final["val"]["support_acc"]),
        "final_val_support_mass": float(final["val"]["support_mass"]),
        "final_val_exact_acc": float(final["val"]["exact_acc"]),
        "peak_support_acc": float(peak_support_acc["val"]["support_acc"]),
        "peak_support_acc_epoch": int(peak_support_acc["epoch"]),
        "peak_support_mass": float(peak_support_mass["val"]["support_mass"]),
        "peak_support_mass_epoch": int(peak_support_mass["epoch"]),
    }


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[3]
    sweep_root = (
        repo
        / "linear_net"
        / "lin_lang_model"
        / "vocab16"
        / "results_language_16_linear_resistor_onehotce"
        / "sweeps"
        / f"full_grid_20ep_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    sweep_root.mkdir(parents=True, exist_ok=True)
    launcher_log = sweep_root / "launcher.log"

    combos = make_combos()
    if int(args.limit) > 0:
        combos = combos[: int(args.limit)]

    print(sweep_root, flush=True)

    completed_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    with launcher_log.open("a", buffering=1) as log:
        log.write(f"sweep_root={sweep_root}\n")
        log.write(f"launched_at={datetime.now().isoformat()}\n")
        log.write(f"epochs={int(args.epochs)} seed={int(args.seed)} workers={int(args.max_workers)}\n")
        log.write(
            f"grid=gammas:{','.join(GAMMAS)} deltas:{','.join(DELTAS)} temps:{','.join(TEMPS)} inits:{','.join(name for name, _ in INITS)}\n"
        )
        log.write(f"combos={len(combos)}\n")

        done = 0
        with ThreadPoolExecutor(max_workers=int(args.max_workers)) as ex:
            futs = {ex.submit(run_one, repo, sweep_root, args, combo): combo for combo in combos}
            for fut in as_completed(futs):
                res = fut.result()
                done += 1
                line = f"[{done}/{len(combos)}] run={res['run_name']} rc={res['returncode']} sec={res['seconds']:.2f}"
                print(line, flush=True)
                log.write(line + "\n")
                if res["stdout_tail"]:
                    log.write(str(res["stdout_tail"]) + "\n")
                if res["stderr_tail"]:
                    log.write("[stderr]\n" + str(res["stderr_tail"]) + "\n")

                if int(res["returncode"]) != 0:
                    failures.append(
                        {
                            "run": res["run_name"],
                            "returncode": int(res["returncode"]),
                            "seconds": float(res["seconds"]),
                        }
                    )
                    continue

                summary = summarize_run(sweep_root / str(res["run_name"]))
                summary["seconds"] = float(res["seconds"])
                completed_rows.append(summary)

        if completed_rows:
            final_support_acc_rows = sorted(completed_rows, key=lambda row: float(row["final_val_support_acc"]), reverse=True)
            final_support_mass_rows = sorted(completed_rows, key=lambda row: float(row["final_val_support_mass"]), reverse=True)
            peak_support_acc_rows = sorted(completed_rows, key=lambda row: float(row["peak_support_acc"]), reverse=True)
            peak_support_mass_rows = sorted(completed_rows, key=lambda row: float(row["peak_support_mass"]), reverse=True)

            write_csv(sweep_root / "leaderboard_final_support_acc.csv", final_support_acc_rows)
            write_csv(sweep_root / "leaderboard_final_support_mass.csv", final_support_mass_rows)
            write_csv(sweep_root / "leaderboard_peak_support_acc.csv", peak_support_acc_rows)
            write_csv(sweep_root / "leaderboard_peak_support_mass.csv", peak_support_mass_rows)

            summary_json = {
                "sweep_root": str(sweep_root),
                "completed_runs": len(completed_rows),
                "failed_runs": len(failures),
                "best_final_support_acc": final_support_acc_rows[0],
                "best_final_support_mass": final_support_mass_rows[0],
                "best_peak_support_acc": peak_support_acc_rows[0],
                "best_peak_support_mass": peak_support_mass_rows[0],
            }
            (sweep_root / "sweep_summary.json").write_text(json.dumps(summary_json, indent=2))
            log.write(json.dumps(summary_json, indent=2) + "\n")

        if failures:
            write_csv(sweep_root / "failures.csv", failures)

    print(f"SWEEP_DONE {sweep_root}", flush=True)


if __name__ == "__main__":
    main()
