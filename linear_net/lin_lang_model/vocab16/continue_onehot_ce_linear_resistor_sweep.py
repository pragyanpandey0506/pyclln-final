#!/usr/bin/env python3
"""Resume an existing vocab16 linear-resistor sweep to a higher epoch target."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Iterable


EPOCH_RE = re.compile(r"0_vg_unique_epoch(\d+)\.npy$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Resume all runs in an existing vocab16 linear-resistor sweep.")
    p.add_argument("--sweep-dir", type=Path, required=True)
    p.add_argument("--target-epochs", type=int, default=50)
    p.add_argument("--max-workers", type=int, default=5)
    p.add_argument("--limit", type=int, default=0, help="If >0, only process the first N run dirs.")
    return p.parse_args()


def detect_current_epoch(run_dir: Path) -> int:
    best = -1
    for path in run_dir.glob("0_vg_unique_epoch*.npy"):
        m = EPOCH_RE.match(path.name)
        if m:
            best = max(best, int(m.group(1)))
    return best


def get_run_dirs(sweep_dir: Path) -> list[Path]:
    run_dirs = sorted(
        path for path in sweep_dir.iterdir() if path.is_dir() and (path / "run_meta.json").exists()
    )
    return run_dirs


def build_resume_cmd(repo: Path, run_dir: Path, epoch: int, target_epochs: int) -> list[str]:
    meta = json.loads((run_dir / "run_meta.json").read_text())
    dataset = meta["dataset"]
    execution = meta.get("execution", {})
    vg_init = meta["vg_init"]
    device = meta["device"]

    cmd = [
        sys.executable,
        str(repo / "linear_net" / "lin_lang_model" / "vocab16" / "clln_lang_trainer_onehot_ce_linear_resistor.py"),
        str(int(meta["seed"])),
        "--epochs",
        str(int(target_epochs)),
        "--gamma",
        str(float(meta["gamma"])),
        "--delta",
        str(float(meta["delta"])),
        "--softmax-temp",
        str(float(meta["softmax_temp"])),
        "--bit-v0",
        str(float(dataset["bit_v0"])),
        "--bit-v1",
        str(float(dataset["bit_v1"])),
        "--vminus",
        str(float(meta["rails"]["vminus"])),
        "--vplus",
        str(float(meta["rails"]["vplus"])),
        "--num-sentences",
        str(int(dataset["num_sentences"])),
        "--max-sentence-words",
        str(int(dataset["max_sentence_words"])),
        "--template-mode",
        str(dataset["template_mode"]),
        "--device-lib",
        str(device["include_path"]),
        "--body-tie",
        str(meta["body_tie"]),
        "--solver",
        str(meta["solver"]),
        "--sample-prompts",
        str(int(meta.get("generation", {}).get("sample_prompts", 0))),
        "--sample-max-len",
        str(int(meta.get("generation", {}).get("sample_max_len", 10))),
        "--process-mode",
        str(execution.get("epoch_process_mode", "in_process")),
        "--eval-every",
        str(int(execution.get("eval_every", 1))),
        "--sample-every",
        str(int(execution.get("sample_every", 0))),
        "--plot-every",
        str(int(execution.get("plot_every", 0))),
        "--worker-run-dir",
        str(run_dir),
        "--worker-epoch",
        str(int(epoch)),
    ]

    if str(vg_init["mode"]) == "fixed":
        cmd.extend(["--vg-init", "fixed", "--vg-init-fixed", str(float(vg_init["fixed"]))])
    else:
        cmd.extend(
            [
                "--vg-init",
                "random",
                "--vg-init-lo",
                str(float(vg_init["lo"])),
                "--vg-init-hi",
                str(float(vg_init["hi"])),
            ]
        )
    return cmd


def summarize_run(run_dir: Path) -> dict[str, object]:
    rows = []
    for path in sorted(run_dir.glob("0_epoch_summary_epoch*.json")):
        rows.append(json.loads(path.read_text()))
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


def write_sweep_summary(sweep_dir: Path, completed_rows: list[dict[str, object]], failures: list[dict[str, object]]) -> None:
    if not completed_rows:
        return

    final_support_acc_rows = sorted(completed_rows, key=lambda row: float(row["final_val_support_acc"]), reverse=True)
    final_support_mass_rows = sorted(completed_rows, key=lambda row: float(row["final_val_support_mass"]), reverse=True)
    peak_support_acc_rows = sorted(completed_rows, key=lambda row: float(row["peak_support_acc"]), reverse=True)
    peak_support_mass_rows = sorted(completed_rows, key=lambda row: float(row["peak_support_mass"]), reverse=True)

    write_csv(sweep_dir / "leaderboard_final_support_acc.csv", final_support_acc_rows)
    write_csv(sweep_dir / "leaderboard_final_support_mass.csv", final_support_mass_rows)
    write_csv(sweep_dir / "leaderboard_peak_support_acc.csv", peak_support_acc_rows)
    write_csv(sweep_dir / "leaderboard_peak_support_mass.csv", peak_support_mass_rows)

    summary_json = {
        "sweep_root": str(sweep_dir),
        "completed_runs": len(completed_rows),
        "failed_runs": len(failures),
        "best_final_support_acc": final_support_acc_rows[0],
        "best_final_support_mass": final_support_mass_rows[0],
        "best_peak_support_acc": peak_support_acc_rows[0],
        "best_peak_support_mass": peak_support_mass_rows[0],
    }
    (sweep_dir / "sweep_summary.json").write_text(json.dumps(summary_json, indent=2))
    if failures:
        write_csv(sweep_dir / "failures.csv", failures)


def continue_one(repo: Path, run_dir: Path, target_epochs: int) -> dict[str, object]:
    current_epoch = detect_current_epoch(run_dir)
    t0 = time.time()
    if current_epoch >= int(target_epochs):
        return {
            "run": run_dir.name,
            "returncode": 0,
            "seconds": 0.0,
            "start_epoch": current_epoch + 1,
            "end_epoch": current_epoch,
            "stdout_tail": "already_at_target",
            "stderr_tail": "",
        }

    stdout_tail = ""
    stderr_tail = ""
    for epoch in range(current_epoch + 1, int(target_epochs) + 1):
        cmd = build_resume_cmd(repo, run_dir, epoch=epoch, target_epochs=target_epochs)
        proc = subprocess.run(cmd, cwd=repo, capture_output=True, text=True)
        stdout_tail = "\n".join(proc.stdout.splitlines()[-8:])
        stderr_tail = "\n".join(proc.stderr.splitlines()[-20:])
        if proc.returncode != 0:
            return {
                "run": run_dir.name,
                "returncode": proc.returncode,
                "seconds": time.time() - t0,
                "start_epoch": current_epoch + 1,
                "end_epoch": epoch,
                "stdout_tail": stdout_tail,
                "stderr_tail": stderr_tail,
            }

    meta = json.loads((run_dir / "run_meta.json").read_text())
    meta["epochs"] = int(target_epochs)
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    return {
        "run": run_dir.name,
        "returncode": 0,
        "seconds": time.time() - t0,
        "start_epoch": current_epoch + 1,
        "end_epoch": int(target_epochs),
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
    }


def main() -> None:
    args = parse_args()
    repo = Path(__file__).resolve().parents[3]
    sweep_dir = args.sweep_dir.resolve()
    if not sweep_dir.exists():
        raise FileNotFoundError(f"Sweep dir not found: {sweep_dir}")

    run_dirs = get_run_dirs(sweep_dir)
    if int(args.limit) > 0:
        run_dirs = run_dirs[: int(args.limit)]
    if not run_dirs:
        raise RuntimeError(f"No run directories found in {sweep_dir}")

    launcher_log = sweep_dir / "resume_launcher.log"
    completed_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    with launcher_log.open("a", buffering=1) as log:
        log.write(f"resume_started_at={datetime.now().isoformat()}\n")
        log.write(f"sweep_dir={sweep_dir}\n")
        log.write(f"target_epochs={int(args.target_epochs)} workers={int(args.max_workers)} runs={len(run_dirs)}\n")

        done = 0
        with ThreadPoolExecutor(max_workers=int(args.max_workers)) as ex:
            futs = {ex.submit(continue_one, repo, run_dir, int(args.target_epochs)): run_dir for run_dir in run_dirs}
            for fut in as_completed(futs):
                res = fut.result()
                done += 1
                line = (
                    f"[{done}/{len(run_dirs)}] run={res['run']} rc={res['returncode']} "
                    f"epochs={res['start_epoch']}..{res['end_epoch']} sec={res['seconds']:.2f}"
                )
                print(line, flush=True)
                log.write(line + "\n")
                if res["stdout_tail"]:
                    log.write(str(res["stdout_tail"]) + "\n")
                if res["stderr_tail"]:
                    log.write("[stderr]\n" + str(res["stderr_tail"]) + "\n")

                if int(res["returncode"]) != 0:
                    failures.append(
                        {
                            "run": str(res["run"]),
                            "returncode": int(res["returncode"]),
                            "seconds": float(res["seconds"]),
                        }
                    )

        for run_dir in run_dirs:
            if detect_current_epoch(run_dir) >= int(args.target_epochs):
                completed_rows.append(summarize_run(run_dir))

        write_sweep_summary(sweep_dir, completed_rows, failures)

    print(f"RESUME_DONE {sweep_dir}", flush=True)


if __name__ == "__main__":
    main()
