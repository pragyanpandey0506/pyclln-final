#!/usr/bin/env python3
"""Exact 54-run gamma/delta/temp/init sweep used for the vocab32 linear-resistor chat."""

from __future__ import annotations

import itertools
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path


GAMMAS = ("0.03", "0.3", "3")
DELTAS = ("0.05", "0.1", "0.2")
TEMPS = ("0.001", "0.01", "0.1")
INIT_MODES = ("fixed", "random")
MAX_WORKERS = 5


def run_one(repo: Path, sweep_root: Path, combo: tuple[str, str, str, str]) -> dict[str, object]:
    gamma, delta, temp, init = combo
    if init == "fixed":
        run_name = f"g{gamma}_d{delta}_t{temp}_init-fixed1p5"
        extra = ["--vg-init", "fixed", "--vg-init-fixed", "1.5"]
    else:
        run_name = f"g{gamma}_d{delta}_t{temp}_init-rand1to3"
        extra = ["--vg-init", "random", "--vg-init-lo", "1.0", "--vg-init-hi", "3.0"]

    run_dir = sweep_root / run_name
    env = os.environ.copy()
    env["RUN_DIR"] = str(run_dir)
    cmd = [
        sys.executable,
        str(repo / "lang_model" / "vocab32" / "linear_resistor" / "train_embed4_onehot_ce.py"),
        "0",
        "--epochs",
        "20",
        "--num-sentences",
        "12000",
        "--min-target-count",
        "80",
        "--max-sentence-words",
        "9",
        "--template-mode",
        "broad",
        "--gamma",
        gamma,
        "--delta",
        delta,
        "--softmax-temp",
        temp,
    ] + extra

    t0 = time.time()
    proc = subprocess.run(cmd, cwd=repo, env=env, capture_output=True, text=True)
    return {
        "combo": combo,
        "run_name": run_name,
        "returncode": proc.returncode,
        "seconds": time.time() - t0,
        "stdout_tail": "\n".join(proc.stdout.splitlines()[-6:]),
        "stderr_tail": "\n".join(proc.stderr.splitlines()[-20:]),
    }


def main() -> None:
    repo = Path(__file__).resolve().parents[3]
    sweep_root = (
        repo
        / "lang_model"
        / "vocab32"
        / "results_language_32_embed4_linear_resistor_onehotce"
        / "sweeps"
        / f"gdt_inits_p5_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    sweep_root.mkdir(parents=True, exist_ok=True)
    launcher_log = sweep_root / "launcher.log"

    combos = list(itertools.product(GAMMAS, DELTAS, TEMPS, INIT_MODES))
    print(sweep_root, flush=True)

    with launcher_log.open("a", buffering=1) as log:
        log.write(f"sweep_root={sweep_root}\n")
        log.write(f"launched_at={datetime.now().isoformat()}\n")
        log.write(f"combos={len(combos)} workers={MAX_WORKERS}\n")
        for combo in combos:
            log.write(f"queued={combo}\n")

        done = 0
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futs = {ex.submit(run_one, repo, sweep_root, combo): combo for combo in combos}
            for fut in as_completed(futs):
                res = fut.result()
                done += 1
                line = f"[{done}/{len(combos)}] run={res['run_name']} rc={res['returncode']} sec={res['seconds']:.1f}"
                print(line, flush=True)
                log.write(line + "\n")
                if res["stdout_tail"]:
                    log.write(str(res["stdout_tail"]) + "\n")
                if res["stderr_tail"]:
                    log.write("[stderr]\n" + str(res["stderr_tail"]) + "\n")
                if int(res["returncode"]) != 0:
                    print(f"FAILED {res['run_name']}", flush=True)

    print(f"SWEEP_DONE {sweep_root}", flush=True)


if __name__ == "__main__":
    main()
