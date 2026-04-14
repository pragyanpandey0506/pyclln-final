#!/usr/bin/env python3
"""
Prompt-conditioned inference for the quantum 16-input / 15-output one-hot CLLN.

This model does not predict BOS. BOS is input padding only, so inference starts
from a user-supplied prompt and predicts only real tokens plus EOS.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List

import numpy as np
from PySpice.Spice.NgSpice.Shared import NgSpiceShared

from clln_lang_trainer_onehot_ce_quantum import (
    CONTEXT_LEN,
    ID_TO_WORD,
    OUTPUT_ID_TO_TOKEN_ID,
    OUTPUT_ID_TO_WORD,
    WORD_TO_ID,
    alter_inputs_named,
    encode_context_tokens,
    make_dense_io_topology,
    mk_free_all,
    mk_netlist,
    restore_gate_voltages,
    run_and_read,
    softmax_logits,
)


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    default_run = here / "results_language_16_onehotce_quantum" / "latest"
    p = argparse.ArgumentParser(description="Prompt-conditioned sampling from the quantum one-hot CLLN")
    p.add_argument("--prompt", type=str, required=True, help="Starting text, e.g. 'the detector'")
    p.add_argument("--run-dir", type=Path, default=default_run)
    p.add_argument("--epoch", type=int, default=None, help="Required unless --run-dir points at latest")
    p.add_argument("--sample-temp", type=float, default=None)
    p.add_argument("--max-new-tokens", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--top-k", type=int, default=5)
    return p.parse_args()


def normalize_prompt(prompt: str) -> List[str]:
    tokens = re.findall(r"[a-z<>]+", prompt.lower())
    if not tokens:
        raise ValueError("Prompt is empty after normalization")
    bad = [tok for tok in tokens if tok not in WORD_TO_ID or tok == "<BOS>"]
    if bad:
        allowed = ", ".join([w for w in WORD_TO_ID.keys() if w != "<BOS>"])
        raise ValueError(f"Unknown or invalid prompt tokens: {bad}. Allowed vocab: {allowed}")
    return tokens


def to_context_ids(tokens: List[str]) -> List[int]:
    bos = WORD_TO_ID["<BOS>"]
    ids = [WORD_TO_ID[t] for t in tokens]
    if len(ids) >= CONTEXT_LEN:
        return ids[-CONTEXT_LEN:]
    return [bos] * (CONTEXT_LEN - len(ids)) + ids


def topk_str(probs: np.ndarray, k: int) -> str:
    idx = np.argsort(-probs)[: max(1, int(k))]
    return ", ".join(f"{OUTPUT_ID_TO_WORD[int(i)]}:{float(probs[int(i)]):.4f}" for i in idx.tolist())


def resolve_epoch(run_dir: Path, epoch_arg: int | None) -> int:
    if epoch_arg is not None:
        return int(epoch_arg)
    candidates = sorted(run_dir.glob("0_vg_unique_epoch*.npy"), key=lambda p: int(p.stem.split("epoch")[-1]))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in {run_dir}")
    return int(candidates[-1].stem.split("epoch")[-1])


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")

    meta_path = run_dir / "run_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing run metadata: {meta_path}")
    meta = json.loads(meta_path.read_text())
    epoch = resolve_epoch(run_dir, args.epoch)

    vg_path = run_dir / f"0_vg_unique_epoch{epoch}.npy"
    if not vg_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {vg_path}")
    vg_unique = np.load(vg_path)

    dataset = meta.get("dataset", {})
    rails = meta.get("rails", {})
    device = meta.get("device", {})
    sample_temp = float(args.sample_temp if args.sample_temp is not None else meta.get("softmax_temp", 1.0))
    if sample_temp <= 0.0:
        raise ValueError("--sample-temp must be > 0")

    topo = make_dense_io_topology()
    netlist = mk_netlist(
        topo=topo,
        vg_unique=vg_unique,
        vminus_val=float(rails.get("vminus", 0.0)),
        vplus_val=float(rails.get("vplus", 0.45)),
        solver=str(meta.get("solver", "klu")),
        body_res=float(meta.get("body_res", 10.0)),
        body_tie=str(meta.get("body_tie", "ground")),
        device_lib_path=str(device.get("include_path", "")),
    )

    prompt_tokens = normalize_prompt(args.prompt)
    ctx = to_context_ids(prompt_tokens)

    rng = np.random.default_rng(int(args.seed))
    ng = NgSpiceShared(send_data=False)
    ng.load_circuit(netlist)
    restore_gate_voltages(ng, vg_unique)

    print(f"run_dir={run_dir}")
    print(f"epoch={epoch}")
    print(f"prompt={' '.join(prompt_tokens)}")
    print(f"initial_context={' '.join(ID_TO_WORD[i] for i in ctx)}")

    generated: List[str] = []
    for step in range(1, int(args.max_new_tokens) + 1):
        mk_free_all(ng, topo.K)
        xin = encode_context_tokens(
            ctx,
            bit_v0=float(dataset.get("bit_v0", 0.0)),
            bit_v1=float(dataset.get("bit_v1", 1.0)),
        )
        alter_inputs_named(ng, xin)
        ok, _, data, err = run_and_read(ng, {"out": topo.out_nodes.tolist()})
        if (not ok) or (data is None):
            raise RuntimeError(f"ngspice inference failed at step {step}: {err}")

        vout = np.asarray(data["out"], dtype=float)
        probs = softmax_logits(vout / sample_temp)
        next_out_id = int(rng.choice(np.arange(topo.K), p=probs))
        next_token_id = int(OUTPUT_ID_TO_TOKEN_ID[next_out_id])
        next_word = ID_TO_WORD[next_token_id]
        generated.append(next_word)

        print(f"step={step:02d} next={next_word:>8s} topk=[{topk_str(probs, args.top_k)}]")

        ctx = ctx[1:] + [next_token_id]
        if next_word == "<EOS>":
            break

    print(f"generated={' '.join(generated)}")
    print(f"full_sequence={' '.join(prompt_tokens + generated)}")
    print(f"stopped_on_eos={bool(generated and generated[-1] == '<EOS>')}")


if __name__ == "__main__":
    main()
