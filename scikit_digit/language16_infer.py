#!/usr/bin/env python3
"""
Inference helper for clln_language_dense_trainer_16 runs.

Usage example:
  python scikit_digit/language16_infer.py \
    --run-dir lang_model/vocab16/results_language_16_hinge/latest \
    --text "the boy likes a dog"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PySpice.Spice.NgSpice.Shared import NgSpiceShared

REPO_ROOT = Path(__file__).resolve().parents[1]
VOCAB16_DIR = REPO_ROOT / "lang_model" / "vocab16"
if str(VOCAB16_DIR) not in sys.path:
    sys.path.insert(0, str(VOCAB16_DIR))

from clln_language_dense_trainer_16 import (
    CONTEXT_LEN,
    DenseIOTopology,
    ID_TO_WORD,
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
    p = argparse.ArgumentParser(description="Next-word inference from a trained 16-word CLLN run")
    p.add_argument(
        "--run-dir",
        type=Path,
        default=Path("lang_model/vocab16/results_language_16_hinge/latest"),
        help="Run directory or symlink for a 16-token hinge run",
    )
    p.add_argument(
        "--text",
        type=str,
        required=True,
        help="Input sequence, e.g. 'the boy likes a dog'",
    )
    p.add_argument(
        "--epoch",
        type=int,
        default=-1,
        help="Epoch checkpoint to use. -1 = best val_acc epoch (default).",
    )
    p.add_argument("--top-k", type=int, default=5, help="How many top predictions to print")
    return p.parse_args()


def resolve_run_dir(run_dir: Path) -> Path:
    return run_dir.resolve()


def pick_epoch(run_dir: Path, epoch_arg: int) -> int:
    if epoch_arg >= 0:
        ckpt = run_dir / f"0_vg_unique_epoch{epoch_arg}.npy"
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found for epoch {epoch_arg}: {ckpt}")
        return int(epoch_arg)

    val_acc_path = run_dir / "0_val_acc.npy"
    if not val_acc_path.exists():
        raise FileNotFoundError(f"Missing val accuracy history: {val_acc_path}")
    val_acc = np.load(val_acc_path)
    best_epoch = int(np.nanargmax(val_acc))
    if best_epoch == 0:
        # Epoch-0 has no saved VG checkpoint in this trainer; fall back to epoch 1.
        best_epoch = 1
    ckpt = run_dir / f"0_vg_unique_epoch{best_epoch}.npy"
    if not ckpt.exists():
        raise FileNotFoundError(f"Best-epoch checkpoint missing: {ckpt}")
    return best_epoch


def normalize_tokens(text: str) -> List[str]:
    toks = [t.strip().lower() for t in text.strip().split() if t.strip()]
    if not toks:
        raise ValueError("Input text is empty")
    unknown = [t for t in toks if t not in WORD_TO_ID]
    if unknown:
        vocab = ", ".join(WORD_TO_ID.keys())
        raise ValueError(f"Unknown token(s): {unknown}. Allowed vocabulary: {vocab}")
    return toks


def make_context_ids(tokens: List[str]) -> List[int]:
    bos = WORD_TO_ID["<BOS>"]
    ids = [WORD_TO_ID[t] for t in tokens]
    if len(ids) >= CONTEXT_LEN:
        return ids[-CONTEXT_LEN:]
    pad = [bos] * (CONTEXT_LEN - len(ids))
    return pad + ids


def build_topology_from_meta(meta: dict) -> DenseIOTopology:
    # Current trainer always uses the canonical 24->16 dense topology.
    _ = meta
    return make_dense_io_topology()


def run_inference(
    run_dir: Path,
    topo: DenseIOTopology,
    vg_unique: np.ndarray,
    context_ids: List[int],
    bit_v0: float,
    bit_v1: float,
    vminus: float,
    vplus: float,
    solver: str,
    body_tie: str,
    device_lib: str,
    softmax_temp: float,
) -> Tuple[int, np.ndarray]:
    netlist = mk_netlist(
        topo=topo,
        vg_unique=vg_unique,
        vminus_val=vminus,
        vplus_val=vplus,
        solver=solver,
        body_res=10.0,
        body_tie=body_tie,
        device_lib_path=device_lib,
    )

    ng = NgSpiceShared(send_data=False)
    ng.load_circuit(netlist)
    restore_gate_voltages(ng, vg_unique)
    mk_free_all(ng, topo.K)

    x = encode_context_tokens(context_ids, bit_v0=bit_v0, bit_v1=bit_v1)
    alter_inputs_named(ng, x)
    ok, _, data, err = run_and_read(ng, {"out": topo.out_nodes.tolist()})
    if (not ok) or (data is None):
        raise RuntimeError(f"Ngspice inference failed: {err}")

    vout = np.asarray(data["out"], dtype=float)
    logits = vout / float(softmax_temp)
    probs = softmax_logits(logits)
    pred_id = int(np.argmax(probs))
    return pred_id, probs


def main() -> None:
    args = parse_args()
    run_dir = resolve_run_dir(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")

    meta_path = run_dir / "run_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing run metadata: {meta_path}")
    meta = json.loads(meta_path.read_text())

    epoch = pick_epoch(run_dir, args.epoch)
    vg_path = run_dir / f"0_vg_unique_epoch{epoch}.npy"
    vg_unique = np.load(vg_path)

    toks = normalize_tokens(args.text)
    ctx_ids = make_context_ids(toks)
    ctx_words = [ID_TO_WORD[i] for i in ctx_ids]

    dataset = meta.get("dataset", {})
    rails = meta.get("rails", {})
    device = meta.get("device", {})
    topo = build_topology_from_meta(meta)

    pred_id, probs = run_inference(
        run_dir=run_dir,
        topo=topo,
        vg_unique=vg_unique,
        context_ids=ctx_ids,
        bit_v0=float(dataset.get("bit_v0", 0.0)),
        bit_v1=float(dataset.get("bit_v1", 1.0)),
        vminus=float(rails.get("vminus", 0.0)),
        vplus=float(rails.get("vplus", 0.45)),
        solver=str(meta.get("solver", "klu")),
        body_tie=str(meta.get("body_tie", "ground")),
        device_lib=str(device.get("include_path", "")),
        softmax_temp=float(meta.get("softmax_temp", 1.0)),
    )

    top_k = max(1, int(args.top_k))
    order = np.argsort(-probs)[:top_k]

    print(f"run_dir: {run_dir}")
    print(f"checkpoint_epoch: {epoch}")
    print(f"context(6): {' '.join(ctx_words)}")
    print(f"input_text: {args.text.strip().lower()}")
    print(f"pred_next: {ID_TO_WORD[pred_id]}")
    print("top_k:")
    for i in order.tolist():
        print(f"  {ID_TO_WORD[int(i)]:>6s}  p={float(probs[int(i)]):.6f}")


if __name__ == "__main__":
    main()
