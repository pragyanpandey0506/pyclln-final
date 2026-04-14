#!/usr/bin/env python3
"""Fine-tune the best hidden-path checkpoint on hard training examples plus replay."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np


REPO_DIR = Path(__file__).resolve().parents[3]
VOCAB32_DIR = Path(__file__).resolve().parents[1]
if str(VOCAB32_DIR) not in sys.path:
    sys.path.insert(0, str(VOCAB32_DIR))

import clln_lang_trainer_embed4_onehot_ce_linear_resistor_hidden_path as trainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Resume the best vocab32 hidden-path resistor run on hard train examples plus replay"
    )
    p.add_argument("--base-run-dir", type=str, default="", help="Existing hidden-path run directory. Defaults to the best saved hidden-path checkpoint by val_qmass_mean.")
    p.add_argument("--base-epoch", type=int, default=-1, help="Checkpoint epoch to resume from. Defaults to the best saved epoch in the base run by val_qmass_mean.")
    p.add_argument("--epochs", type=int, default=25, help="Fine-tuning epochs to run from the selected checkpoint.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--focus-mode", type=str, choices=["support", "exact"], default="support")
    p.add_argument("--replay-frac", type=float, default=1.0, help="Replay examples sampled from already-correct train rows as a multiple of hard-example count.")
    p.add_argument("--max-hard", type=int, default=0, help="Optional cap on the number of hard examples. 0 means use all.")
    p.add_argument("--gamma", type=float, default=-1.0, help="Override gamma. Default keeps the base run gamma.")
    p.add_argument("--delta", type=float, default=-1.0, help="Override delta. Default keeps the base run delta.")
    p.add_argument("--softmax-temp", type=float, default=-1.0, help="Override temperature. Default keeps the base run temperature.")
    p.add_argument("--process-mode", type=str, choices=["fresh_process", "in_process"], default="in_process")
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--sample-every", type=int, default=0)
    p.add_argument("--plot-every", type=int, default=0)
    p.add_argument("--sample-prompts", type=int, default=0)
    p.add_argument("--sample-max-len", type=int, default=12)
    return p.parse_args()


def comparisons_root() -> Path:
    return VOCAB32_DIR / "results_language_32_embed4_linear_resistor_hiddenpath_onehotce" / "comparisons"


def find_best_hiddenpath_run() -> tuple[Path, int, float]:
    best_item: tuple[float, int, Path] | None = None
    for path in comparisons_root().glob("**/0_epoch_summary_epoch*.json"):
        try:
            data = json.loads(path.read_text())
            qmass = float(data["val"]["val_qmass_mean"])
            epoch = int(data["epoch"])
        except Exception:
            continue
        item = (qmass, epoch, path)
        if best_item is None or item[0] > best_item[0]:
            best_item = item
    if best_item is None:
        raise FileNotFoundError("Could not find any hidden-path epoch summary under comparisons/")
    qmass, epoch, path = best_item
    return path.parent, epoch, qmass


def find_best_epoch_in_run(run_dir: Path) -> tuple[int, float]:
    best_epoch = -1
    best_qmass = float("-inf")
    for path in run_dir.glob("0_epoch_summary_epoch*.json"):
        try:
            data = json.loads(path.read_text())
            qmass = float(data["val"]["val_qmass_mean"])
            epoch = int(data["epoch"])
        except Exception:
            continue
        if qmass > best_qmass:
            best_qmass = qmass
            best_epoch = epoch
    if best_epoch < 0:
        raise FileNotFoundError(f"Could not find any saved epoch summary in {run_dir}")
    return best_epoch, best_qmass


def load_edge_state_for_epoch(run_dir: Path, epoch: int, topo: trainer.DenseIOTopology) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    active_path = run_dir / f"0_edge_active_epoch{epoch}.npy"
    utility_path = run_dir / f"0_edge_utility_epoch{epoch}.npy"
    backbone_path = run_dir / f"0_edge_backbone_epoch{epoch}.npy"
    age_path = run_dir / f"0_edge_age_epoch{epoch}.npy"
    total_edges = trainer.PATH_PARAM_BLOCKS * topo.num_edges
    if active_path.exists():
        edge_active = np.asarray(np.load(active_path), dtype=bool).reshape(total_edges)
    else:
        edge_active = np.ones((total_edges,), dtype=bool)
    if utility_path.exists():
        utility_ema = np.asarray(np.load(utility_path), dtype=float).reshape(total_edges)
    else:
        utility_ema = np.zeros((total_edges,), dtype=float)
    if backbone_path.exists():
        backbone_ema = np.asarray(np.load(backbone_path), dtype=float).reshape(total_edges)
    else:
        backbone_ema = np.zeros((total_edges,), dtype=float)
    if age_path.exists():
        edge_age = np.asarray(np.load(age_path), dtype=int).reshape(total_edges)
    else:
        edge_age = np.zeros((total_edges,), dtype=int)
    return edge_active, utility_ema, backbone_ema, edge_age


def compute_train_focus_mask(
    *,
    run_dir: Path,
    epoch: int,
    temp: float,
    focus_mode: str,
) -> tuple[np.ndarray, dict[str, float]]:
    topo = trainer.make_dense_io_topology()
    train_x, train_y, train_ctx_arr, _, _, _ = trainer.load_saved_dataset(run_dir)
    train_ctx = [tuple(int(v) for v in row.tolist()) for row in np.asarray(train_ctx_arr, dtype=int)]
    q_map_train = trainer.build_context_target_distributions(train_ctx, train_y.tolist(), trainer.OUTPUT_DIM)
    unigram_q_train = trainer.build_unigram_target_distribution(train_y.tolist(), trainer.OUTPUT_DIM)
    q_train, _ = trainer.build_q_matrix(train_ctx, q_map_train, unigram_q_train)

    vg_unique = np.asarray(np.load(run_dir / f"0_vg_unique_epoch{epoch}.npy"), dtype=float)
    edge_active, _, _, _ = load_edge_state_for_epoch(run_dir, epoch, topo)
    _, _, _, g_total = trainer.params_to_masked_conductances(vg_unique, edge_active, topo)
    sum_g = np.sum(g_total, axis=1)
    vout = trainer.solve_outputs_batch_free(g_total, train_x, sum_g=sum_g)
    probs = trainer.softmax_rows(vout / float(temp))
    pred = np.argmax(vout, axis=1)

    exact_ok = pred == train_y
    support_ok = q_train[np.arange(q_train.shape[0]), pred] > 0.0
    if focus_mode == "exact":
        hard_mask = ~exact_ok
    else:
        hard_mask = ~support_ok

    stats = {
        "train_count_full": int(train_x.shape[0]),
        "exact_acc_full": float(np.mean(exact_ok)),
        "support_acc_full": float(np.mean(support_ok)),
        "exact_miss_count": int(np.sum(~exact_ok)),
        "support_miss_count": int(np.sum(~support_ok)),
        "mean_true_token_prob": float(np.mean(probs[np.arange(probs.shape[0]), train_y])),
    }
    return np.asarray(hard_mask, dtype=bool), stats


def build_subset_indices(
    *,
    hard_mask: np.ndarray,
    seed: int,
    replay_frac: float,
    max_hard: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    hard_idx = np.flatnonzero(np.asarray(hard_mask, dtype=bool))
    if max_hard > 0 and hard_idx.size > max_hard:
        hard_idx = np.sort(rng.choice(hard_idx, size=max_hard, replace=False))
    correct_idx = np.flatnonzero(~np.asarray(hard_mask, dtype=bool))
    replay_n = int(round(max(0.0, float(replay_frac)) * float(hard_idx.size)))
    replay_n = min(replay_n, int(correct_idx.size))
    if replay_n > 0:
        replay_idx = np.sort(rng.choice(correct_idx, size=replay_n, replace=False))
    else:
        replay_idx = np.zeros((0,), dtype=int)
    selected_idx = np.concatenate([hard_idx, replay_idx])
    if selected_idx.size <= 0:
        raise ValueError("No training examples selected for fine-tuning")
    selected_idx = np.asarray(selected_idx, dtype=int)
    rng.shuffle(selected_idx)
    return hard_idx, replay_idx, selected_idx


def build_run_dir_name(base_run_dir: Path, focus_mode: str, seed: int) -> str:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"hard_finetune_{stamp}_{base_run_dir.name}_{focus_mode}_seed-{seed}"


def write_preview_json(
    *,
    run_dir: Path,
    train_ctx: np.ndarray,
    train_y: np.ndarray,
) -> None:
    ctx_rows = [tuple(int(v) for v in row.tolist()) for row in np.asarray(train_ctx, dtype=int)]
    y_rows = [int(v) for v in np.asarray(train_y, dtype=int).tolist()]
    q_map = trainer.build_context_target_distributions(ctx_rows, y_rows, trainer.OUTPUT_DIM)
    unigram_q = trainer.build_unigram_target_distribution(y_rows, trainer.OUTPUT_DIM)
    preview_items = []
    for ctx, y in list(zip(ctx_rows, y_rows))[:120]:
        q_target = q_map.get(tuple(ctx), unigram_q)
        preview_items.append(
            {
                "context": [trainer.INPUT_ID_TO_WORD[t] for t in ctx],
                "target_word": trainer.OUTPUT_ID_TO_WORD[int(y)],
                "valid_next_token_distribution": {
                    trainer.OUTPUT_ID_TO_WORD[k]: float(q_target[k])
                    for k in range(trainer.OUTPUT_DIM)
                    if float(q_target[k]) > 0.0
                },
                "valid_next_token_distribution_top": trainer.top_words_from_q(q_target, topk=6),
            }
        )
    (run_dir / "sample_target_preview.json").write_text(json.dumps(preview_items, indent=2))


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        shutil.copy2(src, dst)


def prepare_run_dir(
    *,
    args: argparse.Namespace,
    base_run_dir: Path,
    base_epoch: int,
    base_best_qmass: float,
    base_meta: dict,
    hard_idx: np.ndarray,
    replay_idx: np.ndarray,
    selected_idx: np.ndarray,
    hard_stats: dict[str, float],
) -> Path:
    topo = trainer.make_dense_io_topology()
    train_x, train_y, train_ctx, val_x, val_y, val_ctx = trainer.load_saved_dataset(base_run_dir)
    run_dir = comparisons_root() / build_run_dir_name(base_run_dir, args.focus_mode, int(args.seed))
    run_dir.mkdir(parents=True, exist_ok=False)

    selected_train_x = np.asarray(train_x[selected_idx], dtype=float)
    selected_train_y = np.asarray(train_y[selected_idx], dtype=int)
    selected_train_ctx = np.asarray(train_ctx[selected_idx], dtype=int)

    np.save(run_dir / "train_x.npy", selected_train_x)
    np.save(run_dir / "train_y.npy", selected_train_y)
    np.save(run_dir / "train_ctx.npy", selected_train_ctx)
    np.save(run_dir / "val_x.npy", np.asarray(val_x, dtype=float))
    np.save(run_dir / "val_y.npy", np.asarray(val_y, dtype=int))
    np.save(run_dir / "val_ctx.npy", np.asarray(val_ctx, dtype=int))
    np.save(run_dir / "hard_indices.npy", np.asarray(hard_idx, dtype=int))
    np.save(run_dir / "replay_indices.npy", np.asarray(replay_idx, dtype=int))
    np.save(run_dir / "selected_indices.npy", np.asarray(selected_idx, dtype=int))

    np.save(run_dir / "0_vg_unique_epoch0.npy", np.asarray(np.load(base_run_dir / f"0_vg_unique_epoch{base_epoch}.npy"), dtype=float))
    edge_active, utility_ema, backbone_ema, edge_age = load_edge_state_for_epoch(base_run_dir, base_epoch, topo)
    np.save(run_dir / "0_edge_active_epoch0.npy", np.asarray(edge_active, dtype=bool))
    np.save(run_dir / "0_edge_utility_epoch0.npy", np.asarray(utility_ema, dtype=float))
    np.save(run_dir / "0_edge_backbone_epoch0.npy", np.asarray(backbone_ema, dtype=float))
    np.save(run_dir / "0_edge_age_epoch0.npy", np.asarray(edge_age, dtype=int))

    total_counter = Counter()
    for word in base_meta["dataset"]["output_vocab"]:
        total_counter[word] = int(json.loads((base_run_dir / "target_coverage_total.json").read_text()).get(word, 0))
    train_counter = Counter(trainer.OUTPUT_ID_TO_WORD[int(v)] for v in selected_train_y.tolist())
    val_counter = Counter(trainer.OUTPUT_ID_TO_WORD[int(v)] for v in np.asarray(val_y, dtype=int).tolist())

    (run_dir / "input_vocab.json").write_text(json.dumps({"input_vocab": trainer.INPUT_VOCAB}, indent=2))
    (run_dir / "output_vocab.json").write_text(json.dumps({"output_vocab": trainer.OUTPUT_VOCAB}, indent=2))
    (run_dir / "token_embed_4d.json").write_text(json.dumps(trainer.TOKEN_EMBED_4D, indent=2))
    copy_if_exists(base_run_dir / "sample_sentences.txt", run_dir / "sample_sentences.txt")
    copy_if_exists(base_run_dir / "netlist_initial.cir", run_dir / "netlist_initial.cir")
    copy_if_exists(base_run_dir / "0.graphml", run_dir / "0.graphml")
    (run_dir / "target_coverage_total.json").write_text(json.dumps(dict(total_counter), indent=2))
    (run_dir / "target_coverage_train.json").write_text(
        json.dumps({k: int(train_counter.get(k, 0)) for k in trainer.OUTPUT_VOCAB}, indent=2)
    )
    (run_dir / "target_coverage_val.json").write_text(
        json.dumps({k: int(val_counter.get(k, 0)) for k in trainer.OUTPUT_VOCAB}, indent=2)
    )
    write_preview_json(run_dir=run_dir, train_ctx=selected_train_ctx, train_y=selected_train_y)

    gamma = float(args.gamma) if float(args.gamma) > 0.0 else float(base_meta["gamma"])
    delta = float(args.delta) if float(args.delta) > 0.0 else float(base_meta["delta"])
    temp = float(args.softmax_temp) if float(args.softmax_temp) > 0.0 else float(base_meta["softmax_temp"])
    sample_max_len = int(args.sample_max_len) if int(args.sample_max_len) > 0 else int(base_meta["generation"]["sample_max_len"])
    fine_tune_meta = json.loads(json.dumps(base_meta))
    fine_tune_meta["script"] = str(Path(__file__).resolve())
    fine_tune_meta["argv"] = list(sys.argv)
    fine_tune_meta["timestamp"] = datetime.now().isoformat()
    fine_tune_meta["seed"] = int(args.seed)
    fine_tune_meta["gamma"] = gamma
    fine_tune_meta["delta"] = delta
    fine_tune_meta["softmax_temp"] = temp
    fine_tune_meta["epochs"] = int(args.epochs)
    fine_tune_meta["train_count"] = int(selected_train_x.shape[0])
    fine_tune_meta["val_count"] = int(np.asarray(val_x).shape[0])
    fine_tune_meta["generation"]["sample_prompts"] = int(args.sample_prompts)
    fine_tune_meta["generation"]["sample_max_len"] = sample_max_len
    fine_tune_meta["execution"]["epoch_process_mode"] = str(args.process_mode)
    fine_tune_meta["execution"]["eval_every"] = int(args.eval_every)
    fine_tune_meta["execution"]["sample_every"] = int(args.sample_every)
    fine_tune_meta["execution"]["plot_every"] = int(args.plot_every)
    fine_tune_meta["fine_tune"] = {
        "mode": "hard_example_replay",
        "base_run_dir": str(base_run_dir.resolve()),
        "base_epoch": int(base_epoch),
        "base_best_val_qmass_mean": float(base_best_qmass),
        "focus_mode": str(args.focus_mode),
        "replay_frac": float(args.replay_frac),
        "hard_count": int(hard_idx.size),
        "replay_count": int(replay_idx.size),
        "selected_count": int(selected_idx.size),
        "train_count_full": int(train_x.shape[0]),
        **{k: float(v) if isinstance(v, (float, np.floating)) else int(v) for k, v in hard_stats.items()},
    }
    (run_dir / "run_meta.json").write_text(json.dumps(fine_tune_meta, indent=2))
    (run_dir / "hard_example_summary.json").write_text(json.dumps(fine_tune_meta["fine_tune"], indent=2))
    return run_dir


def build_worker_args(args: argparse.Namespace, base_meta: dict, run_dir: Path) -> argparse.Namespace:
    device_meta = base_meta.get("device", {})
    backend_meta = base_meta.get("backend", {})
    remodel_meta = backend_meta.get("remodel", {})
    dataset_meta = base_meta.get("dataset", {})
    gamma = float(args.gamma) if float(args.gamma) > 0.0 else float(base_meta["gamma"])
    delta = float(args.delta) if float(args.delta) > 0.0 else float(base_meta["delta"])
    temp = float(args.softmax_temp) if float(args.softmax_temp) > 0.0 else float(base_meta["softmax_temp"])
    return argparse.Namespace(
        seed=int(args.seed),
        epochs=int(args.epochs),
        gamma=gamma,
        delta=delta,
        softmax_temp=temp,
        vminus=float(base_meta["rails"]["vminus"]),
        vplus=float(base_meta["rails"]["vplus"]),
        num_sentences=int(dataset_meta["num_sentences_actual"]),
        min_target_count=int(dataset_meta["min_target_count"]),
        max_sentence_words=int(dataset_meta["max_sentence_words"]),
        val_frac=0.2,
        max_train=0,
        max_val=0,
        template_mode=str(dataset_meta["template_mode"]),
        device_lib=str(device_meta.get("include_path", trainer.DEFAULT_DEVICE_LIB_PATH)),
        body_tie=str(base_meta.get("body_tie", "ground")),
        solver=str(base_meta.get("solver", "klu")),
        vg_init=str(base_meta.get("vg_init", {}).get("mode", "fixed")),
        vg_init_lo=float(base_meta.get("vg_init", {}).get("lo", trainer.VG_CLIP_LO)),
        vg_init_hi=float(base_meta.get("vg_init", {}).get("hi", trainer.VG_CLIP_HI)),
        vg_init_fixed=float(base_meta.get("vg_init", {}).get("fixed", trainer.VG_INIT_SINGLE)),
        sample_prompts=int(args.sample_prompts),
        sample_max_len=int(args.sample_max_len),
        final_val=False,
        process_mode=str(args.process_mode),
        eval_every=int(args.eval_every),
        sample_every=int(args.sample_every),
        plot_every=int(args.plot_every),
        remodel_every_epochs=int(remodel_meta.get("every_epochs", 0)),
        utility_beta=float(remodel_meta.get("utility_beta", 0.999)),
        prune_utility_quantile=float(remodel_meta.get("prune_utility_quantile", 0.10)),
        prune_backbone_quantile=float(remodel_meta.get("prune_backbone_quantile", 0.10)),
        prune_rand_prob=float(remodel_meta.get("prune_rand_prob", 0.0)),
        max_prune_frac=float(remodel_meta.get("max_prune_frac", 0.03)),
        min_edge_age=int(remodel_meta.get("min_edge_age", 3)),
        birth_vg_lo=float(remodel_meta.get("birth_vg_lo", trainer.VG_CLIP_LO)),
        birth_vg_hi=float(remodel_meta.get("birth_vg_hi", trainer.VG_CLIP_LO)),
        worker_run_dir=str(run_dir),
        worker_epoch=-1,
    )


def main() -> None:
    args = parse_args()
    if args.replay_frac < 0.0:
        raise ValueError("--replay-frac must be >= 0")

    if args.base_run_dir:
        base_run_dir = Path(args.base_run_dir).resolve()
        if not base_run_dir.exists():
            raise FileNotFoundError(f"Base run directory not found: {base_run_dir}")
        if args.base_epoch >= 0:
            base_epoch = int(args.base_epoch)
            base_best_qmass = float("nan")
        else:
            base_epoch, base_best_qmass = find_best_epoch_in_run(base_run_dir)
    else:
        base_run_dir, auto_epoch, base_best_qmass = find_best_hiddenpath_run()
        base_epoch = int(args.base_epoch) if args.base_epoch >= 0 else auto_epoch

    base_meta = json.loads((base_run_dir / "run_meta.json").read_text())
    base_temp = float(base_meta["softmax_temp"])
    hard_mask, hard_stats = compute_train_focus_mask(
        run_dir=base_run_dir,
        epoch=base_epoch,
        temp=base_temp,
        focus_mode=str(args.focus_mode),
    )
    hard_idx, replay_idx, selected_idx = build_subset_indices(
        hard_mask=hard_mask,
        seed=int(args.seed),
        replay_frac=float(args.replay_frac),
        max_hard=int(args.max_hard),
    )
    run_dir = prepare_run_dir(
        args=args,
        base_run_dir=base_run_dir,
        base_epoch=base_epoch,
        base_best_qmass=base_best_qmass,
        base_meta=base_meta,
        hard_idx=hard_idx,
        replay_idx=replay_idx,
        selected_idx=selected_idx,
        hard_stats=hard_stats,
    )
    worker_args = build_worker_args(args, base_meta, run_dir)

    print(f"[hard-ft] base_run_dir={base_run_dir}", flush=True)
    print(f"[hard-ft] base_epoch={base_epoch} focus_mode={args.focus_mode} hard_count={hard_idx.size} replay_count={replay_idx.size} selected_count={selected_idx.size}", flush=True)
    print(f"[hard-ft] output_run_dir={run_dir}", flush=True)

    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    for epoch in range(0, int(args.epochs) + 1):
        print(f"[hard-ft] launching worker epoch={epoch}", flush=True)
        worker_args.worker_epoch = int(epoch)
        trainer.run_worker_epoch(worker_args)


if __name__ == "__main__":
    main()
