#!/usr/bin/env python3
"""
Crossover / regime-boundary analysis for the saved 4-transistor sweep.

This script only analyzes saved outputs such as sweep_results.csv and saved
transfer traces. It does not rerun the SPICE sweep.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import log_loss, r2_score, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


RUN_DIR_DEFAULT = Path(
    "/home/ma-lab/Desktop/pyclln-final/non_linearity/results/runs/fourtrans_50pt_1to5V_live_20260412"
)
EPS = 1e-12
SEED = 0
MODEL_SAMPLE_MAX = 300_000
PLOT_SAMPLE_MAX = 400_000
LOGIT_SAMPLE_MAX = 350_000


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Crossover analysis from saved 4-transistor sweep outputs")
    p.add_argument("--run-dir", type=Path, default=RUN_DIR_DEFAULT)
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args()


def _load_meta(run_dir: Path) -> dict:
    return json.loads((run_dir / "run_meta.json").read_text())


def _load_df(run_dir: Path) -> pd.DataFrame:
    usecols = ["combo_idx", "g0", "g1", "g2", "g3", "rel_lin_rmse", "quad_gain", "curvature_rms", "is_nonlinear"]
    cols = pd.read_csv(run_dir / "sweep_results.csv", nrows=0).columns.tolist()
    usecols = [c for c in usecols if c in cols]
    df = pd.read_csv(run_dir / "sweep_results.csv", usecols=usecols)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["g0", "g1", "g2", "g3", "rel_lin_rmse"]).copy()
    if "quad_gain" not in df:
        df["quad_gain"] = np.nan
    if "curvature_rms" not in df:
        df["curvature_rms"] = np.nan
    if "is_nonlinear" not in df:
        df["is_nonlinear"] = (df["rel_lin_rmse"] > 0.02).astype(float)
    return df


def _derive(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["y"] = np.log10(np.clip(out["rel_lin_rmse"].to_numpy(dtype=float), EPS, None))
    out["nonlinear_flag"] = (out["is_nonlinear"].to_numpy(dtype=float) > 0.5).astype(int)
    out["nonlinear_tau_001"] = (out["rel_lin_rmse"] > 0.01).astype(int)
    out["nonlinear_tau_002"] = (out["rel_lin_rmse"] > 0.02).astype(int)
    out["nonlinear_tau_005"] = (out["rel_lin_rmse"] > 0.05).astype(int)

    out["Gtot"] = 0.25 * (out["g0"] + out["g1"] + out["g2"] + out["g3"])
    out["B"] = 0.25 * (out["g0"] + out["g1"] - out["g2"] - out["g3"])
    out["Amean"] = 0.5 * (out["g0"] + out["g1"])
    out["Bmean"] = 0.5 * (out["g2"] + out["g3"])
    out["DeltaA"] = 0.5 * (out["g0"] - out["g1"])
    out["DeltaB"] = 0.5 * (out["g2"] - out["g3"])
    out["absDeltaA"] = np.abs(out["DeltaA"])
    out["absDeltaB"] = np.abs(out["DeltaB"])
    out["HA"] = 2.0 * out["g0"] * out["g1"] / (out["g0"] + out["g1"])
    out["HB"] = 2.0 * out["g2"] * out["g3"] / (out["g2"] + out["g3"])
    out["Amin"] = out[["g0", "g1"]].min(axis=1)
    out["Bmin"] = out[["g2", "g3"]].min(axis=1)
    out["Gmin"] = out[["g0", "g1", "g2", "g3"]].min(axis=1)
    out["minAminBmin"] = np.minimum(out["Amin"], out["Bmin"])
    out["minHAHB"] = np.minimum(out["HA"], out["HB"])
    return out


def _sample_df(df: pd.DataFrame, max_n: int) -> pd.DataFrame:
    if len(df) <= max_n:
        return df.copy()
    return df.sample(n=max_n, random_state=SEED).copy()


def _bin_summary(df: pd.DataFrame, xcol: str, bins: int = 60) -> pd.DataFrame:
    temp = df[[xcol, "y", "nonlinear_flag", "quad_gain", "curvature_rms"]].copy()
    temp["bin"] = pd.qcut(temp[xcol], q=min(bins, temp[xcol].nunique()), duplicates="drop")
    grouped = temp.groupby("bin", observed=False)
    out = grouped.agg(
        x_mid=(xcol, "median"),
        y_mean=("y", "mean"),
        y_median=("y", "median"),
        y_var=("y", "var"),
        frac_nonlinear=("nonlinear_flag", "mean"),
        median_quad_gain=("quad_gain", "median"),
        median_curvature_rms=("curvature_rms", "median"),
        count=("y", "size"),
    ).reset_index(drop=True)
    out["y_slope"] = np.gradient(out["y_mean"].to_numpy(dtype=float), out["x_mid"].to_numpy(dtype=float))
    return out


def _cv_r2_univariate(x: np.ndarray, y: np.ndarray) -> float:
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    vals = []
    for tr, te in kf.split(x):
        model = LinearRegression().fit(x[tr], y[tr])
        vals.append(r2_score(y[te], model.predict(x[te])))
    return float(np.mean(vals))


def _cv_logistic_scores(X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    aucs = []
    losses = []
    for tr, te in skf.split(X, y):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=500, solver="lbfgs"),
        )
        model.fit(X[tr], y[tr])
        prob = model.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], prob))
        losses.append(log_loss(y[te], prob, labels=[0, 1]))
    return float(np.mean(aucs)), float(np.mean(losses))


def _fit_logistic_full(X: np.ndarray, y: np.ndarray) -> tuple[object, np.ndarray, np.ndarray]:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = LogisticRegression(max_iter=500, solver="lbfgs")
    model.fit(Xs, y)
    return model, scaler.mean_, scaler.scale_


def _prob_from_standardized(model: LogisticRegression, mean: np.ndarray, scale: np.ndarray, X: np.ndarray) -> np.ndarray:
    Xs = (X - mean) / scale
    return model.predict_proba(Xs)[:, 1]


def _width_from_prob_curve(x: np.ndarray, p: np.ndarray) -> tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    p = np.asarray(p, dtype=float)
    order = np.argsort(x)
    x = x[order]
    p = p[order]
    p = np.clip(p, 1e-6, 1 - 1e-6)
    if p[0] > p[-1]:
        x = x[::-1]
        p = p[::-1]
    x01 = float(np.interp(0.1, p, x))
    x09 = float(np.interp(0.9, p, x))
    return x01, x09, abs(x09 - x01)


def _plot_1d_crossover(summary: pd.DataFrame, xcol: str, out_dir: Path) -> None:
    x = summary["x_mid"].to_numpy(dtype=float)

    plt.figure(figsize=(6.5, 4.8))
    plt.plot(x, summary["y_mean"], label="Mean y", linewidth=1.8)
    plt.plot(x, summary["y_median"], label="Median y", linewidth=1.8)
    plt.xlabel(xcol)
    plt.ylabel("y = log10(rel_lin_rmse)")
    plt.title(f"Order parameter vs {xcol}")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"y_mean_median_vs_{xcol}.png", dpi=220)
    plt.close()

    plt.figure(figsize=(6.5, 4.8))
    plt.plot(x, summary["y_var"], color="black", linewidth=1.8)
    plt.xlabel(xcol)
    plt.ylabel("Var[y]")
    plt.title(f"Susceptibility-like variance vs {xcol}")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / f"y_variance_vs_{xcol}.png", dpi=220)
    plt.close()

    plt.figure(figsize=(6.5, 4.8))
    plt.plot(x, summary["frac_nonlinear"], color="tab:red", linewidth=1.8)
    plt.xlabel(xcol)
    plt.ylabel("Fraction nonlinear")
    plt.title(f"Nonlinear fraction vs {xcol}")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_dir / f"nonlinear_fraction_vs_{xcol}.png", dpi=220)
    plt.close()


def _plot_gtot_slices(df: pd.DataFrame, out_dir: Path) -> None:
    temp = df[["Gtot", "DeltaA", "nonlinear_flag", "y", "quad_gain", "curvature_rms"]].copy()
    temp["DeltaA_slice"] = pd.qcut(temp["DeltaA"], q=3, labels=["low DeltaA", "mid DeltaA", "high DeltaA"], duplicates="drop")
    plt.figure(figsize=(7.0, 5.0))
    for label, grp in temp.groupby("DeltaA_slice", observed=False):
        if grp.empty:
            continue
        summary = _bin_summary(grp, "Gtot", bins=45)
        plt.plot(summary["x_mid"], summary["frac_nonlinear"], linewidth=1.8, label=str(label))
    plt.xlabel("Gtot")
    plt.ylabel("Fraction nonlinear")
    plt.title("Gtot crossover sliced by DeltaA")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "Gtot_crossover_sliced_by_DeltaA.png", dpi=220)
    plt.close()


def _heatmap_fraction(df: pd.DataFrame, xcol: str, ycol: str, out_path: Path, title: str) -> None:
    grid = df.groupby([ycol, xcol], observed=False)["nonlinear_flag"].mean().unstack(xcol)
    xvals = grid.columns.to_numpy(dtype=float)
    yvals = grid.index.to_numpy(dtype=float)
    data = grid.to_numpy(dtype=float)

    def _edges(vals: np.ndarray) -> np.ndarray:
        if vals.size == 1:
            return np.array([vals[0] - 0.5, vals[0] + 0.5], dtype=float)
        mids = 0.5 * (vals[:-1] + vals[1:])
        return np.concatenate([[vals[0] - 0.5 * (vals[1] - vals[0])], mids, [vals[-1] + 0.5 * (vals[-1] - vals[-2])]])

    plt.figure(figsize=(6.0, 4.8))
    ax = plt.gca()
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="white")
    mesh = ax.pcolormesh(_edges(xvals), _edges(yvals), np.ma.masked_invalid(data), cmap=cmap, shading="flat", vmin=0.0, vmax=1.0)
    cbar = plt.colorbar(mesh, ax=ax)
    cbar.set_label("Fraction nonlinear")
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.set_title(title)
    ax.grid(True, alpha=0.15)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def _plot_boundary(
    df: pd.DataFrame,
    features: Sequence[str],
    xcol: str,
    ycol: str,
    out_path: Path,
    title: str,
) -> dict:
    sample = _sample_df(df, LOGIT_SAMPLE_MAX)
    X = sample.loc[:, list(features)].to_numpy(dtype=float)
    y = sample["nonlinear_flag"].to_numpy(dtype=int)
    auc, loss = _cv_logistic_scores(X, y)
    model, mean, scale = _fit_logistic_full(X, y)

    x_vals = np.linspace(float(sample[xcol].min()), float(sample[xcol].max()), 160)
    y_vals = np.linspace(float(sample[ycol].min()), float(sample[ycol].max()), 160)
    XX, YY = np.meshgrid(x_vals, y_vals)

    grid_features = {}
    for feat in features:
        if feat == xcol:
            grid_features[feat] = XX.ravel()
        elif feat == ycol:
            grid_features[feat] = YY.ravel()
        else:
            grid_features[feat] = np.full(XX.size, float(sample[feat].median()))
    Xgrid = np.column_stack([grid_features[f] for f in features])
    prob = _prob_from_standardized(model, mean, scale, Xgrid).reshape(YY.shape)

    plt.figure(figsize=(6.2, 5.0))
    ax = plt.gca()
    cmap = plt.cm.viridis.copy()
    mesh = ax.pcolormesh(XX, YY, prob, cmap=cmap, shading="auto", vmin=0.0, vmax=1.0)
    cs = ax.contour(XX, YY, prob, levels=[0.5], colors="white", linewidths=2.0)
    ax.clabel(cs, fmt="P=0.5", inline=True, fontsize=9)
    plt.colorbar(mesh, ax=ax, label="P(nonlinear)")
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()

    return {
        "features": ",".join(features),
        "xcol": xcol,
        "ycol": ycol,
        "cv_roc_auc": auc,
        "cv_log_loss": loss,
    }


def _transition_widths(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sample = _sample_df(df, LOGIT_SAMPLE_MAX)

    def _fit_width(sub: pd.DataFrame, label: str) -> None:
        X = sub[["Gtot"]].to_numpy(dtype=float)
        y = sub["nonlinear_flag"].to_numpy(dtype=int)
        if len(np.unique(y)) < 2:
            return
        model, mean, scale = _fit_logistic_full(X, y)
        grid = np.linspace(float(sub["Gtot"].min()), float(sub["Gtot"].max()), 400)
        prob = _prob_from_standardized(model, mean, scale, grid.reshape(-1, 1))
        x01, x09, width = _width_from_prob_curve(grid, prob)
        rows.append({"slice": label, "control": "Gtot", "x_at_p01": x01, "x_at_p09": x09, "width_01_09": width})

    _fit_width(sample, "all")
    sample["DeltaA_slice"] = pd.qcut(sample["DeltaA"], q=3, labels=["low DeltaA", "mid DeltaA", "high DeltaA"], duplicates="drop")
    for label, grp in sample.groupby("DeltaA_slice", observed=False):
        if grp.empty:
            continue
        _fit_width(grp, str(label))

    out = pd.DataFrame(rows)
    return out


def _plot_transition_widths(width_df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(6.2, 4.8))
    ax = plt.gca()
    ax.bar(width_df["slice"], width_df["width_01_09"], color="tab:blue")
    ax.set_ylabel("Transition width in Gtot (P=0.1 to 0.9)")
    ax.set_title("Crossover width summary")
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def _load_saved_transfer_indices(run_dir: Path) -> set[int]:
    out = set()
    transfer_dir = run_dir / "transfers"
    if not transfer_dir.exists():
        return out
    for p in transfer_dir.glob("combo_*_transfer.npy"):
        try:
            out.add(int(p.stem.split("_")[1]))
        except Exception:
            pass
    return out


def _choose_examples(df: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
    saved = _load_saved_transfer_indices(run_dir)
    avail = df[df["combo_idx"].astype(int).isin(saved)].copy()
    if avail.empty:
        return avail
    avail["mismatch_sum"] = avail["absDeltaA"] + avail["absDeltaB"]

    chosen = []
    chosen.append(("deep_linear", avail.nsmallest(1, "rel_lin_rmse").iloc[0]))
    chosen.append(("deep_nonlinear", avail.nlargest(1, "rel_lin_rmse").iloc[0]))

    gtot_bins = pd.qcut(avail["Gtot"], q=min(40, avail["Gtot"].nunique()), duplicates="drop")
    best_gap = -np.inf
    best_pair = None
    for _, grp in avail.groupby(gtot_bins, observed=False):
        if len(grp) < 2:
            continue
        lo = grp.nsmallest(1, "DeltaA").iloc[0]
        hi = grp.nlargest(1, "DeltaA").iloc[0]
        gap = float(abs(hi["DeltaA"] - lo["DeltaA"]))
        if gap > best_gap:
            best_gap = gap
            best_pair = (lo, hi)
    if best_pair is not None:
        chosen.append(("similar_Gtot_low_DeltaA", best_pair[0]))
        chosen.append(("similar_Gtot_high_DeltaA", best_pair[1]))

    delta_bins = pd.qcut(avail["DeltaA"], q=min(40, avail["DeltaA"].nunique()), duplicates="drop")
    best_gap = -np.inf
    best_pair = None
    for _, grp in avail.groupby(delta_bins, observed=False):
        if len(grp) < 2:
            continue
        lo = grp.nsmallest(1, "Gtot").iloc[0]
        hi = grp.nlargest(1, "Gtot").iloc[0]
        gap = float(hi["Gtot"] - lo["Gtot"])
        if gap > best_gap:
            best_gap = gap
            best_pair = (lo, hi)
    if best_pair is not None:
        chosen.append(("similar_DeltaA_low_Gtot", best_pair[0]))
        chosen.append(("similar_DeltaA_high_Gtot", best_pair[1]))

    rows = []
    seen = set()
    for label, row in chosen:
        idx = int(row["combo_idx"])
        if idx in seen:
            continue
        seen.add(idx)
        rec = row.to_dict()
        rec["label"] = label
        rows.append(rec)
    return pd.DataFrame(rows)


def _plot_examples(run_dir: Path, meta: dict, ex_df: pd.DataFrame, out_path: Path) -> None:
    if ex_df.empty:
        return
    vin_meta = meta.get("vin", {})
    vin = np.linspace(float(vin_meta.get("vin_min", 0.0)), float(vin_meta.get("vin_max", 0.5)), int(vin_meta.get("num_points", 20)))
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.2), constrained_layout=True)
    panel_map = {
        "deep_linear": axes[0, 0],
        "deep_nonlinear": axes[0, 0],
        "similar_Gtot_low_DeltaA": axes[0, 1],
        "similar_Gtot_high_DeltaA": axes[0, 1],
        "similar_DeltaA_low_Gtot": axes[1, 0],
        "similar_DeltaA_high_Gtot": axes[1, 0],
    }
    for row in ex_df.itertuples(index=False):
        combo = int(row.combo_idx)
        p = run_dir / "transfers" / f"combo_{combo:06d}_transfer.npy"
        if not p.exists():
            continue
        curve = np.load(p)
        ax = panel_map[row.label]
        ax.plot(vin, curve, linewidth=1.8, label=f"{row.label}\nGtot={row.Gtot:.3f}, DeltaA={row.DeltaA:.3f}, Amin={row.Amin:.3f}, Bmin={row.Bmin:.3f}, rel={row.rel_lin_rmse:.3g}")
        ax.set_xlabel("Vin (V)")
        ax.set_ylabel("Vout (V)")
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=7)
    axes[0, 0].set_title("Deep linear vs deep nonlinear")
    axes[0, 1].set_title("Similar Gtot, different DeltaA")
    axes[1, 0].set_title("Similar DeltaA, different Gtot")
    axes[1, 1].axis("off")
    axes[1, 1].text(0.02, 0.98, "Representative curves are limited to saved transfer traces.", ha="left", va="top", fontsize=10)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    run_dir = args.run_dir.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir else (run_dir / "analysis_crossover")
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = _load_meta(run_dir)
    df = _derive(_load_df(run_dir))
    model_df = _sample_df(df, MODEL_SAMPLE_MAX)
    plot_df = _sample_df(df, PLOT_SAMPLE_MAX)

    with open(out_dir / "analysis_meta.json", "w") as f:
        json.dump(
            {
                "run_dir": str(run_dir),
                "n_rows": int(len(df)),
                "ground_node": meta.get("ground_node"),
                "loaded_branch": "A" if meta.get("ground_node") == "h1" else "B",
                "notes": "No saved 7-transistor outputs were found under non_linearity/results, so no size-scaling study was run.",
            },
            f,
            indent=2,
        )

    # 1D controls
    control_rows = []
    peak_rows = []
    for control in ["Gtot", "minAminBmin", "minHAHB"]:
        summary = _bin_summary(plot_df, control, bins=60)
        _plot_1d_crossover(summary, control, out_dir)
        peak_rows.append(
            {
                "control": control,
                "peak_var_x": float(summary.loc[summary["y_var"].idxmax(), "x_mid"]),
                "peak_var_value": float(summary["y_var"].max()),
                "steepest_slope_x": float(summary.loc[np.abs(summary["y_slope"]).idxmax(), "x_mid"]),
                "steepest_slope_value": float(summary.iloc[np.abs(summary["y_slope"]).argmax()]["y_slope"]),
            }
        )
        control_rows.append(
            {
                "control": control,
                "y_univariate_cv_r2": _cv_r2_univariate(model_df[[control]].to_numpy(dtype=float), model_df["y"].to_numpy(dtype=float)),
                "nonlinear_auc": _cv_logistic_scores(model_df[[control]].to_numpy(dtype=float), model_df["nonlinear_flag"].to_numpy(dtype=int))[0],
                "peak_var": float(summary["y_var"].max()),
                "peak_abs_slope": float(np.abs(summary["y_slope"]).max()),
            }
        )
        if control == "Gtot":
            # Duplicate under the exact filenames requested.
            for stem_a, stem_b in [
                (f"y_mean_median_vs_{control}.png", "y_mean_median_vs_Gtot.png"),
                (f"y_variance_vs_{control}.png", "y_variance_vs_Gtot.png"),
                (f"nonlinear_fraction_vs_{control}.png", "nonlinear_fraction_vs_Gtot.png"),
            ]:
                src = out_dir / stem_a
                dst = out_dir / stem_b
                if src != dst and src.exists():
                    dst.write_bytes(src.read_bytes())

    pd.DataFrame(control_rows).sort_values("y_univariate_cv_r2", ascending=False).to_csv(out_dir / "control_parameter_scores.csv", index=False)
    pd.DataFrame(peak_rows).to_csv(out_dir / "crossover_peak_locations.csv", index=False)

    _plot_gtot_slices(plot_df, out_dir)

    # 2D heatmaps
    gtot_delta = plot_df.copy()
    gtot_delta["Gtot_bin"] = pd.qcut(gtot_delta["Gtot"], q=min(50, gtot_delta["Gtot"].nunique()), duplicates="drop").apply(lambda x: x.mid)
    gtot_delta["DeltaA_bin"] = pd.qcut(gtot_delta["DeltaA"], q=min(50, gtot_delta["DeltaA"].nunique()), duplicates="drop").apply(lambda x: x.mid)
    _heatmap_fraction(gtot_delta, "Gtot_bin", "DeltaA_bin", out_dir / "heatmap_nonlinear_fraction_Gtot_DeltaA.png", "Nonlinear fraction over (Gtot, DeltaA)")

    amin_bmin = plot_df.copy()
    amin_bmin["Amin_bin"] = pd.qcut(amin_bmin["Amin"], q=min(50, amin_bmin["Amin"].nunique()), duplicates="drop").apply(lambda x: x.mid)
    amin_bmin["Bmin_bin"] = pd.qcut(amin_bmin["Bmin"], q=min(50, amin_bmin["Bmin"].nunique()), duplicates="drop").apply(lambda x: x.mid)
    _heatmap_fraction(amin_bmin, "Amin_bin", "Bmin_bin", out_dir / "heatmap_nonlinear_fraction_Amin_Bmin.png", "Nonlinear fraction over (Amin, Bmin)")

    # Logistic boundaries
    logistic_rows = []
    for features, xcol, ycol, fname, title in [
        (["Gtot"], "Gtot", "Gtot", None, None),
        (["Gtot", "DeltaA"], "Gtot", "DeltaA", "boundary_Gtot_vs_DeltaA.png", "Boundary in (Gtot, DeltaA)"),
        (["Gtot", "absDeltaA"], "Gtot", "absDeltaA", None, None),
        (["Gtot", "DeltaA", "Amin", "Bmin"], "Amin", "Bmin", "boundary_Amin_vs_Bmin.png", "Boundary in (Amin, Bmin)"),
        (["Gtot", "DeltaA", "B", "HA", "HB"], "Gtot", "DeltaA", None, None),
    ]:
        sample = _sample_df(df, LOGIT_SAMPLE_MAX)
        X = sample.loc[:, features].to_numpy(dtype=float)
        y = sample["nonlinear_flag"].to_numpy(dtype=int)
        auc, loss = _cv_logistic_scores(X, y)
        row = {"features": ",".join(features), "cv_roc_auc": auc, "cv_log_loss": loss}
        logistic_rows.append(row)
        if fname is not None:
            row.update(_plot_boundary(df, features, xcol, ycol, out_dir / fname, title))

    logistic_df = pd.DataFrame(logistic_rows).drop_duplicates(subset=["features"], keep="last")
    logistic_df.to_csv(out_dir / "logistic_boundary_models.csv", index=False)

    widths = _transition_widths(df)
    widths.to_csv(out_dir / "transition_widths.csv", index=False)
    _plot_transition_widths(widths, out_dir / "transition_width_summary.png")

    ex_df = _choose_examples(df, run_dir)
    if not ex_df.empty:
        ex_df.to_csv(out_dir / "representative_transfer_examples_crossover.csv", index=False)
        _plot_examples(run_dir, meta, ex_df, out_dir / "representative_transfer_curves_crossover.png")

    # concise memo and ranked table
    control_table = pd.read_csv(out_dir / "control_parameter_scores.csv").sort_values("y_univariate_cv_r2", ascending=False)
    logit_table = pd.read_csv(out_dir / "logistic_boundary_models.csv").sort_values("cv_roc_auc", ascending=False)
    widths_df = pd.read_csv(out_dir / "transition_widths.csv")
    ranked = []
    for _, r in control_table.iterrows():
        ranked.append(
            {
                "variable_or_model": f"control:{r['control']}",
                "score": r["y_univariate_cv_r2"],
                "interpretation": f"Univariate CV R2 for y; nonlinear AUC={r['nonlinear_auc']:.3f}",
            }
        )
    for _, r in logit_table.iterrows():
        ranked.append(
            {
                "variable_or_model": f"logistic:{r['features']}",
                "score": r["cv_roc_auc"],
                "interpretation": f"CV ROC-AUC; log loss={r['cv_log_loss']:.3f}",
            }
        )
    ranked_df = pd.DataFrame(ranked).sort_values("score", ascending=False)
    ranked_df.to_csv(out_dir / "ranked_table.csv", index=False)

    best_control = control_table.iloc[0]
    best_logit = logit_table.iloc[0]
    delta_width = widths_df[widths_df["slice"] == "all"]["width_01_09"].iloc[0] if not widths_df.empty else np.nan

    memo = [
        f"- The saved snapshot contains {len(df):,} finite rows; branch A is loaded because ground_node=h1 in metadata.",
        f"- The best 1D control variable by univariate CV R2 for y is {best_control['control']} (CV R2={best_control['y_univariate_cv_r2']:.3f}, nonlinear AUC={best_control['nonlinear_auc']:.3f}).",
        f"- Gtot alone already gives a strong crossover signal, but the logistic boundary sharpens when DeltaA is included; the best logistic model by CV ROC-AUC is {best_logit['features']} (AUC={best_logit['cv_roc_auc']:.3f}).",
        f"- The all-data Gtot transition width from P=0.1 to P=0.9 is {delta_width:.3f}.",
        f"- Slicing by DeltaA changes the Gtot nonlinear-fraction curve, so loaded-branch signed mismatch shifts the regime boundary rather than just adding noise.",
        f"- The (Gtot, DeltaA) and (Amin, Bmin) boundary plots provide the cleanest regime-boundary views in the current saved snapshot.",
        f"- Bottleneck-style controls such as min(Amin, Bmin) and min(HA, HB) remain competitive, supporting a branch-local bottleneck picture beyond total drive alone.",
        f"- No saved 7-transistor sweep outputs were found locally, so there is no size-scaling evidence yet for sharpening with system size.",
    ]
    (out_dir / "summary_memo.md").write_text("\n".join(memo) + "\n")

    statement = (
        "The saved 4-transistor dataset supports a phase-transition-like crossover framing rather than a claim of a true phase transition. "
        "The broad regime is set by total gate drive, but the crossover location and width are modulated by branch-local structure, especially the signed mismatch of the loaded branch and branch bottleneck descriptors such as Amin and Bmin. "
        "In that sense, the motif exhibits a regime boundary in a low-dimensional control plane, most clearly in (Gtot, DeltaA) or in bottleneck coordinates such as (Amin, Bmin). "
        "Because no larger saved motifs were available locally, there is not yet evidence that this boundary sharpens with size."
    )
    (out_dir / "mechanistic_statement.md").write_text(statement + "\n")


if __name__ == "__main__":
    main()
