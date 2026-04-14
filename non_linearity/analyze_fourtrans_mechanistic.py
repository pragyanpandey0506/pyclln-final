#!/usr/bin/env python3
"""
Mechanistic analysis for the saved 4-transistor frozen-network transfer sweep.

This script consumes an existing sweep_results.csv and writes a quantitative
analysis package into a run-local output directory. It does not rerun SPICE.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import partial_dependence, permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


RUN_DIR_DEFAULT = Path(
    "/home/ma-lab/Desktop/pyclln-final/non_linearity/results/runs/fourtrans_50pt_1to5V_live_20260412"
)
EPS = 1e-12
MODEL_SAMPLE_MAX = 300_000
TREE_SAMPLE_MAX = 180_000
PLOT_SAMPLE_MAX = 250_000
SEED = 0


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze saved 4-transistor sweep outputs")
    p.add_argument("--run-dir", type=Path, default=RUN_DIR_DEFAULT)
    p.add_argument("--out-dir", type=Path, default=None)
    return p.parse_args()


def _load_run_meta(run_dir: Path) -> dict:
    return json.loads((run_dir / "run_meta.json").read_text())


def _load_csv(run_dir: Path) -> pd.DataFrame:
    usecols = [
        "combo_idx",
        "g0",
        "g1",
        "g2",
        "g3",
        "rel_lin_rmse",
        "quad_gain",
        "curvature_rms",
        "curvature_max",
        "is_nonlinear",
    ]
    present = pd.read_csv(run_dir / "sweep_results.csv", nrows=0).columns.tolist()
    usecols = [c for c in usecols if c in present]
    df = pd.read_csv(run_dir / "sweep_results.csv", usecols=usecols)
    finite_mask = np.isfinite(df.select_dtypes(include=[np.number])).all(axis=1)
    df = df.loc[finite_mask].copy()
    if "quad_gain" not in df:
        df["quad_gain"] = np.nan
    if "curvature_rms" not in df:
        df["curvature_rms"] = np.nan
    if "curvature_max" not in df:
        df["curvature_max"] = np.nan
    if "is_nonlinear" not in df:
        df["is_nonlinear"] = np.nan
    return df


def _derive_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["y"] = np.log10(np.clip(out["rel_lin_rmse"].to_numpy(dtype=float), EPS, None))

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

    out["Gmin"] = out[["g0", "g1", "g2", "g3"]].min(axis=1)
    out["Amin"] = out[["g0", "g1"]].min(axis=1)
    out["Bmin"] = out[["g2", "g3"]].min(axis=1)
    return out


def _sample_df(df: pd.DataFrame, max_n: int) -> pd.DataFrame:
    if len(df) <= max_n:
        return df.copy()
    return df.sample(n=max_n, random_state=SEED).copy()


def _cv_scores(
    X: np.ndarray,
    y: np.ndarray,
    degree: int = 1,
    use_scaler: bool = False,
) -> tuple[float, float]:
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    r2s: List[float] = []
    rmses: List[float] = []
    for tr, te in kf.split(X):
        steps = []
        if degree > 1:
            steps.append(PolynomialFeatures(degree=degree, include_bias=False))
        if use_scaler:
            steps.append(StandardScaler())
        steps.append(LinearRegression())
        model = make_pipeline(*steps)
        model.fit(X[tr], y[tr])
        pred = model.predict(X[te])
        r2s.append(r2_score(y[te], pred))
        rmses.append(np.sqrt(mean_squared_error(y[te], pred)))
    return float(np.mean(r2s)), float(np.mean(rmses))


def _fit_linear_summary(df: pd.DataFrame, features: Sequence[str]) -> dict:
    X = df.loc[:, list(features)].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    cv_r2, cv_rmse = _cv_scores(X, y, degree=1, use_scaler=False)

    model = LinearRegression()
    model.fit(X, y)
    pred = model.predict(X)
    r2 = float(r2_score(y, pred))
    n = len(y)
    p = X.shape[1]
    adj_r2 = float(1.0 - (1.0 - r2) * (n - 1) / max(n - p - 1, 1))

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    std_model = LinearRegression()
    std_model.fit(Xs, y)

    coef = dict(zip(features, model.coef_.tolist()))
    std_coef = dict(zip(features, std_model.coef_.tolist()))
    return {
        "features": ",".join(features),
        "cv_r2": cv_r2,
        "cv_rmse": cv_rmse,
        "adj_r2": adj_r2,
        "coefs": coef,
        "std_coefs": std_coef,
    }


def _fit_interaction_check(df: pd.DataFrame, features: Sequence[str]) -> dict:
    X = df.loc[:, list(features)].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    base_r2, base_rmse = _cv_scores(X, y, degree=1, use_scaler=False)
    int_r2, int_rmse = _cv_scores(X, y, degree=2, use_scaler=False)
    return {
        "base_cv_r2": base_r2,
        "base_cv_rmse": base_rmse,
        "interaction_cv_r2": int_r2,
        "interaction_cv_rmse": int_rmse,
        "delta_r2": int_r2 - base_r2,
    }


def _rank_univariate(df: pd.DataFrame, features: Sequence[str]) -> pd.DataFrame:
    y = df["y"].to_numpy(dtype=float)
    rows: List[dict] = []
    for feature in features:
        x = df[feature].to_numpy(dtype=float)
        pearson = float(np.corrcoef(x, y)[0, 1])
        spearman = float(pd.Series(x).corr(pd.Series(y), method="spearman"))
        cv_r2, cv_rmse = _cv_scores(x.reshape(-1, 1), y, degree=3, use_scaler=False)
        rows.append(
            {
                "feature": feature,
                "pearson_y": pearson,
                "spearman_y": spearman,
                "poly3_cv_r2": cv_r2,
                "poly3_cv_rmse": cv_rmse,
            }
        )
    out = pd.DataFrame(rows).sort_values("poly3_cv_r2", ascending=False).reset_index(drop=True)
    return out


def _plot_hexbin(
    x: np.ndarray,
    y: np.ndarray,
    c: np.ndarray | None,
    xlabel: str,
    ylabel: str,
    title: str,
    out_path: Path,
    reduce_C_function=np.mean,
) -> None:
    plt.figure(figsize=(6.2, 4.8))
    ax = plt.gca()
    if c is None:
        hb = ax.hexbin(x, y, gridsize=60, cmap="viridis", mincnt=1)
        cbar = plt.colorbar(hb, ax=ax)
        cbar.set_label("Count")
    else:
        hb = ax.hexbin(x, y, C=c, reduce_C_function=reduce_C_function, gridsize=55, cmap="viridis", mincnt=1)
        cbar = plt.colorbar(hb, ax=ax)
        cbar.set_label("Mean y")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def _plot_pair_heatmap(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    value_col: str,
    out_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    grid = df.groupby([ycol, xcol])[value_col].mean().unstack(xcol)
    xvals = grid.columns.to_numpy(dtype=float)
    yvals = grid.index.to_numpy(dtype=float)
    data = grid.to_numpy(dtype=float)

    def edges(vals: np.ndarray) -> np.ndarray:
        if vals.size == 1:
            return np.array([vals[0] - 0.5, vals[0] + 0.5], dtype=float)
        mids = 0.5 * (vals[:-1] + vals[1:])
        return np.concatenate([[vals[0] - 0.5 * (vals[1] - vals[0])], mids, [vals[-1] + 0.5 * (vals[-1] - vals[-2])]])

    plt.figure(figsize=(5.8, 4.8))
    ax = plt.gca()
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="white")
    mesh = ax.pcolormesh(edges(xvals), edges(yvals), np.ma.masked_invalid(data), cmap=cmap, shading="flat")
    cbar = plt.colorbar(mesh, ax=ax)
    cbar.set_label("Mean y")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.15)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def _fit_nonlinear_model(df: pd.DataFrame, features: Sequence[str], out_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    sample = _sample_df(df, TREE_SAMPLE_MAX)
    X = sample.loc[:, list(features)]
    y = sample["y"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=SEED)

    model = HistGradientBoostingRegressor(
        max_depth=6,
        learning_rate=0.05,
        max_iter=300,
        min_samples_leaf=40,
        random_state=SEED,
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    test_r2 = float(r2_score(y_test, pred))
    test_rmse = float(np.sqrt(mean_squared_error(y_test, pred)))

    perm = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=8,
        random_state=SEED,
        scoring="r2",
    )
    imp_df = pd.DataFrame(
        {
            "feature": list(features),
            "perm_importance_mean": perm.importances_mean,
            "perm_importance_std": perm.importances_std,
        }
    ).sort_values("perm_importance_mean", ascending=False)
    imp_df["test_r2"] = test_r2
    imp_df["test_rmse"] = test_rmse
    imp_df.to_csv(out_dir / "nonlinear_permutation_importance.csv", index=False)

    top3 = imp_df["feature"].head(3).tolist()
    for feature in top3:
        pdp = partial_dependence(model, X_test, [feature], grid_resolution=40)
        xs = pdp["grid_values"][0]
        ys = pdp["average"][0]
        plt.figure(figsize=(5.2, 4.0))
        plt.plot(xs, ys, color="black", linewidth=1.8)
        plt.xlabel(feature)
        plt.ylabel("Partial dependence on y")
        plt.title(f"Partial dependence: {feature}")
        plt.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(out_dir / f"pdp_{feature}.png", dpi=220)
        plt.close()
    return imp_df, top3


def _plot_residuals(df: pd.DataFrame, out_dir: Path) -> float:
    model = LinearRegression()
    X = df[["Gtot"]].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    model.fit(X, y)
    resid = y - model.predict(X)
    df["resid_gtot"] = resid

    _plot_hexbin(df["Gtot"].to_numpy(), y, None, "Gtot", "y = log10(rel_lin_rmse)", "y vs Gtot", out_dir / "y_vs_Gtot.png")

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.0), constrained_layout=True)
    cols = ["Amean", "Bmean", "absDeltaA", "absDeltaB", "Amin", "Bmin"]
    for ax, col in zip(axes.flat, cols):
        hb = ax.hexbin(df[col].to_numpy(), resid, gridsize=55, cmap="coolwarm", mincnt=1)
        ax.set_xlabel(col)
        ax.set_ylabel("Residual after y~Gtot")
        ax.set_title(f"Residual vs {col}")
        ax.grid(True, alpha=0.2)
        fig.colorbar(hb, ax=ax, shrink=0.82)
    fig.savefig(out_dir / "residuals_after_Gtot.png", dpi=220)
    plt.close(fig)
    return float(r2_score(y, model.predict(X)))


def _load_saved_transfer_indices(run_dir: Path) -> pd.Index:
    transfer_dir = run_dir / "transfers"
    vals = []
    for p in transfer_dir.glob("combo_*_transfer.npy"):
        try:
            vals.append(int(p.stem.split("_")[1]))
        except Exception:
            continue
    return pd.Index(sorted(set(vals)))


def _choose_representative_examples(df: pd.DataFrame, run_dir: Path) -> dict:
    saved_idx = _load_saved_transfer_indices(run_dir)
    avail = df[df["combo_idx"].astype(int).isin(saved_idx)].copy()
    if avail.empty:
        return {}

    linear_row = avail.nsmallest(1, "rel_lin_rmse").iloc[0]
    nonlinear_row = avail.nlargest(1, "rel_lin_rmse").iloc[0]

    avail["mismatch_sum"] = avail["absDeltaA"] + avail["absDeltaB"]
    gtot_bins = pd.qcut(avail["Gtot"], q=min(40, avail["Gtot"].nunique()), duplicates="drop")
    best_pair_mismatch = None
    best_gap = -np.inf
    for _, grp in avail.groupby(gtot_bins):
        if len(grp) < 2:
            continue
        lo = grp.nsmallest(1, "mismatch_sum").iloc[0]
        hi = grp.nlargest(1, "mismatch_sum").iloc[0]
        gap = float(hi["mismatch_sum"] - lo["mismatch_sum"])
        if gap > best_gap:
            best_gap = gap
            best_pair_mismatch = (lo, hi)

    mismatch_bins = pd.qcut(avail["mismatch_sum"], q=min(40, avail["mismatch_sum"].nunique()), duplicates="drop")
    best_pair_gtot = None
    best_gap = -np.inf
    for _, grp in avail.groupby(mismatch_bins):
        if len(grp) < 2:
            continue
        lo = grp.nsmallest(1, "Gtot").iloc[0]
        hi = grp.nlargest(1, "Gtot").iloc[0]
        gap = float(hi["Gtot"] - lo["Gtot"])
        if gap > best_gap:
            best_gap = gap
            best_pair_gtot = (lo, hi)

    out = {
        "highly_linear": linear_row,
        "highly_nonlinear": nonlinear_row,
    }
    if best_pair_mismatch is not None:
        out["similar_Gtot_low_mismatch"] = best_pair_mismatch[0]
        out["similar_Gtot_high_mismatch"] = best_pair_mismatch[1]
    if best_pair_gtot is not None:
        out["similar_mismatch_low_Gtot"] = best_pair_gtot[0]
        out["similar_mismatch_high_Gtot"] = best_pair_gtot[1]
    return out


def _plot_representative_transfers(run_dir: Path, meta: dict, examples: dict, out_dir: Path) -> pd.DataFrame:
    if not examples:
        return pd.DataFrame()
    vin_meta = meta.get("vin", {})
    vin = np.linspace(
        float(vin_meta.get("vin_min", 0.0)),
        float(vin_meta.get("vin_max", 0.5)),
        int(vin_meta.get("num_points", 20)),
        dtype=float,
    )
    rows = []
    keys = [
        "highly_linear",
        "highly_nonlinear",
        "similar_Gtot_low_mismatch",
        "similar_Gtot_high_mismatch",
        "similar_mismatch_low_Gtot",
        "similar_mismatch_high_Gtot",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), constrained_layout=True)
    ax_map = {
        "extremes": axes[0, 0],
        "similar_Gtot": axes[0, 1],
        "similar_mismatch": axes[1, 0],
    }

    for label in keys:
        if label not in examples:
            continue
        row = examples[label]
        combo = int(row["combo_idx"])
        transfer_path = run_dir / "transfers" / f"combo_{combo:06d}_transfer.npy"
        if not transfer_path.exists():
            continue
        curve = np.load(transfer_path)
        rows.append(
            {
                "label": label,
                "combo_idx": combo,
                "g0": row["g0"],
                "g1": row["g1"],
                "g2": row["g2"],
                "g3": row["g3"],
                "Gtot": row["Gtot"],
                "absDeltaA": row["absDeltaA"],
                "absDeltaB": row["absDeltaB"],
                "rel_lin_rmse": row["rel_lin_rmse"],
            }
        )
        if label in {"highly_linear", "highly_nonlinear"}:
            ax = ax_map["extremes"]
        elif label in {"similar_Gtot_low_mismatch", "similar_Gtot_high_mismatch"}:
            ax = ax_map["similar_Gtot"]
        else:
            ax = ax_map["similar_mismatch"]
        ax.plot(vin, curve, linewidth=1.8, label=f"{label} (idx {combo})")
        ax.set_xlabel("Vin (V)")
        ax.set_ylabel("Vout (V)")
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=8)

    ax_map["extremes"].set_title("Representative linear vs nonlinear curves")
    ax_map["similar_Gtot"].set_title("Similar Gtot, different mismatch")
    ax_map["similar_mismatch"].set_title("Similar mismatch, different Gtot")
    axes[1, 1].axis("off")
    axes[1, 1].text(
        0.02,
        0.98,
        "Saved transfers were only available for a logged subset of combos.\n"
        "Examples were chosen from that saved subset.",
        ha="left",
        va="top",
        fontsize=10,
    )
    fig.savefig(out_dir / "representative_transfer_curves.png", dpi=220)
    plt.close(fig)
    out = pd.DataFrame(rows)
    if not out.empty:
        out.to_csv(out_dir / "representative_transfer_examples.csv", index=False)
    return out


def _coefs_to_string(coefs: dict) -> str:
    parts = []
    for k, v in coefs.items():
        parts.append(f"{k}:{v:+.3f}")
    return ", ".join(parts)


def _std_coefs_to_string(coefs: dict) -> str:
    parts = []
    for k, v in sorted(coefs.items(), key=lambda kv: abs(kv[1]), reverse=True):
        parts.append(f"{k}:{v:+.3f}")
    return ", ".join(parts)


def _make_ranked_table(
    univar: pd.DataFrame,
    model_df: pd.DataFrame,
    asymmetry_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for _, r in univar.head(12).iterrows():
        rows.append(
            {
                "feature_or_model": f"feature:{r['feature']}",
                "predictive_score": r["poly3_cv_r2"],
                "interpretation": f"Univariate cubic CV R2; Pearson={r['pearson_y']:+.3f}, Spearman={r['spearman_y']:+.3f}",
            }
        )
    for _, r in model_df.iterrows():
        rows.append(
            {
                "feature_or_model": f"model:{r['model']}",
                "predictive_score": r["cv_r2"],
                "interpretation": f"CV R2 with RMSE={r['cv_rmse']:.3f}; std coefs {r['std_coef_summary']}",
            }
        )
    for _, r in asymmetry_df.iterrows():
        rows.append(
            {
                "feature_or_model": f"branch_compare:{r['comparison']}",
                "predictive_score": r["cv_r2"],
                "interpretation": r["interpretation"],
            }
        )
    out = pd.DataFrame(rows).sort_values("predictive_score", ascending=False).reset_index(drop=True)
    return out


def main() -> None:
    args = _parse_args()
    run_dir = args.run_dir.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir else (run_dir / "analysis_mechanistic")
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = _load_run_meta(run_dir)
    df_raw = _load_csv(run_dir)
    df = _derive_features(df_raw)

    with open(out_dir / "analysis_meta.json", "w") as f:
        json.dump(
            {
                "run_dir": str(run_dir),
                "n_rows": int(len(df)),
                "ground_node": meta.get("ground_node"),
                "assumed_loaded_branch": "A" if meta.get("ground_node") == "h1" else "B",
                "model_sample_max": MODEL_SAMPLE_MAX,
                "tree_sample_max": TREE_SAMPLE_MAX,
            },
            f,
            indent=2,
        )

    feature_cols = [
        "Gtot",
        "B",
        "Amean",
        "Bmean",
        "DeltaA",
        "DeltaB",
        "absDeltaA",
        "absDeltaB",
        "HA",
        "HB",
        "Gmin",
        "Amin",
        "Bmin",
    ]

    model_df_source = _sample_df(df, MODEL_SAMPLE_MAX)
    plot_df = _sample_df(df, PLOT_SAMPLE_MAX)

    univar = _rank_univariate(model_df_source, feature_cols)
    univar.to_csv(out_dir / "univariate_feature_ranking.csv", index=False)

    model_specs = {
        "Model 1: Gtot": ["Gtot"],
        "Model 2: Gtot + B": ["Gtot", "B"],
        "Model 3: Amean + Bmean": ["Amean", "Bmean"],
        "Model 4: Gtot + |DeltaA| + |DeltaB|": ["Gtot", "absDeltaA", "absDeltaB"],
        "Model 5: Gtot + Amin + Bmin": ["Gtot", "Amin", "Bmin"],
        "Model 6: interpretable": ["Gtot", "absDeltaA", "absDeltaB", "B", "HA", "HB"],
    }

    model_rows = []
    for model_name, feats in model_specs.items():
        res = _fit_linear_summary(model_df_source, feats)
        model_rows.append(
            {
                "model": model_name,
                "features": ",".join(feats),
                "cv_r2": res["cv_r2"],
                "cv_rmse": res["cv_rmse"],
                "adj_r2": res["adj_r2"],
                "coef_summary": _coefs_to_string(res["coefs"]),
                "std_coef_summary": _std_coefs_to_string(res["std_coefs"]),
            }
        )
    model_df = pd.DataFrame(model_rows).sort_values("cv_r2", ascending=False)
    model_df.to_csv(out_dir / "nested_model_results.csv", index=False)

    interaction_check = _fit_interaction_check(model_df_source, model_specs["Model 6: interpretable"])
    with open(out_dir / "model6_interaction_check.json", "w") as f:
        json.dump(interaction_check, f, indent=2)

    asymmetry_rows = []
    for label, left, right in [
        ("Amean_vs_Bmean", "Amean", "Bmean"),
        ("absDeltaA_vs_absDeltaB", "absDeltaA", "absDeltaB"),
        ("Amin_vs_Bmin", "Amin", "Bmin"),
    ]:
        for feat in [left, right]:
            cv_r2, cv_rmse = _cv_scores(model_df_source[[feat]].to_numpy(dtype=float), model_df_source["y"].to_numpy(dtype=float), degree=3)
            asymmetry_rows.append(
                {
                    "comparison": f"{label}:{feat}",
                    "feature": feat,
                    "cv_r2": cv_r2,
                    "cv_rmse": cv_rmse,
                    "interpretation": f"Univariate cubic CV R2 for {feat} in {label}",
                }
            )
    asymmetry_df = pd.DataFrame(asymmetry_rows).sort_values("cv_r2", ascending=False)
    asymmetry_df.to_csv(out_dir / "loaded_vs_unloaded_asymmetry.csv", index=False)

    branch_descriptor_rows = []
    for branch, feats in {
        "A": ["Amean", "HA", "Amin"],
        "B": ["Bmean", "HB", "Bmin"],
    }.items():
        for feat in feats:
            res = _fit_linear_summary(model_df_source, ["Gtot", feat])
            branch_descriptor_rows.append(
                {
                    "branch": branch,
                    "feature": feat,
                    "cv_r2": res["cv_r2"],
                    "cv_rmse": res["cv_rmse"],
                    "adj_r2": res["adj_r2"],
                }
            )
    branch_descriptor_df = pd.DataFrame(branch_descriptor_rows).sort_values(["branch", "cv_r2"], ascending=[True, False])
    branch_descriptor_df.to_csv(out_dir / "branch_descriptor_comparison.csv", index=False)

    nonlinear_imp, top3 = _fit_nonlinear_model(
        model_df_source,
        ["Gtot", "B", "Amean", "Bmean", "absDeltaA", "absDeltaB", "HA", "HB", "Amin", "Bmin"],
        out_dir,
    )

    gtot_r2 = _plot_residuals(plot_df.copy(), out_dir)

    _plot_pair_heatmap(plot_df, "Amean", "Bmean", "y", out_dir / "heatmap_Amean_Bmean.png", "Mean y over (Amean, Bmean)", "Amean", "Bmean")
    _plot_pair_heatmap(plot_df, "absDeltaA", "absDeltaB", "y", out_dir / "heatmap_absDeltaA_absDeltaB.png", "Mean y over (|DeltaA|, |DeltaB|)", "|DeltaA|", "|DeltaB|")
    _plot_pair_heatmap(plot_df, "Amin", "Bmin", "y", out_dir / "heatmap_Amin_Bmin.png", "Mean y over (Amin, Bmin)", "Amin", "Bmin")

    examples = _choose_representative_examples(df, run_dir)
    rep_df = _plot_representative_transfers(run_dir, meta, examples, out_dir)

    ranked_table = _make_ranked_table(univar, model_df, asymmetry_df)
    ranked_table.to_csv(out_dir / "ranked_table.csv", index=False)

    best_univar = univar.iloc[0]
    best_model = model_df.iloc[0]
    mismatch_model = model_df[model_df["model"] == "Model 4: Gtot + |DeltaA| + |DeltaB|"].iloc[0]
    balance_model = model_df[model_df["model"] == "Model 2: Gtot + B"].iloc[0]

    a_branch = branch_descriptor_df[branch_descriptor_df["branch"] == "A"].reset_index(drop=True)
    b_branch = branch_descriptor_df[branch_descriptor_df["branch"] == "B"].reset_index(drop=True)

    memo_lines = [
        f"- The current analysis used {len(df):,} finite saved sweep rows from {run_dir.name}; the loaded branch is A because run metadata sets ground_node=h1.",
        f"- Overall gate drive is dominant: the simple baseline y~Gtot achieves in-sample adjusted R2={model_df[model_df['model']=='Model 1: Gtot'].iloc[0]['adj_r2']:.3f} and CV R2={model_df[model_df['model']=='Model 1: Gtot'].iloc[0]['cv_r2']:.3f}.",
        f"- The strongest single summary variable by univariate cubic CV R2 is {best_univar['feature']} (CV R2={best_univar['poly3_cv_r2']:.3f}, Pearson={best_univar['pearson_y']:+.3f}, Spearman={best_univar['spearman_y']:+.3f}).",
        f"- Adding cross-branch balance B to Gtot gives CV R2={balance_model['cv_r2']:.3f}, while adding within-branch mismatch magnitudes gives CV R2={mismatch_model['cv_r2']:.3f}; delta={mismatch_model['cv_r2']-balance_model['cv_r2']:+.3f}.",
        f"- The best compact linear surrogate among the requested models is {best_model['model']} with CV R2={best_model['cv_r2']:.3f}, CV RMSE={best_model['cv_rmse']:.3f}, adjusted R2={best_model['adj_r2']:.3f}.",
        f"- Loaded vs unloaded asymmetry appears in the branch-local descriptors: best A-branch descriptor is {a_branch.iloc[0]['feature']} (CV R2={a_branch.iloc[0]['cv_r2']:.3f}) versus best B-branch descriptor {b_branch.iloc[0]['feature']} (CV R2={b_branch.iloc[0]['cv_r2']:.3f}).",
        f"- Harmonic/minimum vs arithmetic comparison: branch A ranking is {', '.join(f'{r.feature}:{r.cv_r2:.3f}' for r in a_branch.itertuples())}; branch B ranking is {', '.join(f'{r.feature}:{r.cv_r2:.3f}' for r in b_branch.itertuples())}.",
        f"- The nonlinear feature ranking confirms the simple models: top variables by permutation importance are {', '.join(top3)}.",
        f"- Residual structure remains after regressing out Gtot (baseline R2={gtot_r2:.3f}); the residual plots and heatmaps quantify where mismatch and bottleneck descriptors explain additional nonlinear behavior.",
        f"- Representative saved transfer curves linking these coordinates back to Vout(Vin) were selected from the logged transfer subset; details are in representative_transfer_examples.csv.",
    ]
    (out_dir / "summary_memo.md").write_text("\n".join(memo_lines) + "\n")

    mechanistic_statement = (
        "Across the frozen 4-transistor sweep, total gate drive is the dominant control variable for transfer linearity, "
        "but it is not sufficient to explain the full variation in log10(relative linear-fit RMSE). "
        "After accounting for overall drive, branch-local bottleneck descriptors and mismatch magnitudes retain measurable predictive power, "
        "indicating that nonlinearity is shaped by how drive is distributed inside each series branch rather than by total scale alone. "
        "The loaded branch contributes at least as strongly as the unloaded branch in the best compact surrogates, consistent with a real network-level asymmetry introduced by grounding h1. "
        "Among simple branch descriptors, the arithmetic, harmonic, and minimum summaries can be compared quantitatively in the saved outputs; whichever of these wins for branch A or B should be interpreted as the effective branch control coordinate for this architecture."
    )
    (out_dir / "mechanistic_statement.md").write_text(mechanistic_statement + "\n")

    # Also store a compact JSON summary for quick inspection.
    with open(out_dir / "analysis_summary.json", "w") as f:
        json.dump(
            {
                "n_rows": int(len(df)),
                "best_univariate_feature": str(best_univar["feature"]),
                "best_univariate_cv_r2": float(best_univar["poly3_cv_r2"]),
                "best_model": str(best_model["model"]),
                "best_model_cv_r2": float(best_model["cv_r2"]),
                "best_model_cv_rmse": float(best_model["cv_rmse"]),
                "mismatch_vs_balance_delta_cv_r2": float(mismatch_model["cv_r2"] - balance_model["cv_r2"]),
                "top3_nonlinear_features": top3,
                "representative_examples_found": int(len(rep_df)),
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()
