#!/usr/bin/env python3
"""UCI Heart Disease external lineage extension for SCI-f04.

The experiment tests whether site assembly, missing-value policy, and label
definition choices alter model behavior on a fixed Cleveland evaluation set.
It writes raw metrics, predictions, provenance, summaries, and timing files.

Smoke mode is only an operational check and must not be cited as evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
SOURCE_BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease"
FILES = [
    "processed.cleveland.data",
    "processed.hungarian.data",
    "processed.switzerland.data",
    "processed.va.data",
    "heart-disease.names",
]

COLS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]
NUM_COLS = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CAT_COLS = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
MODELS = ["logreg", "rf", "hgb"]
BASELINE = "cleveland_impute_any"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_sources(data_dir: Path) -> dict:
    data_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for name in FILES:
        url = f"{SOURCE_BASE}/{name}"
        path = data_dir / name
        if not path.exists():
            urlretrieve(url, path)
        records.append(
            {
                "file": name,
                "url": url,
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    return {"source_base": SOURCE_BASE, "files": records}


def load_site(data_dir: Path, name: str, site: str) -> pd.DataFrame:
    path = data_dir / name
    df = pd.read_csv(path, names=COLS, na_values="?", dtype=str)
    df["site"] = site
    for col in NUM_COLS + ["num"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in CAT_COLS:
        df[col] = df[col].astype(object)
    df = df.dropna(subset=["num"]).copy()
    df["target_any"] = (df["num"] > 0).astype(int)
    df["target_severe"] = (df["num"] >= 2).astype(int)
    return df


def load_all(data_dir: Path) -> pd.DataFrame:
    parts = [
        load_site(data_dir, "processed.cleveland.data", "cleveland"),
        load_site(data_dir, "processed.hungarian.data", "hungarian"),
        load_site(data_dir, "processed.switzerland.data", "switzerland"),
        load_site(data_dir, "processed.va.data", "va"),
    ]
    return pd.concat(parts, ignore_index=True)


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    out = 0.0
    for b in range(n_bins):
        m = idx == b
        if not np.any(m):
            continue
        out += float(m.mean() * abs(y[m].mean() - p[m].mean()))
    return out


def subgroup_metrics(y: np.ndarray, yhat: np.ndarray, sex: pd.Series) -> dict:
    male = sex.astype(str).to_numpy() == "1.0"
    female = sex.astype(str).to_numpy() == "0.0"
    out = {
        "sex_dp_diff_male_minus_female": np.nan,
        "sex_tpr_gap_male_minus_female": np.nan,
    }
    if male.sum() and female.sum():
        out["sex_dp_diff_male_minus_female"] = float(yhat[male].mean() - yhat[female].mean())
    pos_male = male & (y == 1)
    pos_female = female & (y == 1)
    if pos_male.sum() and pos_female.sum():
        out["sex_tpr_gap_male_minus_female"] = float(yhat[pos_male].mean() - yhat[pos_female].mean())
    return out


def make_model(name: str, seed: int, n_trees: int) -> Pipeline:
    if name == "logreg":
        pre = ColumnTransformer(
            [
                ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUM_COLS),
                (
                    "cat",
                    Pipeline(
                        [
                            ("imp", SimpleImputer(strategy="most_frequent")),
                            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
                        ]
                    ),
                    CAT_COLS,
                ),
            ]
        )
        clf = LogisticRegression(max_iter=2000, solver="lbfgs", C=1.0, random_state=seed)
    elif name == "rf":
        pre = ColumnTransformer(
            [
                ("num", SimpleImputer(strategy="median"), NUM_COLS),
                (
                    "cat",
                    Pipeline(
                        [
                            ("imp", SimpleImputer(strategy="most_frequent")),
                            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
                        ]
                    ),
                    CAT_COLS,
                ),
            ]
        )
        clf = RandomForestClassifier(
            n_estimators=n_trees,
            min_samples_leaf=3,
            random_state=seed,
            n_jobs=2,
        )
    elif name == "hgb":
        pre = ColumnTransformer(
            [
                ("num", SimpleImputer(strategy="median"), NUM_COLS),
                (
                    "cat",
                    Pipeline(
                        [
                            ("imp", SimpleImputer(strategy="most_frequent")),
                            ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                        ]
                    ),
                    CAT_COLS,
                ),
            ]
        )
        clf = HistGradientBoostingClassifier(random_state=seed)
    else:
        raise ValueError(name)
    return Pipeline([("pre", pre), ("clf", clf)])


def variant_frame(df: pd.DataFrame, variant: str, eval_idx: set[int]) -> tuple[pd.DataFrame, str, str, str]:
    if variant.startswith("cleveland"):
        sub = df[df["site"] == "cleveland"].copy()
        sub = sub[~sub.index.isin(eval_idx)].copy()
        site_policy = "cleveland_only"
    elif variant.startswith("multi_site"):
        sub = df.copy()
        sub = sub[~((sub["site"] == "cleveland") & sub.index.isin(eval_idx))].copy()
        site_policy = "cleveland_hungarian_switzerland_va"
    else:
        raise ValueError(variant)

    missing_policy = "drop_any_missing" if "_drop_" in variant else "median_mode_impute"
    if missing_policy == "drop_any_missing":
        sub = sub.dropna(subset=NUM_COLS + CAT_COLS).copy()

    target_col = "target_severe" if variant.endswith("_severe") else "target_any"
    label_policy = "num_ge_2" if target_col == "target_severe" else "num_gt_0"
    return sub, site_policy, missing_policy, label_policy


def eval_metrics(pipe: Pipeline, frame: pd.DataFrame, target_col: str) -> dict:
    y = frame[target_col].to_numpy()
    p = pipe.predict_proba(frame[NUM_COLS + CAT_COLS])[:, 1]
    yhat = (p >= 0.5).astype(int)
    out = {
        "acc": float(accuracy_score(y, yhat)),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
        "ece": ece(y, p),
        "pred_pos_rate": float(yhat.mean()),
    }
    out.update(subgroup_metrics(y, yhat, frame["sex"]))
    return out, yhat


def choose_common_eval(cleveland: pd.DataFrame, n_eval: int, seed: int) -> list[int]:
    complete = cleveland.dropna(subset=NUM_COLS + CAT_COLS).copy()
    n_eval = min(n_eval, len(complete) - 20)
    y = complete["target_any"].to_numpy()
    rng = np.random.RandomState(seed)
    pos = complete.index[y == 1].to_numpy()
    neg = complete.index[y == 0].to_numpy()
    n_pos = int(round(n_eval * y.mean()))
    chosen = np.concatenate(
        [
            rng.choice(pos, n_pos, replace=False),
            rng.choice(neg, n_eval - n_pos, replace=False),
        ]
    )
    return sorted(int(x) for x in chosen)


def run(args: argparse.Namespace) -> None:
    t_start = time.time()
    result_dir = Path(args.result_dir)
    data_dir = Path(args.data_dir)
    raw_dir = result_dir / "raw"
    pred_dir = result_dir / "preds"
    raw_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    provenance = download_sources(data_dir)
    df = load_all(data_dir)
    seeds = [0] if args.smoke else list(range(args.seeds))
    variants = [
        "cleveland_impute_any",
        "cleveland_drop_any",
        "multi_site_impute_any",
        "multi_site_drop_any",
        "cleveland_impute_severe",
    ]
    n_trees = args.smoke_trees if args.smoke else args.rf_trees
    eval_idx = choose_common_eval(df[df["site"] == "cleveland"], args.eval_n_smoke if args.smoke else args.eval_n, args.eval_seed)
    common_eval = df.loc[eval_idx].copy()

    provenance["loaded_rows"] = {
        site: int((df["site"] == site).sum()) for site in ["cleveland", "hungarian", "switzerland", "va"]
    }
    provenance["missing_feature_cells"] = {
        site: int(df.loc[df["site"] == site, NUM_COLS + CAT_COLS].isna().sum().sum())
        for site in ["cleveland", "hungarian", "switzerland", "va"]
    }
    provenance["common_eval"] = {
        "site": "cleveland",
        "n": len(eval_idx),
        "seed": args.eval_seed,
        "complete_case": True,
        "target_any_positive_rate": float(common_eval["target_any"].mean()),
        "target_severe_positive_rate": float(common_eval["target_severe"].mean()),
    }
    provenance["run_mode"] = "smoke" if args.smoke else "full"
    (result_dir / "heart_source_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    rows = []
    predictions: dict[tuple[str, str, int], np.ndarray] = {}
    for variant in variants:
        train_frame, site_policy, missing_policy, label_policy = variant_frame(df, variant, set(eval_idx))
        target_col = "target_severe" if label_policy == "num_ge_2" else "target_any"
        for model_name in MODELS:
            for seed in seeds:
                tag = f"{variant}_{model_name}_s{seed}"
                if train_frame[target_col].nunique() < 2:
                    raise RuntimeError(f"{tag} has only one target class")
                strat = train_frame[target_col] if train_frame[target_col].nunique() == 2 else None
                train_part, own_test = train_test_split(
                    train_frame,
                    test_size=0.25,
                    random_state=seed,
                    stratify=strat,
                )
                pipe = make_model(model_name, seed, n_trees)
                fit_t0 = time.time()
                pipe.fit(train_part[NUM_COLS + CAT_COLS], train_part[target_col].to_numpy())
                fit_seconds = time.time() - fit_t0
                own_metrics, own_pred = eval_metrics(pipe, own_test, target_col)
                common_metrics, common_pred = eval_metrics(pipe, common_eval, target_col)
                predictions[(variant, model_name, seed)] = common_pred
                np.save(pred_dir / f"{tag}.npy", common_pred.astype(np.uint8))
                row = {
                    "variant": variant,
                    "model": model_name,
                    "seed": seed,
                    "site_policy": site_policy,
                    "missing_policy": missing_policy,
                    "label_policy": label_policy,
                    "target_col": target_col,
                    "n_train": int(len(train_part)),
                    "n_own_test": int(len(own_test)),
                    "n_common_eval": int(len(common_eval)),
                    "train_pos_rate": float(train_part[target_col].mean()),
                    "common_label_pos_rate": float(common_eval[target_col].mean()),
                    "fit_seconds": round(fit_seconds, 4),
                    "own_acc": own_metrics["acc"],
                    "own_auc": own_metrics["auc"],
                    "own_ece": own_metrics["ece"],
                    "own_pred_pos_rate": own_metrics["pred_pos_rate"],
                    "common_acc": common_metrics["acc"],
                    "common_auc": common_metrics["auc"],
                    "common_ece": common_metrics["ece"],
                    "common_pred_pos_rate": common_metrics["pred_pos_rate"],
                    "common_sex_dp_diff": common_metrics["sex_dp_diff_male_minus_female"],
                    "common_sex_tpr_gap": common_metrics["sex_tpr_gap_male_minus_female"],
                    "smoke": bool(args.smoke),
                }
                rows.append(row)
                (raw_dir / f"{tag}.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
                print(f"completed {tag}: common_acc={row['common_acc']:.3f} common_auc={row['common_auc']:.3f} fit={fit_seconds:.2f}s")

    unit = pd.DataFrame(rows)
    unit.to_csv(result_dir / "heart_unit_metrics.csv", index=False)

    summary = (
        unit.groupby(["variant", "model"])
        [
            [
                "n_train",
                "train_pos_rate",
                "common_label_pos_rate",
                "common_acc",
                "common_auc",
                "common_ece",
                "common_pred_pos_rate",
                "common_sex_dp_diff",
                "common_sex_tpr_gap",
                "fit_seconds",
            ]
        ]
        .agg(["mean", "std"])
    )
    summary.columns = ["_".join(col) for col in summary.columns]
    summary.reset_index().to_csv(result_dir / "heart_variant_model_summary.csv", index=False)

    delta_rows = []
    for variant in variants:
        if variant == BASELINE:
            continue
        for model_name in MODELS:
            for seed in seeds:
                base = unit[(unit.variant == BASELINE) & (unit.model == model_name) & (unit.seed == seed)].iloc[0]
                cur = unit[(unit.variant == variant) & (unit.model == model_name) & (unit.seed == seed)].iloc[0]
                pred = predictions[(variant, model_name, seed)]
                base_pred = predictions[(BASELINE, model_name, seed)]
                same_target = cur["label_policy"] == base["label_policy"]
                rec = {
                    "variant": variant,
                    "model": model_name,
                    "seed": seed,
                    "same_target_as_baseline": bool(same_target),
                    "churn_vs_baseline": float((pred != base_pred).mean()),
                    "d_common_acc": float(cur["common_acc"] - base["common_acc"]) if same_target else np.nan,
                    "d_common_auc": float(cur["common_auc"] - base["common_auc"]) if same_target else np.nan,
                    "d_common_ece": float(cur["common_ece"] - base["common_ece"]) if same_target else np.nan,
                    "d_common_pred_pos_rate": float(cur["common_pred_pos_rate"] - base["common_pred_pos_rate"]),
                    "d_common_sex_tpr_gap": float(cur["common_sex_tpr_gap"] - base["common_sex_tpr_gap"]) if same_target else np.nan,
                }
                delta_rows.append(rec)
    deltas = pd.DataFrame(delta_rows)
    deltas.to_csv(result_dir / "heart_deltas_vs_baseline.csv", index=False)

    effect_rows = []
    for variant, group in deltas.groupby("variant"):
        for outcome in ["churn_vs_baseline", "d_common_acc", "d_common_auc", "d_common_ece", "d_common_sex_tpr_gap"]:
            x = group[outcome].dropna().to_numpy()
            if len(x) == 0:
                continue
            rec = {
                "variant": variant,
                "outcome": outcome,
                "n": int(len(x)),
                "mean": float(np.mean(x)),
                "sd": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
                "min": float(np.min(x)),
                "max": float(np.max(x)),
                "wilcoxon_p": np.nan,
            }
            if outcome != "churn_vs_baseline" and len(x) > 1 and not np.allclose(x, 0):
                try:
                    rec["wilcoxon_p"] = float(wilcoxon(x).pvalue)
                except ValueError:
                    rec["wilcoxon_p"] = np.nan
            effect_rows.append(rec)
    pd.DataFrame(effect_rows).to_csv(result_dir / "heart_effect_sizes.csv", index=False)

    churn = pd.DataFrame(index=variants, columns=variants, dtype=float)
    for v1 in variants:
        for v2 in variants:
            vals = []
            for model_name in MODELS:
                for seed in seeds:
                    vals.append(float((predictions[(v1, model_name, seed)] != predictions[(v2, model_name, seed)]).mean()))
            churn.loc[v1, v2] = float(np.mean(vals))
    churn.to_csv(result_dir / "heart_churn_matrix.csv")

    run_summary = {
        "run_mode": "smoke" if args.smoke else "full",
        "variants": variants,
        "models": MODELS,
        "seeds": seeds,
        "n_units": int(len(unit)),
        "baseline": BASELINE,
        "elapsed_seconds": round(time.time() - t_start, 3),
        "main_effects": pd.DataFrame(effect_rows).to_dict(orient="records"),
    }
    (result_dir / "heart_results_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    print(json.dumps(run_summary, indent=2))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--result-dir", default=str(ROOT / "results" / "heart_lineage_extension"))
    p.add_argument("--data-dir", default=str(ROOT / "data" / "heart_disease_uci"))
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--eval-n", type=int, default=90)
    p.add_argument("--eval-n-smoke", type=int, default=30)
    p.add_argument("--eval-seed", type=int, default=20260615)
    p.add_argument("--rf-trees", type=int, default=300)
    p.add_argument("--smoke-trees", type=int, default=30)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
