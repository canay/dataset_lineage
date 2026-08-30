#!/usr/bin/env python3
"""Paper 4: resumable experiment runner.

Work unit = (case, variant, model, seed). Each unit trains one model on one
dataset variant and evaluates on (a) the variant's own held-out test split and
(b) the frozen common evaluation set. Results -> results/raw/<unit>.json,
common-eval predictions -> results/preds/<unit>.npy (uint8).

Usage: run_units.py <shard> <n_shards>
Internal budget: starts no new unit after MAX_START_SEC; exits cleanly.
"""
import json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = str(Path(__file__).resolve().parents[1])
MAX_START_SEC = 1000000
SEEDS = [0, 1, 2, 3, 4]
ADULT_VARIANTS = ["V0", "V1_workforce", "V3_drop", "V3_impute", "V4_dedup", "V5_uniform", "V5_hist"]
ACS_VARIANTS = ["T25", "T50", "T85"]
GERMAN_VARIANTS = ["statlog", "corrected"]
MODELS = ["logreg", "rf", "hgb"]

NUM_COLS = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CAT_COLS = ["workclass", "education", "marital-status", "occupation",
            "relationship", "race", "sex", "native-country"]
ACS_NUM = ["AGEP", "WKHP"]
ACS_CAT = ["COW", "SCHL", "MAR", "OCCP", "POBP", "RELP", "SEX", "RAC1P"]


def make_model(name, seed, num_cols, cat_cols):
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    if name == "logreg":
        pre = ColumnTransformer([("num", StandardScaler(), num_cols), ("cat", ohe, cat_cols)])
        clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=seed)
    elif name == "rf":
        pre = ColumnTransformer([("num", "passthrough", num_cols), ("cat", ohe, cat_cols)])
        clf = RandomForestClassifier(n_estimators=500, min_samples_leaf=5,
                                     random_state=seed, n_jobs=-1)
    elif name == "hgb":
        from sklearn.preprocessing import OrdinalEncoder
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        pre = ColumnTransformer([("num", "passthrough", num_cols), ("cat", oe, cat_cols)])
        clf = HistGradientBoostingClassifier(random_state=seed)
    return Pipeline([("pre", pre), ("clf", clf)])


def ece(y, p, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def subgroup_metrics(y, yhat, g_a, g_b, prefix):
    """g_a, g_b boolean masks for the two groups; gaps reported as a-minus-b."""
    out = {}
    out[f"{prefix}_dp_diff"] = float(yhat[g_a].mean() - yhat[g_b].mean())
    pa, pb = (y == 1) & g_a, (y == 1) & g_b
    tpr_a = float(yhat[pa].mean()) if pa.sum() else np.nan
    tpr_b = float(yhat[pb].mean()) if pb.sum() else np.nan
    out[f"{prefix}_tpr_gap"] = tpr_a - tpr_b
    return out


def perm_importance(pipe, Xe, ye, cols, seed, n_rep=3):
    base = roc_auc_score(ye, pipe.predict_proba(Xe)[:, 1])
    rng = np.random.RandomState(seed)
    imp = {}
    for c in cols:
        drops = []
        for _ in range(n_rep):
            Xp = Xe.copy()
            Xp[c] = rng.permutation(Xp[c].to_numpy())
            drops.append(base - roc_auc_score(ye, pipe.predict_proba(Xp)[:, 1]))
        imp[c] = float(np.mean(drops))
    return imp, float(base)


def build_adult_variant(d, variant, seed):
    """Return train_df, owntest_df (variant pipeline applied), missing policy tag."""
    pool = d[~d["is_eval"]].copy()
    policy = "keep_category"
    if variant == "V1_workforce":
        keep = ~(pool["workclass"].isna() | pool["workclass"].isin(["Never-worked", "Without-pay"]))
        keep &= pool["hours-per-week"] >= 1
        pool = pool[keep]
    elif variant == "V3_drop":
        pool = pool.dropna(subset=NUM_COLS + CAT_COLS)
        policy = "drop"
    elif variant == "V3_impute":
        policy = "impute_mode"
    elif variant == "V4_dedup":
        pool = pool.drop_duplicates(subset=NUM_COLS + CAT_COLS + ["y"], keep="first")
    if variant == "V5_hist":
        tr = pool[~pool["hist_test"]]
        te = pool[pool["hist_test"]]
    elif variant == "V5_uniform":
        tr, te = train_test_split(pool, test_size=0.2, random_state=seed, stratify=None)
    else:
        tr, te = train_test_split(pool, test_size=0.2, random_state=seed, stratify=pool["y"])
    return tr, te, policy


def apply_policy(df, policy, modes=None):
    df = df.copy()
    if policy == "keep_category":
        for c in CAT_COLS:
            df[c] = df[c].fillna("Missing")
    elif policy == "impute_mode":
        for c in CAT_COLS:
            df[c] = df[c].fillna(modes[c])
    return df


def run_adult_unit(variant, model, seed):
    d = pd.read_pickle(f"{ROOT}/data/adult_prepared.pkl")
    tr, te, policy = build_adult_variant(d, variant, seed)
    modes = {c: tr[c].mode(dropna=True).iloc[0] for c in CAT_COLS}
    tr_p = apply_policy(tr, policy, modes)
    te_p = apply_policy(te, policy, modes) if policy != "drop" else te
    ev = d[d["is_eval"]]
    ev_p = apply_policy(ev, policy, modes)  # eval rows are complete-case; no-op fills
    pipe = make_model(model, seed, NUM_COLS, CAT_COLS)
    Xtr, ytr = tr_p[NUM_COLS + CAT_COLS], tr_p["y"].to_numpy()
    t0 = time.time(); pipe.fit(Xtr, ytr); fit_s = time.time() - t0
    out = {"case": "adult", "variant": variant, "model": model, "seed": seed,
           "n_train": len(tr_p), "n_owntest": len(te_p), "pos_rate_train": float(ytr.mean()),
           "fit_seconds": round(fit_s, 2)}
    # own test
    yo = te_p["y"].to_numpy()
    po = pipe.predict_proba(te_p[NUM_COLS + CAT_COLS])[:, 1]
    yho = (po >= 0.5).astype(int)
    out.update(own_acc=float(accuracy_score(yo, yho)), own_auc=float(roc_auc_score(yo, po)),
               own_ece=ece(yo, po))
    # common eval
    ye = ev_p["y"].to_numpy()
    Xe = ev_p[NUM_COLS + CAT_COLS]
    pe = pipe.predict_proba(Xe)[:, 1]
    yhe = (pe >= 0.5).astype(int)
    out.update(common_acc=float(accuracy_score(ye, yhe)), common_auc=float(roc_auc_score(ye, pe)),
               common_ece=ece(ye, pe), common_posrate_pred=float(yhe.mean()))
    male = (ev_p["sex"] == "Male").to_numpy(); female = (ev_p["sex"] == "Female").to_numpy()
    out.update(subgroup_metrics(ye, yhe, male, female, "sex"))
    w = (ev_p["race"] == "White").to_numpy(); b = (ev_p["race"] == "Black").to_numpy()
    out.update(subgroup_metrics(ye, yhe, w, b, "race"))
    imp, base = perm_importance(pipe, Xe, ye, NUM_COLS + CAT_COLS, seed)
    out["perm_importance"] = imp
    return out, yhe


def run_acs_unit(variant, model, seed):
    d = pd.read_pickle(f"{ROOT}/data/acs_prepared.pkl")
    ycol = f"y_{variant}"
    pool = d[~d["is_eval"]]
    tr, te = train_test_split(pool, test_size=0.2, random_state=seed, stratify=pool[ycol])
    for df in (tr, te):
        pass
    cat_as_str = lambda df: df.assign(**{c: df[c].astype(int).astype(str) for c in ACS_CAT})
    tr, te, ev = cat_as_str(tr), cat_as_str(te), cat_as_str(d[d["is_eval"]])
    pipe = make_model(model, seed, ACS_NUM, ACS_CAT)
    ytr = tr[ycol].to_numpy()
    t0 = time.time(); pipe.fit(tr[ACS_NUM + ACS_CAT], ytr); fit_s = time.time() - t0
    out = {"case": "acs", "variant": variant, "model": model, "seed": seed,
           "n_train": len(tr), "n_owntest": len(te), "pos_rate_train": float(ytr.mean()),
           "fit_seconds": round(fit_s, 2)}
    yo = te[ycol].to_numpy()
    po = pipe.predict_proba(te[ACS_NUM + ACS_CAT])[:, 1]
    yho = (po >= 0.5).astype(int)
    out.update(own_acc=float(accuracy_score(yo, yho)), own_auc=float(roc_auc_score(yo, po)),
               own_ece=ece(yo, po))
    ye = ev[ycol].to_numpy()
    Xe = ev[ACS_NUM + ACS_CAT]
    pe = pipe.predict_proba(Xe)[:, 1]
    yhe = (pe >= 0.5).astype(int)
    out.update(common_acc=float(accuracy_score(ye, yhe)), common_auc=float(roc_auc_score(ye, pe)),
               common_ece=ece(ye, pe), common_posrate_pred=float(yhe.mean()))
    male = (ev["SEX"] == "1").to_numpy(); female = (ev["SEX"] == "2").to_numpy()
    out.update(subgroup_metrics(ye, yhe, male, female, "sex"))
    w = (ev["RAC1P"] == "1").to_numpy(); b = (ev["RAC1P"] == "2").to_numpy()
    out.update(subgroup_metrics(ye, yhe, w, b, "race"))
    imp, base = perm_importance(pipe, Xe, ye, ACS_NUM + ACS_CAT, seed)
    out["perm_importance"] = imp
    return out, yhe


GER_NUM = ["laufzeit", "hoehe", "rate", "wohnzeit", "alter", "bishkred", "pers"]
GER_CAT = ["laufkont", "moral", "verw", "sparkont", "beszeit", "famges", "buerge",
           "verm", "weitkred", "wohn", "bishkred2", "beruf", "telef", "gastarb"]


def load_german(variant):
    sgc_cols = ["laufkont", "laufzeit", "moral", "verw", "hoehe", "sparkont", "beszeit",
                "rate", "famges", "buerge", "wohnzeit", "verm", "alter", "weitkred",
                "wohn", "bishkred", "beruf", "pers", "telef", "gastarb", "kredit"]
    if variant == "statlog":
        d = pd.read_csv(f"{ROOT}/data/uci/german.data", sep=" ", header=None)
        d.columns = sgc_cols
        d["y"] = (d["kredit"] == 1).astype(int)
    else:
        d = pd.read_csv(f"{ROOT}/data/uci/sgc/SouthGermanCredit.asc", sep=" ")
        d["y"] = (d["kredit"] == 1).astype(int)
    num = ["laufzeit", "hoehe", "rate", "wohnzeit", "alter", "pers"]
    cat = ["laufkont", "moral", "verw", "sparkont", "beszeit", "famges", "buerge",
           "verm", "weitkred", "wohn", "bishkred", "beruf", "telef", "gastarb"]
    for c in cat:
        d[c] = d[c].astype(str)
    return d, num, cat


def run_german_unit(variant, model, seed):
    d, num, cat = load_german(variant)
    tr, te = train_test_split(d, test_size=0.3, random_state=seed, stratify=d["y"])
    pipe = make_model(model, seed, num, cat)
    ytr = tr["y"].to_numpy()
    t0 = time.time(); pipe.fit(tr[num + cat], ytr); fit_s = time.time() - t0
    yo = te["y"].to_numpy()
    po = pipe.predict_proba(te[num + cat])[:, 1]
    yho = (po >= 0.5).astype(int)
    out = {"case": "german", "variant": variant, "model": model, "seed": seed,
           "n_train": len(tr), "n_owntest": len(te), "pos_rate_train": float(ytr.mean()),
           "fit_seconds": round(fit_s, 2),
           "own_acc": float(accuracy_score(yo, yho)), "own_auc": float(roc_auc_score(yo, po)),
           "own_ece": ece(yo, po)}
    imp, base = perm_importance(pipe, te[num + cat], yo, num + cat, seed)
    out["perm_importance"] = imp
    return out, yho


def all_units():
    units = []
    for v in ADULT_VARIANTS:
        for m in MODELS:
            for s in SEEDS:
                units.append(("adult", v, m, s))
    for v in ACS_VARIANTS:
        for m in MODELS:
            for s in SEEDS:
                units.append(("acs", v, m, s))
    for v in GERMAN_VARIANTS:
        for m in MODELS:
            for s in SEEDS:
                units.append(("german", v, m, s))
    return units


def main():
    shard, n_shards = int(sys.argv[1]), int(sys.argv[2])
    os.makedirs(f"{ROOT}/results/raw", exist_ok=True)
    os.makedirs(f"{ROOT}/results/preds", exist_ok=True)
    t0 = time.time()
    units = [u for i, u in enumerate(all_units()) if i % n_shards == shard]
    done = ran = 0
    for case, v, m, s in units:
        tag = f"{case}_{v}_{m}_s{s}"
        fj = f"{ROOT}/results/raw/{tag}.json"
        if os.path.exists(fj):
            done += 1; continue
        if time.time() - t0 > MAX_START_SEC:
            print(f"shard {shard}: budget reached, ran {ran}, done {done}/{len(units)}")
            return
        runner = {"adult": run_adult_unit, "acs": run_acs_unit, "german": run_german_unit}[case]
        out, yhe = runner(v, m, s)
        np.save(f"{ROOT}/results/preds/{tag}.npy", yhe.astype(np.uint8))
        json.dump(out, open(fj, "w"))
        ran += 1
        print(f"shard {shard}: {tag} done ({out['fit_seconds']}s)")
    print(f"shard {shard}: ALL DONE ({done + ran}/{len(units)})")


if __name__ == "__main__":
    main()

