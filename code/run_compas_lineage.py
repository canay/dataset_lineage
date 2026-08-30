#!/usr/bin/env python3
"""Second primary lineage case: COMPAS (ProPublica recidivism).

The ProPublica preprocessing of the COMPAS file is itself a documented and contested
lineage decision. This script re-enacts selected COMPAS lineage decisions as controlled
variants under fixed models and a frozen common evaluation set drawn from the canonical
(ProPublica) rows, so that identical defendants are scored under each variant. It mirrors
the Adult protocol (common-set accuracy/AUC/ECE, prediction churn, subgroup gaps,
permutation importance) and adds the same rigor layer (paired Wilcoxon with SciPy's
automatic method selection + Benjamini-Hochberg FDR, bootstrap subgroup CIs,
seed-baseline churn).

Variants:
  C0_propublica : screening-window filter (|days_b_screening_arrest|<=30), is_recid!=-1,
                  c_charge_degree!='O', score_text!='N/A'; label two_year_recid; stratified split (BASELINE)
  C1_nowindow   : drop the screening-window filter (keep all such rows); label two_year_recid; stratified
  C2_isrecid    : C0 rows; label is_recid (target-definition change; reported as target sensitivity)
  C3_uniform    : C0 rows; label two_year_recid; non-stratified split
"""
import glob, json, os, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
SRC = f"{ROOT}/data/compas/compas-scores-two-years.csv"
NUM = ["age", "priors_count", "juv_fel_count", "juv_misd_count", "juv_other_count"]
CAT = ["sex", "c_charge_degree"]
COLS = NUM + CAT
MODELS = ["logreg", "rf", "hgb"]
SEEDS = [0, 1, 2, 3, 4]
VARIANTS = ["C0_propublica", "C1_nowindow", "C2_isrecid", "C3_uniform"]
BASE = "C0_propublica"
COMPATIBLE = ["C1_nowindow", "C3_uniform"]  # same label as C0; C2 changes the target
RNG_EVAL = 20260624
N_EVAL = 1500
RNG = np.random.RandomState(20260624)


def make_model(name, seed):
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    if name == "logreg":
        pre = ColumnTransformer([("num", StandardScaler(), NUM), ("cat", ohe, CAT)])
        clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=seed)
    elif name == "rf":
        pre = ColumnTransformer([("num", "passthrough", NUM), ("cat", ohe, CAT)])
        clf = RandomForestClassifier(n_estimators=500, min_samples_leaf=5, random_state=seed, n_jobs=-1)
    else:
        oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        pre = ColumnTransformer([("num", "passthrough", NUM), ("cat", oe, CAT)])
        clf = HistGradientBoostingClassifier(random_state=seed)
    return Pipeline([("pre", pre), ("clf", clf)])


def ece(y, p, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def subgroup(y, yhat, ga, gb, prefix):
    out = {}
    out[f"{prefix}_dp_diff"] = float(yhat[ga].mean() - yhat[gb].mean())
    pa, pb = (y == 1) & ga, (y == 1) & gb
    out[f"{prefix}_tpr_gap"] = (float(yhat[pa].mean()) if pa.sum() else np.nan) - \
                              (float(yhat[pb].mean()) if pb.sum() else np.nan)
    return out


def perm_importance(pipe, Xe, ye, seed, n_rep=3):
    base = roc_auc_score(ye, pipe.predict_proba(Xe)[:, 1])
    rng = np.random.RandomState(seed)
    imp = {}
    Xp = Xe.copy()
    for c in COLS:
        orig = Xe[c].to_numpy(); drops = []
        for _ in range(n_rep):
            Xp[c] = rng.permutation(orig)
            drops.append(base - roc_auc_score(ye, pipe.predict_proba(Xp)[:, 1]))
        Xp[c] = orig; imp[c] = float(np.mean(drops))
    return imp


def load_and_flag():
    d = pd.read_csv(SRC)
    win = (d.days_b_screening_arrest <= 30) & (d.days_b_screening_arrest >= -30)
    keep_other = (d.is_recid != -1) & (d.c_charge_degree != "O") & (d.score_text != "N/A")
    d["in_C0"] = win & keep_other
    d["in_C1"] = keep_other  # no window filter
    return d


def variant_pool(d, variant, eval_ids):
    if variant == "C1_nowindow":
        rows = d[d["in_C1"]].copy()
    else:
        rows = d[d["in_C0"]].copy()
    rows["y"] = (rows["is_recid"] == 1).astype(int) if variant == "C2_isrecid" else rows["two_year_recid"].astype(int)
    return rows[~rows["id"].isin(eval_ids)]


def main():
    os.makedirs(f"{ROOT}/results/raw", exist_ok=True)
    os.makedirs(f"{ROOT}/results/preds", exist_ok=True)
    d = load_and_flag()
    c0 = d[d["in_C0"]].copy()
    c0["y"] = c0["two_year_recid"].astype(int)
    # freeze common eval from C0 rows, stratified by label
    rng = np.random.RandomState(RNG_EVAL)
    y = c0["y"].to_numpy(); idx = c0.index.to_numpy()
    pos, neg = idx[y == 1], idx[y == 0]
    n_pos = int(round(N_EVAL * y.mean()))
    eval_idx = np.sort(np.concatenate([rng.choice(pos, n_pos, replace=False),
                                       rng.choice(neg, N_EVAL - n_pos, replace=False)]))
    eval_ids = set(d.loc[eval_idx, "id"])
    ev = d.loc[eval_idx].copy()
    ev_y2 = ev["two_year_recid"].astype(int).to_numpy()
    race_a = (ev["race"] == "African-American").to_numpy(); race_c = (ev["race"] == "Caucasian").to_numpy()
    sex_m = (ev["sex"] == "Male").to_numpy(); sex_f = (ev["sex"] == "Female").to_numpy()

    lineage = {"raw_rows": int(len(d)), "C0_rows": int(d["in_C0"].sum()),
               "C1_rows": int(d["in_C1"].sum()), "window_filtered_out": int((~((d.days_b_screening_arrest<=30)&(d.days_b_screening_arrest>=-30))).sum()),
               "two_year_recid_rate": float(c0["y"].mean()), "is_recid_rate": float((c0["is_recid"]==1).mean()),
               "common_eval_n": int(len(eval_idx)), "common_eval_posrate": float(ev_y2.mean()),
               "eval_african_american": int(race_a.sum()), "eval_caucasian": int(race_c.sum()),
               "eval_male": int(sex_m.sum()), "eval_female": int(sex_f.sum())}

    preds = {}
    rows_out = []
    for v in VARIANTS:
        pool = variant_pool(d, v, eval_ids)
        ycol_eval = (ev["is_recid"] == 1).astype(int).to_numpy() if v == "C2_isrecid" else ev_y2
        for m in MODELS:
            for s in SEEDS:
                strat = None if v == "C3_uniform" else pool["y"]
                tr, te = train_test_split(pool, test_size=0.2, random_state=s, stratify=strat)
                pipe = make_model(m, s)
                pipe.fit(tr[COLS], tr["y"].to_numpy())
                pe = pipe.predict_proba(ev[COLS])[:, 1]
                yhe = (pe >= 0.5).astype(int)
                preds[(v, m, s)] = yhe
                np.save(f"{ROOT}/results/preds/compas_{v}_{m}_s{s}.npy", yhe.astype(np.uint8))
                rec = {"case": "compas", "variant": v, "model": m, "seed": s,
                       "n_train": int(len(tr)), "pos_rate_train": float(tr["y"].mean()),
                       "common_acc": float(accuracy_score(ycol_eval, yhe)),
                       "common_auc": float(roc_auc_score(ycol_eval, pe)),
                       "common_ece": ece(ycol_eval, pe),
                       "common_posrate_pred": float(yhe.mean())}
                rec.update(subgroup(ycol_eval, yhe, sex_m, sex_f, "sex"))
                rec.update(subgroup(ycol_eval, yhe, race_a, race_c, "race"))
                rec["perm_importance"] = perm_importance(pipe, ev[COLS], ycol_eval, s)
                json.dump(rec, open(f"{ROOT}/results/raw/compas_{v}_{m}_s{s}.json", "w"))
                rows_out.append({k: rec[k] for k in rec if k != "perm_importance"})
        print(f"  done {v}", flush=True)
    df = pd.DataFrame(rows_out)
    df.to_csv(f"{ROOT}/results/compas_unit_metrics.csv", index=False)

    # deltas vs C0 (paired by model, seed)
    OUT = ["common_acc", "common_auc", "common_ece", "sex_dp_diff", "sex_tpr_gap", "race_dp_diff", "race_tpr_gap"]
    deltas = []
    for v in VARIANTS:
        if v == BASE:
            continue
        for m in MODELS:
            for s in SEEDS:
                a = df[(df.variant == v) & (df.model == m) & (df.seed == s)]
                b = df[(df.variant == BASE) & (df.model == m) & (df.seed == s)]
                row = {"variant": v, "model": m, "seed": s}
                for o in OUT:
                    row[f"d_{o}"] = float(a[o].iloc[0] - b[o].iloc[0])
                row["churn_vs_base"] = float((preds[(v, m, s)] != preds[(BASE, m, s)]).mean())
                deltas.append(row)
    dd = pd.DataFrame(deltas)
    dd.to_csv(f"{ROOT}/results/compas_deltas.csv", index=False)

    def wilc(x):
        x = np.asarray(x, float)
        return np.nan if np.allclose(x, 0) else float(wilcoxon(x, zero_method="wilcox", alternative="two-sided").pvalue)

    # effect sizes + raw p for compatible variants; collect for BH-FDR
    eff, praw = [], []
    for v in [x for x in VARIANTS if x != BASE]:
        dv = dd[dd.variant == v]
        for o in ["d_common_acc", "d_common_auc", "d_common_ece", "d_sex_dp_diff", "d_sex_tpr_gap", "d_race_tpr_gap"]:
            x = dv[o].to_numpy(); mean = float(np.mean(x)); sd = float(np.std(x, ddof=1))
            p = wilc(x)
            rec = {"variant": v, "outcome": o, "mean": mean, "sd": sd, "d_z": mean/sd if sd>0 else np.nan,
                   "p_raw": p, "churn_mean": float(dv["churn_vs_base"].mean())}
            eff.append(rec)
            if v in COMPATIBLE and not np.isnan(p):
                praw.append((len(eff)-1, p))
    # BH-FDR over compatible-variant paired tests
    if praw:
        ps = sorted(praw, key=lambda t: t[1]); n = len(ps); prev = 1.0; q = {}
        for rank in range(n-1, -1, -1):
            i, p = ps[rank]; prev = min(prev, p*n/(rank+1)); q[i] = prev
        for i, qv in q.items():
            eff[i]["q_bh"] = qv; eff[i]["survives_q05"] = qv < 0.05

    # bootstrap 95% CI for race/sex TPR-gap deltas (C1, C3) on common-eval individuals
    def tpr_gap(yhat, ga, gb, yv, ids):
        pa = ids[(yv[ids]==1) & ga[ids]]; pb = ids[(yv[ids]==1) & gb[ids]]
        if len(pa)==0 or len(pb)==0: return np.nan
        return yhat[pa].mean() - yhat[pb].mean()
    boot = {}
    nE = len(ev_y2)
    for v in COMPATIBLE:
        for grp, ga, gb in [("race", race_a, race_c), ("sex", sex_m, sex_f)]:
            point = np.mean([ (tpr_gap(preds[(v,m,s)], ga, gb, ev_y2, np.arange(nE)) -
                               tpr_gap(preds[(BASE,m,s)], ga, gb, ev_y2, np.arange(nE)))
                              for m in MODELS for s in SEEDS])
            bs = np.empty(2000)
            for bi in range(2000):
                ridx = RNG.randint(0, nE, nE)
                bs[bi] = np.nanmean([ (tpr_gap(preds[(v,m,s)], ga, gb, ev_y2, ridx) -
                                       tpr_gap(preds[(BASE,m,s)], ga, gb, ev_y2, ridx))
                                      for m in MODELS for s in SEEDS])
            lo, hi = np.percentile(bs, [2.5, 97.5])
            boot[f"{v}_{grp}_tpr"] = {"point": float(point), "ci_lo": float(lo), "ci_hi": float(hi),
                                     "excludes_zero": bool(lo>0 or hi<0)}

    # seed-baseline churn (C0 across seeds) vs lineage churn
    base_ch = [float((preds[(BASE,m,si)] != preds[(BASE,m,sj)]).mean())
               for m in MODELS for i,si in enumerate(SEEDS) for sj in SEEDS[i+1:]]
    lin_ch = {v: float(np.mean([ (preds[(v,m,s)] != preds[(BASE,m,s)]).mean() for m in MODELS for s in SEEDS]))
              for v in VARIANTS if v != BASE}

    # Friedman across variants per model (common_acc)
    fried = {}
    for m in MODELS:
        blocks = [[float(df[(df.variant==v)&(df.model==m)&(df.seed==s)]["common_acc"].iloc[0]) for v in VARIANTS] for s in SEEDS]
        arr = np.array(blocks)
        fr = friedmanchisquare(*[arr[:, j] for j in range(arr.shape[1])])
        fried[m] = {"chi2": float(fr.statistic), "p": float(fr.pvalue)}

    summary = {"lineage": lineage,
               "variant_model_means": df.groupby(["variant","model"])[OUT].mean().round(4).reset_index().to_dict("records"),
               "effects": eff, "bootstrap_tpr": boot,
               "seed_baseline_churn_mean": float(np.mean(base_ch)), "seed_baseline_churn_sd": float(np.std(base_ch, ddof=1)),
               "lineage_churn": lin_ch, "friedman_variants": fried}
    json.dump(summary, open(f"{ROOT}/results/compas_summary.json", "w"), indent=1, default=str)
    print("LINEAGE", json.dumps(lineage, indent=1), flush=True)
    print("\nEFFECTS (compatible variants, BH-FDR):", flush=True)
    for r in eff:
        tag = "" if r["variant"]=="C2_isrecid" else (" q=%.4f %s" % (r.get("q_bh",float('nan')), "OK" if r.get("survives_q05") else "drop"))
        print("  %-13s %-14s mean=%+.4f churn=%.3f p=%.4f%s" % (r["variant"], r["outcome"], r["mean"], r["churn_mean"], r["p_raw"], tag), flush=True)
    print("\nseed-baseline churn = %.4f; lineage churn:" % np.mean(base_ch), {k: round(v,4) for k,v in lin_ch.items()}, flush=True)
    print("bootstrap TPR CIs:", json.dumps(boot, indent=1), flush=True)
    print("Friedman:", fried, flush=True)


if __name__ == "__main__":
    main()
