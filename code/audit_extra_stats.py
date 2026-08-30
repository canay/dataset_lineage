#!/usr/bin/env python3
"""Q1 audit extra statistics (no model retraining).

Computed entirely from saved per-run metrics (results/raw/*.json), saved common-eval
prediction vectors (results/preds/*.npy), and the frozen prepared frames
(data/adult_prepared.pkl, data/acs_prepared.pkl). Addresses audit actions:
  A-003 multiple-comparison correction (Benjamini-Hochberg) + Nemenyi post-hoc (RF)
  A-006 bootstrap 95% CIs for subgroup TPR-gap deltas (small-cell uncertainty)
  A-010 seed-baseline churn (within-variant across-seed) vs lineage churn
  A-011 ACSIncome accuracy Wilcoxon p-values
  A-013 permutation-based Friedman p-values (low-block-count robustness)
All Wilcoxon tests: two-sided, zeros dropped (zero_method='wilcox'), with SciPy's
automatic method selection.
"""
import glob, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare, rankdata, studentized_range

ROOT = Path(__file__).resolve().parents[1]
ADULT = ["V0", "V1_workforce", "V3_drop", "V3_impute", "V4_dedup", "V5_uniform", "V5_hist"]
ACS = ["T25", "T50", "T85"]
MODELS = ["logreg", "rf", "hgb"]
SEEDS = [0, 1, 2, 3, 4]
GAP_OUTCOMES = ["d_common_acc", "d_common_auc", "d_common_ece",
                "d_sex_dp_diff", "d_sex_tpr_gap", "d_race_tpr_gap"]
RNG = np.random.RandomState(20260622)


def load_units():
    rows = []
    for f in sorted(glob.glob(f"{ROOT}/results/raw/*.json")):
        d = json.load(open(f)); d.pop("perm_importance", None); rows.append(d)
    return pd.DataFrame(rows)


def preds(case, v, m, s):
    return np.load(f"{ROOT}/results/preds/{case}_{v}_{m}_s{s}.npy")


def wilc(x):
    x = np.asarray(x, float)
    if np.allclose(x, 0):
        return np.nan
    return float(wilcoxon(x, zero_method="wilcox", alternative="two-sided").pvalue)


def bh_fdr(pvals):
    """Benjamini-Hochberg q-values."""
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    q = np.empty(n)
    prev = 1.0
    for rank in range(n - 1, -1, -1):
        i = order[rank]
        val = p[i] * n / (rank + 1)
        prev = min(prev, val)
        q[i] = prev
    return q


def deltas(df, case, variants, base):
    sub = df[df.case == case]
    recs = []
    for v in variants:
        if v == base:
            continue
        for m in MODELS:
            for s in SEEDS:
                a = sub[(sub.variant == v) & (sub.model == m) & (sub.seed == s)]
                b = sub[(sub.variant == base) & (sub.model == m) & (sub.seed == s)]
                if len(a) == 0 or len(b) == 0:
                    continue
                r = {"variant": v, "model": m, "seed": s}
                for o in ["common_acc", "common_auc", "common_ece",
                          "sex_dp_diff", "sex_tpr_gap", "race_dp_diff", "race_tpr_gap"]:
                    r[f"d_{o}"] = float(a[o].iloc[0] - b[o].iloc[0])
                r["churn_vs_base"] = float((preds(case, v, m, s) != preds(case, base, m, s)).mean())
                recs.append(r)
    return pd.DataFrame(recs)


def adult_multiplicity(dd):
    """A-003: paired Wilcoxon per (variant, outcome) + BH-FDR across the full family."""
    rows = []
    for v in [x for x in ADULT if x != "V0"]:
        dv = dd[dd.variant == v]
        for o in GAP_OUTCOMES:
            x = dv[o].to_numpy()
            mean = float(np.mean(x)); sd = float(np.std(x, ddof=1))
            rows.append({"variant": v, "outcome": o, "mean": mean, "sd": sd,
                         "d_z": mean / sd if sd > 0 else np.nan, "p_raw": wilc(x), "n": len(x)})
    res = pd.DataFrame(rows)
    res["q_bh"] = bh_fdr(res["p_raw"].fillna(1.0).to_numpy())
    res["survives_q05"] = res["q_bh"] < 0.05
    return res


def nemenyi_rf(df):
    """A-003: Friedman + Nemenyi post-hoc on RF common_acc across 7 variants, 5 seed blocks."""
    sub = df[(df.case == "adult") & (df.model == "rf")]
    M = np.array([[float(sub[(sub.variant == v) & (sub.seed == s)]["common_acc"].iloc[0])
                   for v in ADULT] for s in SEEDS])  # blocks=seeds x treatments=variants
    fr = friedmanchisquare(*[M[:, j] for j in range(M.shape[1])])
    ranks = np.array([rankdata(-row) for row in M])  # higher acc = rank 1
    avg = ranks.mean(0)
    k, n = len(ADULT), len(SEEDS)
    q_alpha = studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2)
    cd = q_alpha * np.sqrt(k * (k + 1) / (6 * n))
    pairs = []
    for i in range(k):
        for j in range(i + 1, k):
            diff = abs(avg[i] - avg[j])
            pairs.append({"a": ADULT[i], "b": ADULT[j], "rank_diff": float(diff),
                          "exceeds_CD": bool(diff > cd)})
    return {"friedman_chi2": float(fr.statistic), "friedman_p": float(fr.pvalue),
            "avg_ranks": {v: float(r) for v, r in zip(ADULT, avg)},
            "critical_difference_05": float(cd), "pairs": pairs}


def perm_friedman(df, case, variants):
    """A-013: permutation p for the Friedman statistic (low block count robustness)."""
    out = {}
    sub = df[df.case == case]
    for m in MODELS:
        sm = sub[sub.model == m]
        M = np.array([[float(sm[(sm.variant == v) & (sm.seed == s)]["common_acc"].iloc[0])
                       for v in variants] for s in SEEDS])
        obs = friedmanchisquare(*[M[:, j] for j in range(M.shape[1])]).statistic
        count, B = 0, 20000
        for _ in range(B):
            Mp = np.array([RNG.permutation(row) for row in M])
            stat = friedmanchisquare(*[Mp[:, j] for j in range(Mp.shape[1])]).statistic
            if stat >= obs - 1e-12:
                count += 1
        out[m] = {"chi2": float(obs), "p_perm": (count + 1) / (B + 1),
                  "p_asymptotic": float(friedmanchisquare(*[M[:, j] for j in range(M.shape[1])]).pvalue)}
    return out


def seed_baseline_churn(case, variants, base):
    """A-010: within-variant across-seed churn (training-stochasticity floor) vs lineage churn."""
    # baseline: same variant (base), different seeds, same model
    base_ch = []
    for m in MODELS:
        ps = [preds(case, base, m, s) for s in SEEDS]
        for i in range(len(SEEDS)):
            for j in range(i + 1, len(SEEDS)):
                base_ch.append(float((ps[i] != ps[j]).mean()))
    # also within each non-base variant across seeds (for context)
    within = {}
    for v in variants:
        vals = []
        for m in MODELS:
            ps = [preds(case, v, m, s) for s in SEEDS]
            for i in range(len(SEEDS)):
                for j in range(i + 1, len(SEEDS)):
                    vals.append(float((ps[i] != ps[j]).mean()))
        within[v] = {"mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1))}
    # lineage churn: same seed, base vs variant
    lin = {}
    for v in variants:
        if v == base:
            continue
        vals = [float((preds(case, v, m, s) != preds(case, base, m, s)).mean())
                for m in MODELS for s in SEEDS]
        lin[v] = {"mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1))}
    return {"baseline_seed_churn_mean": float(np.mean(base_ch)),
            "baseline_seed_churn_sd": float(np.std(base_ch, ddof=1)),
            "within_variant_seed_churn": within,
            "lineage_churn_same_seed": lin}


def bootstrap_tpr_ci(group="race", variants=("V3_drop", "V5_hist", "V5_uniform"),
                     B=4000):
    """A-006: bootstrap 95% CI for TPR-gap delta (variant - V0) on the common-eval set."""
    d = pd.read_pickle(f"{ROOT}/data/adult_prepared.pkl")
    ev = d[d["is_eval"]]
    y = ev["y"].to_numpy()
    if group == "race":
        ga = (ev["race"] == "White").to_numpy(); gb = (ev["race"] == "Black").to_numpy()
    else:
        ga = (ev["sex"] == "Male").to_numpy(); gb = (ev["sex"] == "Female").to_numpy()
    posa = (y == 1) & ga; posb = (y == 1) & gb
    n = len(y)
    cells = {"n_pos_groupA": int(posa.sum()), "n_pos_groupB": int(posb.sum())}

    def gap(yhat, idx):
        a = yhat[idx][posa[idx]]; b = yhat[idx][posb[idx]]
        if len(a) == 0 or len(b) == 0:
            return np.nan
        return a.mean() - b.mean()

    out = {"group": group, "cells": cells, "variants": {}}
    P0 = {(m, s): preds("adult", "V0", m, s) for m in MODELS for s in SEEDS}
    for v in variants:
        Pv = {(m, s): preds("adult", v, m, s) for m in MODELS for s in SEEDS}
        point = np.mean([gap(Pv[(m, s)], np.arange(n)) - gap(P0[(m, s)], np.arange(n))
                         for m in MODELS for s in SEEDS])
        boot = np.empty(B)
        for bi in range(B):
            idx = RNG.randint(0, n, n)
            deltas_ = [gap(Pv[(m, s)], idx) - gap(P0[(m, s)], idx)
                       for m in MODELS for s in SEEDS]
            boot[bi] = np.nanmean(deltas_)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out["variants"][v] = {"point_delta": float(point), "ci95_lo": float(lo),
                              "ci95_hi": float(hi), "excludes_zero": bool(lo > 0 or hi < 0)}
    return out


def main():
    df = load_units()
    result = {}

    dd = deltas(df, "adult", ADULT, "V0")
    mult = adult_multiplicity(dd)
    result["adult_multiplicity_bh"] = mult.to_dict(orient="records")
    result["adult_nemenyi_rf"] = nemenyi_rf(df)
    result["adult_perm_friedman"] = perm_friedman(df, "adult", ADULT)
    result["adult_seed_churn"] = seed_baseline_churn("adult", ADULT, "V0")
    result["adult_bootstrap_race_tpr"] = bootstrap_tpr_ci("race", ("V3_drop", "V5_hist", "V5_uniform", "V1_workforce", "V4_dedup"))
    result["adult_bootstrap_sex_tpr"] = bootstrap_tpr_ci("sex", ("V3_impute", "V3_drop"))

    # A-011 ACS accuracy Wilcoxon
    ddacs = deltas(df, "acs", ACS, "T50")
    acs_acc = {}
    for v in ["T25", "T85"]:
        x = ddacs[ddacs.variant == v]["d_common_acc"].to_numpy()
        acs_acc[v] = {"mean": float(np.mean(x)), "sd": float(np.std(x, ddof=1)),
                      "d_z": float(np.mean(x) / np.std(x, ddof=1)), "p_raw": wilc(x), "n": len(x)}
    # also AUC p for context
    for v in ["T25", "T85"]:
        x = ddacs[ddacs.variant == v]["d_common_auc"].to_numpy()
        acs_acc[f"{v}_auc"] = {"mean": float(np.mean(x)), "p_raw": wilc(x)}
    result["acs_accuracy_wilcoxon"] = acs_acc

    json.dump(result, open(f"{ROOT}/results/audit_extra_stats.json", "w"), indent=1, default=str)

    # readable summary
    print("=== A-003 Adult BH-FDR (survivors and borderline) ===")
    for r in result["adult_multiplicity_bh"]:
        flag = "OK" if r["survives_q05"] else "DROP"
        print(f"  {r['variant']:13s} {r['outcome']:16s} mean={r['mean']:+.4f} p={r['p_raw']:.4f} q={r['q_bh']:.4f} [{flag}]")
    nem = result["adult_nemenyi_rf"]
    print(f"\n=== A-003 RF Friedman chi2={nem['friedman_chi2']:.2f} p={nem['friedman_p']:.4f} CD={nem['critical_difference_05']:.2f} ===")
    print("  pairs exceeding CD:", [f"{p['a']}|{p['b']}" for p in nem["pairs"] if p["exceeds_CD"]] or "NONE")
    print("\n=== A-013 permutation Friedman (per model) ===")
    for m, r in result["adult_perm_friedman"].items():
        print(f"  {m:7s} chi2={r['chi2']:.2f} p_perm={r['p_perm']:.4f} p_asym={r['p_asymptotic']:.4f}")
    sc = result["adult_seed_churn"]
    print(f"\n=== A-010 seed-baseline churn (V0 across seeds) = {sc['baseline_seed_churn_mean']:.4f} +/- {sc['baseline_seed_churn_sd']:.4f} ===")
    for v, r in sc["lineage_churn_same_seed"].items():
        print(f"  lineage churn {v:13s} = {r['mean']:.4f}  (excess over seed floor = {r['mean']-sc['baseline_seed_churn_mean']:+.4f})")
    print("\n=== A-006 bootstrap race TPR-gap delta 95% CI ===")
    rb = result["adult_bootstrap_race_tpr"]
    print(f"  cells: White+ ={rb['cells']['n_pos_groupA']}, Black+ ={rb['cells']['n_pos_groupB']}")
    for v, r in rb["variants"].items():
        print(f"  {v:13s} delta={r['point_delta']:+.4f} CI=[{r['ci95_lo']:+.4f},{r['ci95_hi']:+.4f}] excl0={r['excludes_zero']}")
    print("\n=== A-006 bootstrap sex TPR-gap delta 95% CI ===")
    for v, r in result["adult_bootstrap_sex_tpr"]["variants"].items():
        print(f"  {v:13s} delta={r['point_delta']:+.4f} CI=[{r['ci95_lo']:+.4f},{r['ci95_hi']:+.4f}] excl0={r['excludes_zero']}")
    print("\n=== A-011 ACS accuracy Wilcoxon ===")
    for v, r in result["acs_accuracy_wilcoxon"].items():
        print(f"  {v:8s} {r}")
    print("\nwrote results/audit_extra_stats.json")


if __name__ == "__main__":
    main()
