#!/usr/bin/env python3
"""Paper 4: aggregate unit results into CSV tables, statistics, and summary JSON."""
import glob, json, os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats as st
from scipy.stats import spearmanr, friedmanchisquare, wilcoxon

ROOT = str(Path(__file__).resolve().parents[1])
ADULT_VARIANTS = ["V0", "V1_workforce", "V3_drop", "V3_impute", "V4_dedup", "V5_uniform", "V5_hist"]
ACS_VARIANTS = ["T25", "T50", "T85"]
MODELS = ["logreg", "rf", "hgb"]
SEEDS = [0, 1, 2, 3, 4]
OUTCOMES = ["common_acc", "common_auc", "common_ece", "sex_dp_diff", "sex_tpr_gap",
            "race_dp_diff", "race_tpr_gap", "own_acc", "own_auc", "own_ece", "common_posrate_pred"]


def load_units():
    rows, imps = [], {}
    for f in sorted(glob.glob(f"{ROOT}/results/raw/*.json")):
        d = json.load(open(f))
        imps[(d["case"], d["variant"], d["model"], d["seed"])] = d.pop("perm_importance")
        rows.append(d)
    return pd.DataFrame(rows), imps


def preds(case, v, m, s):
    return np.load(f"{ROOT}/results/preds/{case}_{v}_{m}_s{s}.npy")


def main():
    df, imps = load_units()
    df.to_csv(f"{ROOT}/results/unit_metrics.csv", index=False)
    summary = {}

    for case, variants, base in (("adult", ADULT_VARIANTS, "V0"), ("acs", ACS_VARIANTS, "T50")):
        sub = df[df["case"] == case]
        # per variant x model means/sds
        agg = sub.groupby(["variant", "model"])[OUTCOMES + ["n_train", "pos_rate_train"]].agg(["mean", "std"])
        agg.columns = ["_".join(c) for c in agg.columns]
        agg.reset_index().to_csv(f"{ROOT}/results/{case}_variant_model_summary.csv", index=False)

        # deltas vs base, paired by (model, seed)
        deltas = []
        for v in variants:
            if v == base: continue
            for m in MODELS:
                for s in SEEDS:
                    a = sub[(sub.variant == v) & (sub.model == m) & (sub.seed == s)]
                    b = sub[(sub.variant == base) & (sub.model == m) & (sub.seed == s)]
                    if len(a) == 0 or len(b) == 0: continue
                    row = {"variant": v, "model": m, "seed": s}
                    for o in OUTCOMES:
                        row[f"d_{o}"] = float(a[o].iloc[0] - b[o].iloc[0])
                    pa, pb = preds(case, v, m, s), preds(case, base, m, s)
                    row["churn_vs_base"] = float((pa != pb).mean())
                    row["abs_d_common_acc"] = abs(row["d_common_acc"])
                    deltas.append(row)
        dd = pd.DataFrame(deltas)
        dd.to_csv(f"{ROOT}/results/{case}_deltas_vs_base.csv", index=False)

        # effect sizes per variant (pooled over models, seed-paired): mean, sd, cohen d, wilcoxon
        eff = []
        for v in [x for x in variants if x != base]:
            dv = dd[dd.variant == v]
            for o in ["d_common_acc", "d_common_auc", "d_common_ece", "d_sex_dp_diff",
                      "d_sex_tpr_gap", "d_race_tpr_gap", "churn_vs_base"]:
                x = dv[o].to_numpy()
                rec = {"variant": v, "outcome": o, "mean": float(np.mean(x)), "sd": float(np.std(x, ddof=1)),
                       "min": float(np.min(x)), "max": float(np.max(x)), "n": len(x)}
                rec["cohen_d"] = float(np.mean(x) / np.std(x, ddof=1)) if np.std(x, ddof=1) > 0 else np.nan
                if o != "churn_vs_base":
                    try:
                        w = wilcoxon(x, zero_method="wilcox")
                        rec["wilcoxon_p"] = float(w.pvalue)
                    except ValueError:
                        rec["wilcoxon_p"] = np.nan
                eff.append(rec)
                # per-model breakdown
                for m in MODELS:
                    xm = dv[dv.model == m][o].to_numpy()
                    if len(xm):
                        eff.append({"variant": v, "outcome": o, "model": m,
                                    "mean": float(np.mean(xm)), "sd": float(np.std(xm, ddof=1)),
                                    "min": float(np.min(xm)), "max": float(np.max(xm)), "n": len(xm)})
        pd.DataFrame(eff).to_csv(f"{ROOT}/results/{case}_effect_sizes.csv", index=False)

        # full pairwise churn matrix (mean over model x seed)
        churn_mat = pd.DataFrame(0.0, index=variants, columns=variants)
        for i, v1 in enumerate(variants):
            for v2 in variants:
                vals = [(preds(case, v1, m, s) != preds(case, v2, m, s)).mean()
                        for m in MODELS for s in SEEDS]
                churn_mat.loc[v1, v2] = float(np.mean(vals))
        churn_mat.to_csv(f"{ROOT}/results/{case}_churn_matrix.csv")

        # importance rank correlation matrix (Spearman, paired model/seed, mean)
        ic = pd.DataFrame(0.0, index=variants, columns=variants)
        feat = sorted(next(iter(imps.values())).keys()) if case == "adult" else None
        for v1 in variants:
            for v2 in variants:
                vals = []
                for m in MODELS:
                    for s in SEEDS:
                        i1 = imps.get((case, v1, m, s)); i2 = imps.get((case, v2, m, s))
                        if i1 is None or i2 is None: continue
                        ks = sorted(i1.keys())
                        vals.append(spearmanr([i1[k] for k in ks], [i2[k] for k in ks]).statistic)
                ic.loc[v1, v2] = float(np.mean(vals))
        ic.to_csv(f"{ROOT}/results/{case}_importance_rankcorr.csv")

        # save mean importance per variant (pooled models/seeds) for reporting
        imp_rows = []
        for v in variants:
            for m in MODELS:
                vecs = [imps[(case, v, m, s)] for s in SEEDS if (case, v, m, s) in imps]
                if not vecs: continue
                ks = sorted(vecs[0].keys())
                mean_imp = {k: float(np.mean([vv[k] for vv in vecs])) for k in ks}
                imp_rows.append({"variant": v, "model": m, **mean_imp})
        pd.DataFrame(imp_rows).to_csv(f"{ROOT}/results/{case}_importance_means.csv", index=False)

        # stats tests
        tests = {}
        # Friedman across variants per model (common_acc over seeds)
        for m in MODELS:
            blocks = []
            for s in SEEDS:
                r = [float(sub[(sub.variant == v) & (sub.model == m) & (sub.seed == s)]["common_acc"].iloc[0])
                     for v in variants]
                blocks.append(r)
            arr = np.array(blocks)
            fr = friedmanchisquare(*[arr[:, j] for j in range(arr.shape[1])])
            tests[f"friedman_variants_{m}"] = {"chi2": float(fr.statistic), "p": float(fr.pvalue),
                                               "blocks": "seeds", "treatments": "variants",
                                               "outcome": "common_acc"}
        # Friedman across models on lineage sensitivity |d_common_acc| (blocks = variant x seed)
        nb = dd[["variant", "seed"]].drop_duplicates()
        cols = {m: [] for m in MODELS}
        for _, r in nb.iterrows():
            for m in MODELS:
                x = dd[(dd.variant == r.variant) & (dd.seed == r.seed) & (dd.model == m)]
                cols[m].append(abs(float(x["d_common_acc"].iloc[0])))
        fr = friedmanchisquare(*[np.array(cols[m]) for m in MODELS])
        tests["friedman_models_sensitivity"] = {"chi2": float(fr.statistic), "p": float(fr.pvalue),
                                                "blocks": "variant x seed", "treatments": "models",
                                                "outcome": "|d_common_acc|"}
        # pairwise wilcoxon between models on |d_common_acc| and churn
        for o in ["abs_d_common_acc", "churn_vs_base"]:
            for m1, m2 in [("logreg", "rf"), ("logreg", "hgb"), ("rf", "hgb")]:
                x1, x2 = [], []
                for _, r in nb.iterrows():
                    x1.append(float(dd[(dd.variant == r.variant) & (dd.seed == r.seed) & (dd.model == m1)][o].iloc[0]))
                    x2.append(float(dd[(dd.variant == r.variant) & (dd.seed == r.seed) & (dd.model == m2)][o].iloc[0]))
                try:
                    w = wilcoxon(np.array(x1), np.array(x2))
                    tests[f"wilcoxon_{o}_{m1}_vs_{m2}"] = {"p": float(w.pvalue),
                                                           "median_1": float(np.median(x1)),
                                                           "median_2": float(np.median(x2)), "n": len(x1)}
                except ValueError:
                    pass
        # mean churn / |dacc| per model
        per_model = dd.groupby("model")[["churn_vs_base", "abs_d_common_acc"]].agg(["mean", "median", "max"])
        per_model.columns = ["_".join(c) for c in per_model.columns]
        tests["per_model_sensitivity"] = json.loads(per_model.to_json())
        summary[case] = {"tests": tests,
                         "n_units": int(len(sub)),
                         "baseline": base}

    json.dump(summary, open(f"{ROOT}/results/results_summary.json", "w"), indent=1, default=str)
    print("aggregate done; units:", len(df))


if __name__ == "__main__":
    main()

