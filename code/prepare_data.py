#!/usr/bin/env python3
"""Paper 4: prepare canonical data objects, verify lineage facts, freeze common eval sets.

Stage A (adult): verify OpenML-1590 copy against UCI adult.data/adult.test,
compute lineage statistics, freeze common evaluation set.
Stage B (acs): build ACSIncome-style re-derivation from raw 2018 1-Year PUMS (Oregon),
reproducing the folktables adult_filter and feature list, keeping continuous PINCP.
Stage C (german): compare UCI Statlog German Credit vs corrected South German Credit.

Resumable: each stage writes its output once; reruns skip completed stages.
"""
import json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = str(Path(__file__).resolve().parents[1])
P2_ADULT = os.path.join(ROOT, "data", "adult_prepared.pkl")
RNG_EVAL = 20260611
N_EVAL_ADULT = 4000
N_EVAL_ACS = 3000

UCI_COLS = ["age", "workclass", "fnlwgt", "education", "education-num",
            "marital-status", "occupation", "relationship", "race", "sex",
            "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]

NUM_COLS = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CAT_COLS = ["workclass", "education", "marital-status", "occupation",
            "relationship", "race", "sex", "native-country"]


def stage_adult():
    out_pkl = f"{ROOT}/data/adult_prepared.pkl"
    out_json = f"{ROOT}/results/lineage_stats.json"
    if os.path.exists(out_pkl) and os.path.exists(out_json):
        print("adult stage done, skip"); return
    d = pd.read_pickle(P2_ADULT).rename(columns={"__target__": "y"})
    for c in d.columns:
        if str(d[c].dtype) == "category":
            d[c] = d[c].astype(object)
    tr = pd.read_csv(f"{ROOT}/data/uci/adult.data", header=None, names=UCI_COLS,
                     skipinitialspace=True, na_values="?")
    te = pd.read_csv(f"{ROOT}/data/uci/adult.test", header=None, names=UCI_COLS,
                     skipinitialspace=True, skiprows=1, na_values="?")
    te["income"] = te["income"].str.rstrip(".")
    uci = pd.concat([te, tr], ignore_index=True)  # OpenML 1590 order = test rows then train rows
    feat = [c for c in UCI_COLS if c != "income"]
    same_num = all(np.array_equal(d[c].to_numpy(), uci[c].to_numpy()) for c in NUM_COLS)
    same_cat = all((d[c].fillna("\0") == uci[c].fillna("\0")).all() for c in CAT_COLS)
    same_y = np.array_equal(d["y"].to_numpy(), (uci["income"] == ">50K").astype(int).to_numpy())
    assert same_num and same_cat and same_y, (same_num, same_cat, same_y)

    n = len(d)
    hist_test = np.zeros(n, bool); hist_test[:16281] = True  # adult.test block
    d["hist_test"] = hist_test

    # lineage statistics (all computed from the data itself)
    becker = {
        "AAGE>16 violations (age<=16)": int((d["age"] <= 16).sum()),
        "AFNLWGT>1 violations (fnlwgt<=1)": int((d["fnlwgt"] <= 1).sum()),
        "HRSWK>0 violations (hours<=0)": int((d["hours-per-week"] <= 0).sum()),
    }
    miss_any = d[feat].isna().any(axis=1)
    dup_full = d[feat + ["y"]].duplicated()
    dup_feat = d[feat].duplicated()
    wf_drop = d["workclass"].isna() | d["workclass"].isin(["Never-worked", "Without-pay"])
    stats = {
        "n_total": n, "n_hist_train": int((~hist_test).sum()), "n_hist_test": int(hist_test.sum()),
        "verified_equal_to_uci_concat": bool(same_num and same_cat and same_y),
        "order": "adult.test rows 0..16280 then adult.data rows 16281..48841",
        "becker_predicate_violations": becker,
        "pos_rate": float(d["y"].mean()),
        "missing_counts": {c: int(d[c].isna().sum()) for c in feat if d[c].isna().any()},
        "rows_with_any_missing": int(miss_any.sum()),
        "exact_duplicates_features_and_label": int(dup_full.sum()),
        "duplicates_features_only": int(dup_feat.sum()),
        "label_conflicting_feature_duplicates": int(dup_feat.sum() - dup_full.sum()),
        "workforce_filter_removed": int(wf_drop.sum()),
        "workforce_filter_kept": int((~wf_drop).sum()),
        "pos_rate_after_workforce": float(d.loc[~wf_drop, "y"].mean()),
        "pos_rate_complete_cases": float(d.loc[~miss_any, "y"].mean()),
        "pos_rate_hist_train": float(d.loc[~hist_test, "y"].mean()),
        "pos_rate_hist_test": float(d.loc[hist_test, "y"].mean()),
    }
    # frozen common evaluation set: complete cases that pass the workforce filter,
    # stratified by label, excluded from every variant's training pool
    elig = np.where((~miss_any) & (~wf_drop))[0]
    rng = np.random.RandomState(RNG_EVAL)
    y_e = d["y"].to_numpy()[elig]
    pos = elig[y_e == 1]; neg = elig[y_e == 0]
    n_pos = int(round(N_EVAL_ADULT * y_e.mean()))
    eval_idx = np.sort(np.concatenate([rng.choice(pos, n_pos, replace=False),
                                       rng.choice(neg, N_EVAL_ADULT - n_pos, replace=False)]))
    stats["common_eval"] = {"n": int(len(eval_idx)), "eligible_pool": int(len(elig)),
                            "pos_rate": float(d["y"].to_numpy()[eval_idx].mean()),
                            "rng_seed": RNG_EVAL,
                            "definition": "complete-case AND workforce-pass, stratified by label"}
    d["is_eval"] = False; d.loc[eval_idx, "is_eval"] = True
    d.to_pickle(out_pkl)
    json.dump(stats, open(out_json, "w"), indent=1)
    print("adult prepared:", stats)


ACS_FEATURES = ["AGEP", "COW", "SCHL", "MAR", "OCCP", "POBP", "RELP", "WKHP", "SEX", "RAC1P"]


def stage_acs():
    out_pkl = f"{ROOT}/data/acs_prepared.pkl"
    out_json = f"{ROOT}/results/acs_stats.json"
    if os.path.exists(out_pkl) and os.path.exists(out_json):
        print("acs stage done, skip"); return
    import zipfile
    cols = ACS_FEATURES + ["PINCP", "PWGTP"]
    with zipfile.ZipFile(f"{ROOT}/data/census_or_2018.zip") as z:
        with z.open("psam_p41.csv") as f:
            d = pd.read_csv(f, usecols=cols)
    n_raw = len(d)
    # folktables ACSIncome adult_filter (Ding et al. 2021): AGEP>16, PINCP>100, WKHP>0, PWGTP>=1
    m = (d["AGEP"] > 16) & (d["PINCP"] > 100) & (d["WKHP"] > 0) & (d["PWGTP"] >= 1)
    counts = {
        "n_raw_person_records": n_raw,
        "fail_AGEP>16": int((~(d["AGEP"] > 16)).sum()),
        "fail_PINCP>100": int((~(d["PINCP"] > 100)).sum()),
        "fail_WKHP>0": int((~(d["WKHP"] > 0)).sum()),
        "fail_PWGTP>=1": int((~(d["PWGTP"] >= 1)).sum()),
        "n_after_filter": int(m.sum()),
    }
    d = d[m].reset_index(drop=True)
    d = d.dropna(subset=ACS_FEATURES).reset_index(drop=True)  # folktables also drops NA in features
    counts["n_after_feature_na_drop"] = len(d)
    thr = {"T25": 25000, "T50": 50000, "T85": 84700}  # 84700 = 50000 CPI-U adjusted 1994->2018
    for k, t in thr.items():
        d[f"y_{k}"] = (d["PINCP"] > t).astype(int)
        counts[f"pos_rate_{k}"] = float(d[f"y_{k}"].mean())
    rng = np.random.RandomState(RNG_EVAL)
    y = d["y_T50"].to_numpy()
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    n_pos = int(round(N_EVAL_ACS * y.mean()))
    eval_idx = np.sort(np.concatenate([rng.choice(pos, n_pos, replace=False),
                                       rng.choice(neg, N_EVAL_ACS - n_pos, replace=False)]))
    d["is_eval"] = False; d.loc[eval_idx, "is_eval"] = True
    counts["common_eval_n"] = int(len(eval_idx))
    counts["thresholds_usd"] = thr
    d.to_pickle(out_pkl)
    json.dump(counts, open(out_json, "w"), indent=1)
    print("acs prepared:", counts)


def stage_german():
    out_json = f"{ROOT}/results/german_credit_check.json"
    if os.path.exists(out_json):
        print("german stage done, skip"); return
    g = pd.read_csv(f"{ROOT}/data/uci/german.data", sep=" ", header=None)
    s = pd.read_csv(f"{ROOT}/data/uci/sgc/SouthGermanCredit.asc", sep=" ")
    # statlog german.data numeric columns by 0-based position
    num_map = {1: "laufzeit", 4: "hoehe", 7: "rate", 10: "wohnzeit", 12: "alter", 15: "bishkred", 17: "pers"}
    checks = {f"col{gi}({sn})_equal": bool(np.array_equal(g[gi].to_numpy(), s[sn].to_numpy()))
              for gi, sn in num_map.items()}
    checks["target_equal_good1"] = bool(np.array_equal((g[20] == 1).to_numpy(), (s["kredit"] == 1).to_numpy()))
    # categorical columns: compare integer code distributions after stripping 'A<col><code>' prefix
    cat_pos = {0: "laufkont", 2: "moral", 3: "verw", 5: "sparkont", 6: "beszeit", 8: "famges",
               9: "buerge", 11: "verm", 13: "weitkred", 14: "wohn", 16: "beruf", 18: "telef", 19: "gastarb"}
    cat_checks = {}
    for gi, sn in cat_pos.items():
        gc = g[gi].astype(str).str.extract(r"A\d*?(\d{1,2})$")[0]
        gv = pd.crosstab(gc, g[20]).to_numpy()
        sv = pd.crosstab(s[sn], s["kredit"]).to_numpy()
        same_marginals = (sorted(gv.flatten().tolist()) == sorted(sv.flatten().tolist()))
        cat_checks[f"{sn}_codecount_match"] = bool(same_marginals)
    out = {"shapes": {"statlog": list(g.shape), "south_german": list(s.shape)},
           "numeric_checks": checks, "categorical_marginal_checks": cat_checks}
    json.dump(out, open(out_json, "w"), indent=1)
    print("german check:", out)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    t0 = time.time()
    if which in ("all", "adult"): stage_adult()
    if which in ("all", "acs"): stage_acs()
    if which in ("all", "german"): stage_german()
    print("elapsed", round(time.time() - t0, 1))

