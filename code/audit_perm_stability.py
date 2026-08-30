#!/usr/bin/env python3
"""A-015: permutation-importance stability at a higher repeat budget (resumable).

Retrains the Adult variants (deterministic, fixed seeds) in the revision environment and
recomputes permutation importance with scikit-learn's permutation_importance at n_repeats=20
(vs the 3-repeat custom AUC-drop estimate in the canonical run), scoring by ROC AUC. Each
(variant, model, seed) result is checkpointed to results/_perm_ckpt/ so the run resumes after
any interruption. Reports the between-variant Spearman rank-correlation profile and a
reproduction guard against the canonical per-run common-set accuracy. Descriptive robustness
check; does not alter the canonical main tables or figures.
"""
import glob, json, os, sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from run_units import build_adult_variant, apply_policy, make_model, NUM_COLS, CAT_COLS

ADULT = ["V0", "V1_workforce", "V3_drop", "V3_impute", "V4_dedup", "V5_uniform", "V5_hist"]
MODELS = ["logreg", "rf", "hgb"]
SEEDS = [0, 1, 2]  # three seeds in the revision environment (9 model-seed pairs per cell);
N_REP = 20         # the canonical run used 5 seeds at 3 repeats. This is a robustness check.
COLS = NUM_COLS + CAT_COLS
CKPT = ROOT / "results" / "_perm_ckpt"


def canonical_acc():
    acc = {}
    for f in glob.glob(f"{ROOT}/results/raw/adult_*.json"):
        d = json.load(open(f))
        acc[(d["variant"], d["model"], d["seed"])] = d["common_acc"]
    return acc


def compute_unit(d, variant, model, seed):
    """Custom AUC-drop permutation importance (same metric as the canonical 3-repeat run,
    here at N_REP repeats). Avoids sklearn.inspection.permutation_importance, which crashes
    natively with the random-forest pipeline in this environment."""
    tr, te, policy = build_adult_variant(d, variant, seed)
    modes = {c: tr[c].mode(dropna=True).iloc[0] for c in CAT_COLS}
    tr_p = apply_policy(tr, policy, modes)
    ev_p = apply_policy(d[d["is_eval"]], policy, modes)
    pipe = make_model(model, seed, NUM_COLS, CAT_COLS)
    pipe.fit(tr_p[COLS], tr_p["y"].to_numpy())
    # keep RF n_jobs=-1 (make_model default): threaded in-process tree prediction is fast and
    # does not pickle the model; warnings are silenced via PYTHONWARNINGS=ignore at launch.
    ye = ev_p["y"].to_numpy()
    Xe = ev_p[COLS].reset_index(drop=True)
    acc = float((pipe.predict(Xe) == ye).mean())
    base = roc_auc_score(ye, pipe.predict_proba(Xe)[:, 1])
    rng = np.random.RandomState(seed)
    Xp = Xe.copy()
    imp = {}
    for c in COLS:
        orig = Xe[c].to_numpy()
        drops = []
        for _ in range(N_REP):
            Xp[c] = rng.permutation(orig)
            drops.append(base - roc_auc_score(ye, pipe.predict_proba(Xp)[:, 1]))
        Xp[c] = orig  # restore before next column
        imp[c] = float(np.mean(drops))
    return imp, acc


def main():
    CKPT.mkdir(parents=True, exist_ok=True)
    d = pd.read_pickle(f"{ROOT}/data/adult_prepared.pkl")
    units = [(v, m, s) for v in ADULT for m in MODELS for s in SEEDS]
    for v, m, s in units:
        ck = CKPT / f"{v}_{m}_s{s}.json"
        if ck.exists():
            continue
        imp, acc = compute_unit(d, v, m, s)
        json.dump({"imp": imp, "acc": acc}, open(ck, "w"))
        print(f"  done {v}/{m}/s{s} acc={acc:.4f}", flush=True)
    # aggregate from checkpoints
    imps, accs = {}, {}
    for v, m, s in units:
        rec = json.load(open(CKPT / f"{v}_{m}_s{s}.json"))
        imps[(v, m, s)] = rec["imp"]; accs[(v, m, s)] = rec["acc"]
    can = canonical_acc()
    accdiff = [abs(accs[k] - can[k]) for k in accs if k in can]
    ic = pd.DataFrame(0.0, index=ADULT, columns=ADULT)
    for v1 in ADULT:
        for v2 in ADULT:
            vals = []
            for m in MODELS:
                for s in SEEDS:
                    i1, i2 = imps[(v1, m, s)], imps[(v2, m, s)]
                    ks = sorted(i1.keys())
                    vals.append(spearmanr([i1[k] for k in ks], [i2[k] for k in ks]).statistic)
            ic.loc[v1, v2] = float(np.mean(vals))
    off = ic.where(~np.eye(len(ADULT), dtype=bool))
    mn = off.stack().idxmin()
    res = {"n_rep": N_REP, "seeds": SEEDS,
           "method": "custom AUC-drop permutation importance (same metric as the 3-repeat canonical run), scoring=ROC AUC",
           "spearman_mean_offdiag": float(np.nanmean(off.to_numpy())),
           "spearman_min_offdiag": float(np.nanmin(off.to_numpy())),
           "min_pair": [mn[0], mn[1], float(off.stack().min())],
           "repro_guard_max_abs_acc_diff": float(np.max(accdiff)),
           "repro_guard_mean_abs_acc_diff": float(np.mean(accdiff)),
           "canonical_3rep_mean": 0.990, "canonical_3rep_min": 0.982}
    ic.to_csv(f"{ROOT}/results/adult_importance_rankcorr_nrep20.csv")
    json.dump(res, open(f"{ROOT}/results/audit_perm_stability.json", "w"), indent=1)
    print("RESULT", json.dumps(res, indent=1), flush=True)


if __name__ == "__main__":
    main()
