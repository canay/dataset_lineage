#!/usr/bin/env python3
"""A-007: complementary all-rows evaluation for the missing-value variants.

The main common-evaluation set is restricted to complete cases that also pass the
workforce filter, which excludes exactly the rows most affected by the missing-value
decision. This script builds a second frozen evaluation set sampled from ALL distributed
rows (so missing-value rows are included), retrains V0 / V3_drop / V3_impute in the
revision environment, scores each variant under its own eval-time missing-value policy,
and reports churn on this all-rows set versus the canonical complete-case set. It is a
directional robustness check; it does not alter the canonical main tables.

Eval-time policies (matching each variant's training policy):
  V0        -> missing categorical filled with the explicit 'Missing' category (trained-in)
  V3_impute -> missing categorical imputed with the training mode
  V3_drop   -> trained on complete cases; missing eval cells imputed with the training mode
"""
import glob, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from run_units import make_model, NUM_COLS, CAT_COLS

MODELS = ["logreg", "rf", "hgb"]
SEEDS = [0, 1, 2, 3, 4]
RNG_EVAL2 = 20260622
N_EVAL = 4000


def main():
    d = pd.read_pickle(f"{ROOT}/data/adult_prepared.pkl").reset_index(drop=True)
    y = d["y"].to_numpy()
    rng = np.random.RandomState(RNG_EVAL2)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    n_pos = int(round(N_EVAL * y.mean()))
    eval_idx = np.sort(np.concatenate([rng.choice(pos, n_pos, replace=False),
                                       rng.choice(neg, N_EVAL - n_pos, replace=False)]))
    is_eval2 = np.zeros(len(d), bool); is_eval2[eval_idx] = True
    ev = d[is_eval2].copy()
    pool_all = d[~is_eval2].copy()
    miss_mask = ev[CAT_COLS].isna().any(axis=1)
    frac_missing_eval = float(miss_mask.mean())

    def fit_predict(variant, model, seed):
        pool = pool_all.copy()
        if variant == "V3_drop":
            pool = pool.dropna(subset=NUM_COLS + CAT_COLS)
        tr, _ = train_test_split(pool, test_size=0.2, random_state=seed,
                                 stratify=pool["y"])
        modes = {c: tr[c].mode(dropna=True).iloc[0] for c in CAT_COLS}
        trf = tr.copy(); evf = ev.copy()
        if variant == "V0":
            for c in CAT_COLS:
                trf[c] = trf[c].fillna("Missing"); evf[c] = evf[c].fillna("Missing")
        else:  # V3_impute and V3_drop both impute at eval; V3_drop train has no missing
            for c in CAT_COLS:
                trf[c] = trf[c].fillna(modes[c]); evf[c] = evf[c].fillna(modes[c])
        pipe = make_model(model, seed, NUM_COLS, CAT_COLS)
        pipe.fit(trf[NUM_COLS + CAT_COLS], trf["y"].to_numpy())
        yhat = pipe.predict(evf[NUM_COLS + CAT_COLS])
        return yhat.astype(int)

    preds = {}
    for v in ["V0", "V3_drop", "V3_impute"]:
        for m in MODELS:
            for s in SEEDS:
                preds[(v, m, s)] = fit_predict(v, m, s)
        print(f"  done {v}")

    ye = ev["y"].to_numpy()
    out = {"eval_set": "all distributed rows (incl. missing/non-workforce)",
           "n_eval": N_EVAL, "frac_eval_rows_with_missing": frac_missing_eval,
           "seed": RNG_EVAL2, "note": "revision-environment supplementary analysis"}
    for v in ["V3_drop", "V3_impute"]:
        ch_all = [float((preds[(v, m, s)] != preds[("V0", m, s)]).mean())
                  for m in MODELS for s in SEEDS]
        # churn restricted to the missing-value rows only
        mm = miss_mask.to_numpy()
        ch_missing = [float((preds[(v, m, s)][mm] != preds[("V0", m, s)][mm]).mean())
                      for m in MODELS for s in SEEDS]
        acc = [float((preds[(v, m, s)] == ye).mean()) for m in MODELS for s in SEEDS]
        out[v] = {"churn_vs_V0_allrows_mean": float(np.mean(ch_all)),
                  "churn_vs_V0_allrows_sd": float(np.std(ch_all, ddof=1)),
                  "churn_vs_V0_missing_rows_only_mean": float(np.mean(ch_missing)),
                  "acc_allrows_mean": float(np.mean(acc))}
    json.dump(out, open(f"{ROOT}/results/audit_allrows_eval.json", "w"), indent=1)
    print("RESULT", json.dumps(out, indent=1))
    print(f"\ncanonical complete-case churn: V3_drop~0.0134, V3_impute~0.0108")


if __name__ == "__main__":
    main()
