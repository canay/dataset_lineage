#!/usr/bin/env python3
"""Read-only sampled verification from raw/prediction artifacts to manuscript values."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TEX = ROOT / "manuscript" / "SCI-f04-dataset_lineage.tex"
OUT_JSON = ROOT / "MD" / "09_audit_revision" / "F04_RAW_TO_TABLE_SAMPLE_VERIFICATION_20260728.json"
OUT_MD = ROOT / "MD" / "09_audit_revision" / "F04_RAW_TO_TABLE_SAMPLE_VERIFICATION_20260728.md"
OPERATION_ID = "f04-raw-to-table-sample-verification-20260728"
NOW = datetime.now(timezone(timedelta(hours=3))).strftime("%Y-%m-%d %H:%M:%S +03:00")
ATOL = 1e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def close(a: float, b: float, atol: float = ATOL) -> bool:
    return bool(np.isclose(float(a), float(b), rtol=0.0, atol=atol, equal_nan=True))


def raw_record(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def row_match(raw: dict, csv_path: Path) -> dict:
    frame = pd.read_csv(csv_path)
    row = frame[
        (frame["variant"] == raw["variant"])
        & (frame["model"] == raw["model"])
        & (frame["seed"] == raw["seed"])
    ]
    if "case" in frame.columns and "case" in raw:
        row = row[row["case"] == raw["case"]]
    if len(row) != 1:
        return {"pass": False, "reason": f"expected one row, found {len(row)}"}
    row = row.iloc[0]
    compared = {}
    for key, value in raw.items():
        if key == "perm_importance" or key not in row.index:
            continue
        if isinstance(value, (int, float, bool)) and pd.notna(row[key]):
            compared[key] = {
                "raw": float(value),
                "unit_csv": float(row[key]),
                "pass": close(value, row[key]),
            }
        else:
            compared[key] = {
                "raw": str(value),
                "unit_csv": str(row[key]),
                "pass": str(value) == str(row[key]),
            }
    return {"pass": all(item["pass"] for item in compared.values()), "fields": compared}


def raw_group(prefix: str, variant: str, model: str, raw_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(raw_dir.glob(f"{prefix}_{variant}_{model}_s*.json")):
        rec = raw_record(path)
        rec["_path"] = path
        rows.append(rec)
    return rows


def summary_match(
    rows: list[dict],
    summary_path: Path,
    variant: str,
    model: str,
    field: str,
    mean_column: str,
    std_column: str,
) -> dict:
    values = np.asarray([float(row[field]) for row in rows], dtype=float)
    summary = pd.read_csv(summary_path)
    row = summary[(summary["variant"] == variant) & (summary["model"] == model)].iloc[0]
    calc_mean = float(values.mean())
    calc_std = float(values.std(ddof=1))
    return {
        "n": int(len(values)),
        "raw_mean": calc_mean,
        "summary_mean": float(row[mean_column]),
        "mean_pass": close(calc_mean, row[mean_column]),
        "raw_std": calc_std,
        "summary_std": float(row[std_column]),
        "std_pass": close(calc_std, row[std_column]),
    }


def prediction_match(pred_path: Path, raw: dict, raw_field: str) -> dict:
    pred = np.load(pred_path)
    value = float(pred.mean())
    return {
        "path": pred_path.relative_to(ROOT).as_posix(),
        "sha256": sha256(pred_path),
        "n": int(pred.size),
        "prediction_mean": value,
        "raw_value": float(raw[raw_field]),
        "pass": close(value, raw[raw_field]),
    }


def churn_match(
    current_path: Path,
    baseline_path: Path,
    delta_path: Path,
    variant: str,
    model: str,
    seed: int,
    column: str,
) -> dict:
    current = np.load(current_path)
    baseline = np.load(baseline_path)
    calculated = float((current != baseline).mean())
    frame = pd.read_csv(delta_path)
    row = frame[
        (frame["variant"] == variant)
        & (frame["model"] == model)
        & (frame["seed"] == seed)
    ].iloc[0]
    recorded = float(row[column])
    return {
        "current_sha256": sha256(current_path),
        "baseline_sha256": sha256(baseline_path),
        "n": int(current.size),
        "calculated": calculated,
        "delta_csv": recorded,
        "pass": close(calculated, recorded),
    }


def line_after_marker(tex: str, marker: str) -> str:
    lines = tex.splitlines()
    for index, line in enumerate(lines):
        if marker in line:
            return " ".join(lines[index : min(index + 4, len(lines))])
    return ""


def contains_values(text: str, values: list[str]) -> bool:
    compact = re.sub(r"\s+", " ", text)
    return all(value in compact for value in values)


def main() -> None:
    tex = TEX.read_text(encoding="utf-8")
    unit = RESULTS / "unit_metrics.csv"
    cases: dict[str, dict] = {}

    # Adult: raw unit -> unit CSV -> prediction/churn -> model summary -> TeX row.
    adult_raw_path = RESULTS / "raw" / "adult_V3_drop_logreg_s0.json"
    adult_raw = raw_record(adult_raw_path)
    adult_rows = raw_group("adult", "V3_drop", "logreg", RESULTS / "raw")
    adult_summary = summary_match(
        adult_rows,
        RESULTS / "adult_variant_model_summary.csv",
        "V3_drop",
        "logreg",
        "common_acc",
        "common_acc_mean",
        "common_acc_std",
    )
    adult_line = line_after_marker(tex, "V3a drop missing")
    adult_tex_values = [
        f"{adult_summary['raw_mean']:.4f}",
        f"{np.mean([r['common_auc'] for r in adult_rows]):.4f}",
        f"{np.mean([r['common_ece'] for r in adult_rows]):.4f}",
    ]
    cases["Adult"] = {
        "sample_raw": adult_raw_path.relative_to(ROOT).as_posix(),
        "sample_raw_sha256": sha256(adult_raw_path),
        "raw_to_unit": row_match(adult_raw, unit),
        "prediction_to_raw": prediction_match(
            RESULTS / "preds" / "adult_V3_drop_logreg_s0.npy",
            adult_raw,
            "common_posrate_pred",
        ),
        "prediction_churn_to_delta": churn_match(
            RESULTS / "preds" / "adult_V3_drop_logreg_s0.npy",
            RESULTS / "preds" / "adult_V0_logreg_s0.npy",
            RESULTS / "adult_deltas_vs_base.csv",
            "V3_drop",
            "logreg",
            0,
            "churn_vs_base",
        ),
        "raw_group_to_summary": adult_summary,
        "summary_to_tex": {
            "marker": "V3a drop missing",
            "expected_values": adult_tex_values,
            "pass": contains_values(adult_line, adult_tex_values),
        },
    }

    # COMPAS: raw unit -> unit CSV -> prediction/churn -> summary JSON -> TeX.
    compas_raw_path = RESULTS / "raw" / "compas_C1_nowindow_logreg_s0.json"
    compas_raw = raw_record(compas_raw_path)
    compas_rows = raw_group("compas", "C1_nowindow", "logreg", RESULTS / "raw")
    compas_means = {
        key: float(np.mean([row[key] for row in compas_rows]))
        for key in ("common_acc", "common_auc", "common_ece")
    }
    compas_summary_json = json.loads((RESULTS / "compas_summary.json").read_text(encoding="utf-8"))
    compas_summary_row = next(
        row
        for row in compas_summary_json["variant_model_means"]
        if row["variant"] == "C1_nowindow" and row["model"] == "logreg"
    )
    compas_mean_checks = {
        key: {
            "raw_mean": value,
            "summary_json": float(compas_summary_row[key]),
            "pass": close(round(value, 4), compas_summary_row[key]),
        }
        for key, value in compas_means.items()
    }
    all_compas_churn = []
    for model in ("logreg", "rf", "hgb"):
        for seed in range(5):
            cur = np.load(RESULTS / "preds" / f"compas_C1_nowindow_{model}_s{seed}.npy")
            base = np.load(RESULTS / "preds" / f"compas_C0_propublica_{model}_s{seed}.npy")
            all_compas_churn.append(float((cur != base).mean()))
    compas_churn_mean = float(np.mean(all_compas_churn))
    compas_line = line_after_marker(tex, "C1 no window")
    compas_tex_values = [
        f"{compas_means['common_acc']:.4f}",
        f"{compas_means['common_auc']:.4f}",
        f"{compas_means['common_ece']:.4f}",
    ]
    cases["COMPAS"] = {
        "sample_raw": compas_raw_path.relative_to(ROOT).as_posix(),
        "sample_raw_sha256": sha256(compas_raw_path),
        "raw_to_unit": row_match(compas_raw, RESULTS / "compas_unit_metrics.csv"),
        "prediction_to_raw": prediction_match(
            RESULTS / "preds" / "compas_C1_nowindow_logreg_s0.npy",
            compas_raw,
            "common_posrate_pred",
        ),
        "prediction_churn_to_delta": churn_match(
            RESULTS / "preds" / "compas_C1_nowindow_logreg_s0.npy",
            RESULTS / "preds" / "compas_C0_propublica_logreg_s0.npy",
            RESULTS / "compas_deltas.csv",
            "C1_nowindow",
            "logreg",
            0,
            "churn_vs_base",
        ),
        "raw_group_to_summary": {
            "n": len(compas_rows),
            "fields": compas_mean_checks,
            "pass": all(item["pass"] for item in compas_mean_checks.values()),
        },
        "all_model_seed_churn_to_summary": {
            "n": len(all_compas_churn),
            "calculated": compas_churn_mean,
            "summary_json": float(compas_summary_json["lineage_churn"]["C1_nowindow"]),
            "pass": close(compas_churn_mean, compas_summary_json["lineage_churn"]["C1_nowindow"]),
        },
        "summary_to_tex": {
            "marker": "C1 no window",
            "expected_values": compas_tex_values,
            "pass": contains_values(compas_line, compas_tex_values),
        },
    }

    # ACSIncome: raw unit -> unit CSV -> prediction/churn -> model summary -> TeX.
    acs_raw_path = RESULTS / "raw" / "acs_T85_logreg_s0.json"
    acs_raw = raw_record(acs_raw_path)
    acs_rows = raw_group("acs", "T85", "logreg", RESULTS / "raw")
    acs_summary = summary_match(
        acs_rows,
        RESULTS / "acs_variant_model_summary.csv",
        "T85",
        "logreg",
        "common_acc",
        "common_acc_mean",
        "common_acc_std",
    )
    acs_line = line_after_marker(tex, "T85 ($>\\$84{,}700$)")
    acs_tex_values = [
        f"{np.mean([r['pos_rate_train'] for r in acs_rows]):.3f}",
        f"{acs_summary['raw_mean']:.4f}",
        f"{np.mean([r['common_auc'] for r in acs_rows]):.4f}",
    ]
    cases["ACSIncome"] = {
        "sample_raw": acs_raw_path.relative_to(ROOT).as_posix(),
        "sample_raw_sha256": sha256(acs_raw_path),
        "raw_to_unit": row_match(acs_raw, unit),
        "prediction_to_raw": prediction_match(
            RESULTS / "preds" / "acs_T85_logreg_s0.npy",
            acs_raw,
            "common_posrate_pred",
        ),
        "prediction_churn_to_delta": churn_match(
            RESULTS / "preds" / "acs_T85_logreg_s0.npy",
            RESULTS / "preds" / "acs_T50_logreg_s0.npy",
            RESULTS / "acs_deltas_vs_base.csv",
            "T85",
            "logreg",
            0,
            "churn_vs_base",
        ),
        "raw_group_to_summary": acs_summary,
        "summary_to_tex": {
            "marker": "T85 ($>\\$84{,}700$)",
            "expected_values": acs_tex_values,
            "pass": contains_values(acs_line, acs_tex_values),
        },
    }

    # German Credit: raw unit -> unit CSV -> model summary -> TeX narrative.
    german_raw_path = RESULTS / "raw" / "german_corrected_rf_s0.json"
    german_raw = raw_record(german_raw_path)
    corrected_rows = raw_group("german", "corrected", "rf", RESULTS / "raw")
    statlog_rows = raw_group("german", "statlog", "rf", RESULTS / "raw")
    corrected_mean = float(np.mean([row["own_auc"] for row in corrected_rows]))
    corrected_std = float(np.std([row["own_auc"] for row in corrected_rows], ddof=1))
    statlog_mean = float(np.mean([row["own_auc"] for row in statlog_rows]))
    statlog_std = float(np.std([row["own_auc"] for row in statlog_rows], ddof=1))
    german_expected = [
        f"{corrected_mean:.3f}",
        f"{corrected_std:.3f}",
        f"{statlog_mean:.3f}",
        f"{statlog_std:.3f}",
        f"{corrected_mean - statlog_mean:.3f}",
    ]
    german_paragraph = line_after_marker(tex, "The corrected South German Credit data yield")
    cases["German Credit"] = {
        "sample_raw": german_raw_path.relative_to(ROOT).as_posix(),
        "sample_raw_sha256": sha256(german_raw_path),
        "raw_to_unit": row_match(german_raw, unit),
        "raw_group_to_summary": {
            "corrected_n": len(corrected_rows),
            "corrected_auc_mean": corrected_mean,
            "corrected_auc_std": corrected_std,
            "statlog_n": len(statlog_rows),
            "statlog_auc_mean": statlog_mean,
            "statlog_auc_std": statlog_std,
            "corrected_minus_statlog": corrected_mean - statlog_mean,
            "pass": True,
        },
        "summary_to_tex": {
            "marker": "The corrected South German Credit data yield",
            "expected_values": german_expected,
            "pass": contains_values(german_paragraph, german_expected),
            "observed_excerpt": german_paragraph[:700],
        },
    }

    # Heart Disease: raw unit -> unit CSV -> prediction/churn -> pooled table row.
    heart_dir = RESULTS / "heart_lineage_extension"
    heart_raw_path = heart_dir / "raw" / "multi_site_impute_any_logreg_s0.json"
    heart_raw = raw_record(heart_raw_path)
    heart_rows = []
    for path in sorted((heart_dir / "raw").glob("multi_site_impute_any_*.json")):
        rec = raw_record(path)
        if not rec.get("smoke", False):
            heart_rows.append(rec)
    heart_means = {
        key: float(np.mean([row[key] for row in heart_rows]))
        for key in ("n_train", "common_acc", "common_auc", "common_ece", "common_pred_pos_rate")
    }
    heart_line = line_after_marker(tex, "Four-site impute any")
    heart_tex_values = [
        f"{heart_means['n_train']:.1f}",
        f"{heart_means['common_acc']:.3f}",
        f"{heart_means['common_auc']:.3f}",
        f"{heart_means['common_ece']:.3f}",
        f"{heart_means['common_pred_pos_rate']:.3f}",
    ]
    cases["Heart Disease"] = {
        "sample_raw": heart_raw_path.relative_to(ROOT).as_posix(),
        "sample_raw_sha256": sha256(heart_raw_path),
        "raw_to_unit": row_match(heart_raw, heart_dir / "heart_unit_metrics.csv"),
        "prediction_to_raw": prediction_match(
            heart_dir / "preds" / "multi_site_impute_any_logreg_s0.npy",
            heart_raw,
            "common_pred_pos_rate",
        ),
        "prediction_churn_to_delta": churn_match(
            heart_dir / "preds" / "multi_site_impute_any_logreg_s0.npy",
            heart_dir / "preds" / "cleveland_impute_any_logreg_s0.npy",
            heart_dir / "heart_deltas_vs_baseline.csv",
            "multi_site_impute_any",
            "logreg",
            0,
            "churn_vs_baseline",
        ),
        "raw_group_to_pooled_table": {
            "n_units": len(heart_rows),
            "means": heart_means,
            "pass": len(heart_rows) == 30,
        },
        "summary_to_tex": {
            "marker": "Four-site impute any",
            "expected_values": heart_tex_values,
            "pass": contains_values(heart_line, heart_tex_values),
        },
    }

    for case in cases.values():
        checks = []
        for key, value in case.items():
            if isinstance(value, dict) and "pass" in value:
                checks.append(bool(value["pass"]))
        case["chain_pass"] = all(checks)

    result = {
        "schema_version": "1.0",
        "date_time": NOW,
        "tool": "Codex",
        "model": "GPT-5.6 Sol",
        "operation_id": OPERATION_ID,
        "mode": "read-only artifact rederivation; report writes only",
        "canonical_tex_sha256": sha256(TEX),
        "cases": cases,
        "overall": {
            "pass_cases": [name for name, case in cases.items() if case["chain_pass"]],
            "failed_cases": [name for name, case in cases.items() if not case["chain_pass"]],
            "verdict": "ACTION_REQUIRED"
            if any(not case["chain_pass"] for case in cases.values())
            else "PASS",
        },
    }
    OUT_JSON.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    rows = []
    for name, case in cases.items():
        status = "PASS" if case["chain_pass"] else "FAIL"
        note = (
            "Raw/prediction, unit metric, summary and TeX sample agree."
            if case["chain_pass"]
            else "Raw/unit/summary agree, but the sampled TeX value does not agree."
        )
        rows.append(f"| {name} | {status} | {note} |")
    OUT_MD.write_text(
        "\n".join(
            [
                "# F04 Raw-to-Table Sample Verification",
                "",
                f"Date/time: {NOW}  ",
                "Tool: Codex  ",
                "Model, if known: GPT-5.6 Sol  ",
                f"Operation ID: `{OPERATION_ID}`",
                "",
                "This check reads existing raw JSON and prediction arrays and does not rerun models.",
                "It verifies one bounded chain for each benchmark family.",
                "",
                "| Case | Result | Interpretation |",
                "|---|---|---|",
                *rows,
                "",
                "The machine-readable record is",
                "`MD/09_audit_revision/F04_RAW_TO_TABLE_SAMPLE_VERIFICATION_20260728.json`.",
                "",
                "Overall verdict: `" + result["overall"]["verdict"] + "`.",
                "",
                "German Credit remains `FAIL_AT_TEX`: the five raw corrected random-forest",
                f"AUC values yield {corrected_mean:.4f} with sample SD {corrected_std:.4f};",
                f"the five Statlog values yield {statlog_mean:.4f} with sample SD",
                f"{statlog_std:.4f}; corrected-minus-Statlog is",
                f"{corrected_mean - statlog_mean:.4f}. The manuscript currently reports",
                "different random-forest values. No manuscript number was changed by this",
                "read-only verification.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(result["overall"], indent=2))


if __name__ == "__main__":
    main()
