from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def validate_case(case: dict, config: dict, evidence_catalog: dict) -> None:
    extras = set(case) - set(config["required_fields"])
    missing = set(config["required_fields"]) - set(case)
    if missing or extras:
        raise ValueError(f"field contract failed; missing={sorted(missing)} extras={sorted(extras)}")
    for field, allowed in config["enums"].items():
        if case[field] not in allowed:
            raise ValueError(f"unsupported {field}: {case[field]}")
    if not isinstance(case["evidence_artifact_ids"], list) or not case["evidence_artifact_ids"]:
        raise ValueError("evidence_artifact_ids must be a non-empty list")
    unknown = set(case["evidence_artifact_ids"]) - set(evidence_catalog)
    if unknown:
        raise ValueError(f"unknown evidence ids: {sorted(unknown)}")


def full_class(case: dict) -> tuple[str, list[dict]]:
    trace = []
    provenance_ok = case["provenance_status"] == "verified"
    trace.append({"gate": "provenance", "passed": provenance_ok, "value": case["provenance_status"]})
    if not provenance_ok:
        return "not_identifiable", trace

    execution = case["behavioral_execution"]
    trace.append({"gate": "executed_evidence", "passed": execution == "verified", "value": execution})
    if execution == "unknown":
        return "not_identifiable", trace
    if execution == "absent":
        lineage_fact_ok = case["target_semantics"] == "equivalent" and case["row_correspondence"] == "verified"
        trace.append({"gate": "lineage_fact_only", "passed": lineage_fact_ok, "value": "identity_without_execution"})
        return ("lineage_only" if lineage_fact_ok else "not_identifiable"), trace

    same_rows = case["row_correspondence"] == "verified"
    same_eval = case["evaluation_alignment"] == "same_frozen_records"
    trace.append({"gate": "pairing", "passed": same_rows and same_eval, "value": f"{case['row_correspondence']}|{case['evaluation_alignment']}"})
    if same_rows and same_eval:
        target = case["target_semantics"]
        trace.append({"gate": "target_semantics", "passed": target in {"equivalent", "different"}, "value": target})
        if target == "equivalent":
            return "paired_behavior", trace
        if target == "different":
            return "target_sensitivity", trace
        return "not_identifiable", trace

    own_test = case["row_correspondence"] == "absent" and case["evaluation_alignment"] == "own_test_splits"
    target_ok = case["target_semantics"] == "equivalent"
    trace.append({"gate": "own_test_boundary", "passed": own_test and target_ok, "value": f"{case['row_correspondence']}|{case['evaluation_alignment']}|{case['target_semantics']}"})
    if own_test and target_ok:
        return "own_test_only", trace
    return "not_identifiable", trace


def classify(case: dict, arm: str) -> tuple[str | None, list[dict]]:
    if arm in {"DOCUMENTATION_ONLY", "PROVENANCE_ONLY"}:
        return None, [{"gate": "comparison_decision", "passed": False, "value": "not_provided_by_baseline"}]
    if arm == "FULL":
        return full_class(case)
    if arm == "NO_PAIRING_GATE":
        if case["provenance_status"] != "verified":
            return "not_identifiable", [{"gate": "provenance", "passed": False, "value": case["provenance_status"]}]
        if case["behavioral_execution"] == "absent":
            return "lineage_only", [{"gate": "executed_evidence", "passed": False, "value": "absent"}]
        if case["behavioral_execution"] != "verified" or case["target_semantics"] == "unknown":
            return "not_identifiable", [{"gate": "remaining_evidence", "passed": False, "value": "incomplete"}]
        result = "paired_behavior" if case["target_semantics"] == "equivalent" else "target_sensitivity"
        return result, [{"gate": "pairing", "passed": True, "value": "ablated"}]
    if arm == "NO_TARGET_GATE":
        result, trace = full_class(case)
        if result == "target_sensitivity":
            trace.append({"gate": "target_semantics", "passed": True, "value": "different_treated_as_equivalent"})
            return "paired_behavior", trace
        return result, trace
    if arm == "NO_EVIDENCE_GATE":
        if (
            case["provenance_status"] == "verified"
            and case["row_correspondence"] == "verified"
            and case["evaluation_alignment"] == "same_frozen_records"
            and case["target_semantics"] in {"equivalent", "different"}
        ):
            result = "paired_behavior" if case["target_semantics"] == "equivalent" else "target_sensitivity"
            return result, [{"gate": "executed_evidence", "passed": True, "value": "ablated"}]
        return full_class(case)
    raise ValueError(f"unknown arm: {arm}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cases_doc = json.loads(Path(args.cases).read_text(encoding="utf-8"))
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    arm = args.arm
    if arm not in config["arms"]:
        raise SystemExit(f"arm not declared: {arm}")

    decisions = []
    catalog = cases_doc["evidence_catalog"]
    claims = cases_doc["metadata"]["claim_catalog"]
    for case in cases_doc["cases"]:
        validate_case(case, config, catalog)
        decision_class, trace = classify(case, arm)
        allowed = set(config["classes"].get(decision_class, {}).get("allowed_claims", []))
        claim_decisions = {claim: ("allowed" if claim in allowed else "forbidden") for claim in claims}
        decisions.append(
            {
                "comparison_id": case["comparison_id"],
                "decision_class": decision_class,
                "gate_trace": trace,
                "claim_decisions": claim_decisions,
                "evidence_artifact_ids": case["evidence_artifact_ids"],
            }
        )

    output = {
        "arm": arm,
        "case_count": len(decisions),
        "input_sha256": sha256(Path(args.cases)),
        "config_sha256": sha256(Path(args.config)),
        "decisions": decisions,
    }
    atomic_json(Path(args.output), output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
