from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--decisions-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    gold = json.loads(Path(args.gold).read_text(encoding="utf-8"))["labels"]
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    expected_allowed = {
        cid: set(config["classes"][expected]["allowed_claims"])
        for cid, expected in gold.items()
    }
    arms = {}
    for arm in config["arms"]:
        doc = json.loads((Path(args.decisions_dir) / f"{arm.lower()}.json").read_text(encoding="utf-8"))
        rows = []
        exact = 0
        unsafe_claims = []
        predicted_counts = Counter()
        closure_count = 0
        for decision in doc["decisions"]:
            cid = decision["comparison_id"]
            predicted = decision["decision_class"]
            expected = gold[cid]
            exact += int(predicted == expected)
            predicted_counts[str(predicted)] += 1
            allowed = {claim for claim, verdict in decision["claim_decisions"].items() if verdict == "allowed"}
            extras = sorted(allowed - expected_allowed[cid])
            if extras:
                unsafe_claims.append({"comparison_id": cid, "claims": extras})
            closure = decision["field_evidence_closure"]
            if closure["passed"]:
                closure_count += closure["valid_field_count"]
            rows.append(
                {
                    "comparison_id": cid,
                    "expected_class": expected,
                    "predicted_class": predicted,
                    "exact": predicted == expected,
                    "unsafe_allowed_claims": extras,
                    "field_evidence_closure_passed": closure["passed"],
                }
            )
        arms[arm] = {
            "exact_class_count": exact,
            "case_count": len(rows),
            "predicted_class_counts": dict(sorted(predicted_counts.items())),
            "unsafe_claim_acceptance_count": sum(len(item["claims"]) for item in unsafe_claims),
            "unsafe_claim_cases": unsafe_claims,
            "field_evidence_binding_count": closure_count,
            "carrier_validation": doc["carrier_validation"],
            "rows": rows,
        }

    full_counts = arms["FULL"]["predicted_class_counts"]
    criteria = [
        {"criterion_id": "jsonld_carrier_validation", "value": arms["FULL"]["carrier_validation"]["passed"], "threshold": True, "passed": arms["FULL"]["carrier_validation"]["passed"]},
        {"criterion_id": "field_evidence_closure", "value": arms["FULL"]["field_evidence_binding_count"], "threshold": 90, "passed": arms["FULL"]["field_evidence_binding_count"] == 90},
        {"criterion_id": "full_exact", "value": arms["FULL"]["exact_class_count"], "threshold": 18, "passed": arms["FULL"]["exact_class_count"] == 18},
        {"criterion_id": "full_all_classes_reachable", "value": len([key for key, value in full_counts.items() if key != "None" and value > 0]), "threshold": 5, "passed": len([key for key, value in full_counts.items() if key != "None" and value > 0]) == 5},
        {"criterion_id": "full_unsafe_claims", "value": arms["FULL"]["unsafe_claim_acceptance_count"], "threshold": 0, "passed": arms["FULL"]["unsafe_claim_acceptance_count"] == 0},
        {"criterion_id": "pairing_ablation_changes_class", "value": 18 - arms["NO_PAIRING_GATE"]["exact_class_count"], "threshold": 1, "passed": 18 - arms["NO_PAIRING_GATE"]["exact_class_count"] >= 1},
        {"criterion_id": "target_ablation_promotions", "value": 18 - arms["NO_TARGET_GATE"]["exact_class_count"], "threshold": 4, "passed": 18 - arms["NO_TARGET_GATE"]["exact_class_count"] == 4},
        {"criterion_id": "evidence_ablation_changes_class", "value": 18 - arms["NO_EVIDENCE_GATE"]["exact_class_count"], "threshold": 1, "passed": 18 - arms["NO_EVIDENCE_GATE"]["exact_class_count"] >= 1},
        {"criterion_id": "documentation_only_no_classes", "value": arms["DOCUMENTATION_ONLY"]["predicted_class_counts"].get("None", 0), "threshold": 18, "passed": arms["DOCUMENTATION_ONLY"]["predicted_class_counts"].get("None", 0) == 18},
        {"criterion_id": "provenance_only_no_classes", "value": arms["PROVENANCE_ONLY"]["predicted_class_counts"].get("None", 0), "threshold": 18, "passed": arms["PROVENANCE_ONLY"]["predicted_class_counts"].get("None", 0) == 18},
    ]
    singleton_classes = sorted([key for key, value in full_counts.items() if key != "None" and value == 1])
    ablation_evidence_n = {
        "NO_PAIRING_GATE": 18 - arms["NO_PAIRING_GATE"]["exact_class_count"],
        "NO_TARGET_GATE": 18 - arms["NO_TARGET_GATE"]["exact_class_count"],
        "NO_EVIDENCE_GATE": 18 - arms["NO_EVIDENCE_GATE"]["exact_class_count"],
    }
    result = {
        "status": "PRELIMINARY_V2_AWAITING_CLEANROOM",
        "preliminary_criteria_passed": all(item["passed"] for item in criteria),
        "criteria": criteria,
        "arms": arms,
        "singleton_class_ids": singleton_classes,
        "class_support_counts": full_counts,
        "ablation_changed_case_counts": ablation_evidence_n,
        "cleanroom_required_for_final_decision": True,
    }
    atomic_json(Path(args.output), result)
    return 0 if result["preliminary_criteria_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
