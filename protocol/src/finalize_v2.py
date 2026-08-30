from __future__ import annotations

import argparse
import hashlib
import json
import os
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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preliminary", required=True)
    parser.add_argument("--cleanroom-comparison", required=True)
    parser.add_argument("--jsonld-diagnostic", required=True)
    parser.add_argument("--comparison-source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    preliminary_path = Path(args.preliminary)
    cleanroom_path = Path(args.cleanroom_comparison)
    diagnostic_path = Path(args.jsonld_diagnostic)
    comparison_source_path = Path(args.comparison_source)
    preliminary = json.loads(preliminary_path.read_text(encoding="utf-8"))
    cleanroom = json.loads(cleanroom_path.read_text(encoding="utf-8"))
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    criteria = list(preliminary["criteria"])
    criteria.extend(
        [
            {
                "criterion_id": "hash_bound_standard_jsonld_processor_pre_gate",
                "value": diagnostic.get("status") == "PASS" and diagnostic.get("input_sha256") == cleanroom["artifact_provenance"]["cases_jsonld_sha256"],
                "threshold": True,
                "passed": diagnostic.get("status") == "PASS" and diagnostic.get("input_sha256") == cleanroom["artifact_provenance"]["cases_jsonld_sha256"],
            },
            {
                "criterion_id": "cleanroom_carrier_validation_exact",
                "value": cleanroom["carrier_validation_exact"],
                "threshold": True,
                "passed": cleanroom["carrier_validation_exact"],
            },
            {
                "criterion_id": "cleanroom_exact_reproduction_v2",
                "value": cleanroom["exact_case_count"],
                "threshold": 18,
                "passed": cleanroom["all_exact"],
            },
        ]
    )
    all_pass = bool(preliminary["preliminary_criteria_passed"]) and all(item["passed"] for item in criteria)
    decision_label = "SUPPORTED_PROFILE_EXTENSION_V2" if all_pass else "ARTIFACT_VALIDATION_FAILED"
    status = "verified" if all_pass else "failed"
    unsafe_by_arm = {
        arm: {
            "unsafe_claim_acceptance_count": value["unsafe_claim_acceptance_count"],
            "unsafe_claim_cases": value["unsafe_claim_cases"],
        }
        for arm, value in preliminary["arms"].items()
    }
    result = {
        "status": status,
        "decision_label": decision_label,
        "all_required_criteria_passed": all_pass,
        "criteria": criteria,
        "class_support_counts": preliminary["class_support_counts"],
        "singleton_class_ids": preliminary["singleton_class_ids"],
        "ablation_changed_case_counts": preliminary["ablation_changed_case_counts"],
        "cleanroom_exact_case_count": cleanroom["exact_case_count"],
        "cleanroom_case_count": cleanroom["case_count"],
        "unsafe_claims_by_arm": unsafe_by_arm,
        "artifact_provenance": {
            "preliminary_summary_sha256": sha256(preliminary_path),
            "cleanroom_comparison_sha256": sha256(cleanroom_path),
            "comparison_source_sha256": sha256(comparison_source_path),
            "jsonld_processor_validation_sha256": sha256(diagnostic_path),
            **cleanroom["artifact_provenance"],
        },
        "safe_claim": (
            "Across 18 frozen comparison cases, the JSON-LD gate engine and an independent "
            "clean-room implementation reproduced all claim classes and field-evidence bindings "
            "exactly; gate-ablation utility is observed only on this case set."
        ),
        "interpretation_boundary": (
            "The result validates deterministic carrier parsing, field-to-artifact binding, and "
            "claim-class assignment for the frozen cases. It does not prove the empirical truth "
            "of manually adjudicated metadata, universal profile completeness, standard "
            "certification, or model superiority. Pairing-gate evidence and the own_test_only, "
            "lineage_only, and not_identifiable classes each have n=1 support."
        ),
    }
    atomic_json(Path(args.output), result)
    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
