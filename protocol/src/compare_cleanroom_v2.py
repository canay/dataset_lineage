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
    parser.add_argument("--main", required=True)
    parser.add_argument("--cleanroom", required=True)
    parser.add_argument("--cases-jsonld", required=True)
    parser.add_argument("--cleanroom-source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    main_path = Path(args.main)
    cleanroom_path = Path(args.cleanroom)
    cases_path = Path(args.cases_jsonld)
    cleanroom_source_path = Path(args.cleanroom_source)
    main_doc = json.loads(main_path.read_text(encoding="utf-8"))
    clean_doc = json.loads(cleanroom_path.read_text(encoding="utf-8"))
    cases_hash = sha256(cases_path)
    if main_doc.get("input_sha256") != cases_hash:
        raise ValueError("main output is not bound to the compared JSON-LD input")
    main_rows = {row["comparison_id"]: row for row in main_doc["decisions"]}
    clean_rows = {row["comparison_id"]: row for row in clean_doc["decisions"]}
    ids_match = set(main_rows) == set(clean_rows)
    carrier_exact = main_doc["carrier_validation"] == clean_doc["carrier_validation"]
    rows = []
    for comparison_id in sorted(set(main_rows) | set(clean_rows)):
        left = main_rows.get(comparison_id)
        right = clean_rows.get(comparison_id)
        checks = {
            "present_both": left is not None and right is not None,
            "decision_class_exact": left is not None and right is not None and left["decision_class"] == right["decision_class"],
            "gate_trace_exact": left is not None and right is not None and left["gate_trace"] == right["gate_trace"],
            "claim_decisions_exact": left is not None and right is not None and left["claim_decisions"] == right["claim_decisions"],
            "evidence_ids_exact": left is not None and right is not None and left["evidence_artifact_ids"] == right["evidence_artifact_ids"],
            "field_evidence_closure_exact": left is not None and right is not None and left["field_evidence_closure"] == right["field_evidence_closure"],
        }
        rows.append({"comparison_id": comparison_id, "checks": checks, "passed": all(checks.values())})
    exact_count = sum(row["passed"] for row in rows)
    report = {
        "ids_exact": ids_match,
        "carrier_validation_exact": carrier_exact,
        "case_count": len(rows),
        "exact_case_count": exact_count,
        "all_exact": ids_match and carrier_exact and exact_count == len(rows) == 18,
        "artifact_provenance": {
            "cases_jsonld_sha256": cases_hash,
            "main_output_sha256": sha256(main_path),
            "cleanroom_output_sha256": sha256(cleanroom_path),
            "cleanroom_source_sha256": sha256(cleanroom_source_path),
        },
        "rows": rows,
    }
    atomic_json(Path(args.output), report)
    return 0 if report["all_exact"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
