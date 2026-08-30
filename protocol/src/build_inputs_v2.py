from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


FIELD_KEYS = [
    "comparison_id",
    "dataset_family",
    "left_state",
    "right_state",
    "provenance_status",
    "row_correspondence",
    "target_semantics",
    "evaluation_alignment",
    "behavioral_execution",
    "evidence_artifact_ids",
    "field_evidence",
]
TERM = {
    "comparison_id": "bla:comparisonId",
    "dataset_family": "bla:datasetFamily",
    "left_state": "bla:leftState",
    "right_state": "bla:rightState",
    "provenance_status": "bla:provenanceStatus",
    "row_correspondence": "bla:rowCorrespondence",
    "target_semantics": "bla:targetSemantics",
    "evaluation_alignment": "bla:evaluationAlignment",
    "behavioral_execution": "bla:behavioralExecution",
}
FIELD_EQUIVALENT = {
    **TERM,
    "evidence_artifact_ids": "bla:evidenceArtifactIds",
    "field_evidence": "bla:fieldEvidence",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def evidence_urn(artifact_id: str) -> str:
    return f"urn:behavioral-lineage-audit:evidence:{artifact_id}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gold", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    root = Path(args.project_root).resolve()
    profile_path = (root / args.profile).resolve()
    cases_path = (root / args.cases).resolve()
    source = json.loads(cases_path.read_text(encoding="utf-8"))

    for artifact_id, artifact in source["evidence_catalog"].items():
        evidence_path = (root / artifact["path"]).resolve()
        actual = sha256(evidence_path)
        if actual != artifact["sha256"]:
            raise ValueError(f"evidence hash mismatch: {artifact_id}")
        if not artifact.get("supports_fields"):
            raise ValueError(f"supports_fields missing: {artifact_id}")

    labels = {}
    sanitized_cases = []
    for case in source["cases"]:
        row = dict(case)
        labels[row["comparison_id"]] = row.pop("expected_class")
        sanitized_cases.append(row)

    evidence_nodes = []
    for artifact_id, artifact in source["evidence_catalog"].items():
        evidence_nodes.append(
            {
                "@id": evidence_urn(artifact_id),
                "@type": ["prov:Entity", "bla:EvidenceArtifact"],
                "bla:artifactId": artifact_id,
                "sc:contentUrl": artifact["path"],
                "bla:sha256": artifact["sha256"],
                "bla:supportsField": artifact["supports_fields"],
            }
        )

    case_nodes = []
    for case in sanitized_cases:
        node = {
            "@id": f"urn:behavioral-lineage-audit:case:{case['comparison_id']}",
            "@type": ["bla:ComparisonCase", "prov:Entity"],
            **{TERM[key]: case[key] for key in TERM},
            "bla:evidenceArtifactIds": case["evidence_artifact_ids"],
            "prov:wasDerivedFrom": [{"@id": evidence_urn(item)} for item in case["evidence_artifact_ids"]],
            "bla:fieldEvidence": [
                {
                    "@type": "bla:FieldEvidenceBinding",
                    "bla:field": field,
                    "prov:wasDerivedFrom": [{"@id": evidence_urn(item)} for item in ids],
                }
                for field, ids in case["field_evidence"].items()
            ],
        }
        case_nodes.append(node)

    field_nodes = []
    for key in FIELD_KEYS:
        equivalent = FIELD_EQUIVALENT[key]
        field_nodes.append(
            {
                "@id": f"comparison_cases/{key}",
                "@type": "cr:Field",
                "name": key,
                "cr:dataType": "sc:Text",
                "cr:equivalentProperty": equivalent,
            }
        )

    doc = {
        "@context": {
            "@vocab": "http://schema.org/",
            "sc": "http://schema.org/",
            "cr": "http://mlcommons.org/croissant/",
            "prov": "http://www.w3.org/ns/prov#",
            "dct": "http://purl.org/dc/terms/",
            "bla": "urn:behavioral-lineage-audit:",
        },
        "@id": "urn:behavioral-lineage-audit:dataset:comparison-cases-v2",
        "@type": ["sc:Dataset", "prov:Entity"],
        "name": "Behavioral dataset-lineage comparison cases v2",
        "dct:conformsTo": [
            "http://mlcommons.org/croissant/1.1",
            "urn:behavioral-lineage-audit:comparison-profile:2.0.0-candidate",
        ],
        "prov:wasGeneratedBy": {
            "@id": "urn:behavioral-lineage-audit:activity:v2-input-build-20260826",
            "@type": "prov:Activity",
            "prov:hadPlan": {
                "@id": "urn:behavioral-lineage-audit:plan:v2-validation",
                "@type": "prov:Plan",
            },
        },
        "cr:recordSet": [
            {
                "@id": "comparison_cases",
                "@type": "cr:RecordSet",
                "name": "comparison_cases",
                "cr:key": {"@id": "comparison_cases/comparison_id"},
                "cr:field": field_nodes,
            }
        ],
        "bla:claimCatalog": source["metadata"]["claim_catalog"],
        "bla:evidenceCatalog": evidence_nodes,
        "bla:case": case_nodes,
        "bla:caseCount": len(case_nodes),
    }

    output_path = Path(args.output)
    gold_path = Path(args.gold)
    manifest_path = Path(args.manifest)
    atomic_json(output_path, doc)
    atomic_json(gold_path, {"labels": labels})
    atomic_json(
        manifest_path,
        {
            "profile_path": args.profile,
            "profile_sha256": sha256(profile_path),
            "cases_path": args.cases,
            "cases_sha256": sha256(cases_path),
            "sanitized_jsonld_path": output_path.as_posix(),
            "sanitized_jsonld_sha256": sha256(output_path),
            "gold_path": gold_path.as_posix(),
            "gold_sha256": sha256(gold_path),
            "case_count": len(case_nodes),
            "field_binding_count": sum(len(case["field_evidence"]) for case in sanitized_cases),
            "expected_class_present_in_sanitized": "expected_class" in output_path.read_text(encoding="utf-8"),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
