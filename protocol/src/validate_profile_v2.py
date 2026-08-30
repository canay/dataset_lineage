from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
from pathlib import Path


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
SHA256_RE = re.compile(r"[0-9A-F]{64}")
FROZEN_V1_CLASSIFIER_SHA256 = "60CED9C7B8C3131D59D106E059E76AFC3913392B499790D9AE232A6A809ECEF2"


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


def reject_gold_fields(value: object, path: str = "root") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"expected_class", "bla:expectedClass", "gold_label", "bla:goldLabel"}:
                raise ValueError(f"forbidden gold field at {path}.{key}")
            reject_gold_fields(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            reject_gold_fields(item, f"{path}[{index}]")


def load_v1_classifier() -> object:
    source = Path(__file__).resolve().parents[2] / "2026-08-26_codex_local_comparison_admissibility_validation" / "src" / "validate_profile.py"
    actual_hash = sha256(source)
    if actual_hash != FROZEN_V1_CLASSIFIER_SHA256:
        raise RuntimeError(
            f"frozen v1 classifier hash mismatch: {actual_hash} != {FROZEN_V1_CLASSIFIER_SHA256}"
        )
    spec = importlib.util.spec_from_file_location("f04_v1_classifier", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen v1 classifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evidence_id_from_urn(value: str) -> str:
    prefix = "urn:behavioral-lineage-audit:evidence:"
    if not value.startswith(prefix):
        raise ValueError(f"unexpected evidence URI: {value}")
    return value[len(prefix):]


def validate_and_extract(doc: dict, config: dict) -> tuple[list[dict], dict, list[str], dict]:
    reject_gold_fields(doc)
    expected_context = config["jsonld_contract"]["context"]
    context = doc.get("@context")
    if not isinstance(context, dict):
        raise ValueError("@context must be an object")
    if set(context) != set(expected_context) | {"@vocab"}:
        raise ValueError("JSON-LD @context must contain exactly the six frozen bindings")
    for prefix, uri in expected_context.items():
        if context.get(prefix) != uri:
            raise ValueError(f"JSON-LD namespace mismatch: {prefix}")
    if context.get("@vocab") != expected_context["sc"]:
        raise ValueError("JSON-LD @vocab must be schema.org")
    if set(doc.get("@type", [])) != set(config["jsonld_contract"]["required_root_types"]):
        raise ValueError("root JSON-LD types do not match the profile")
    conforms = set(doc.get("dct:conformsTo", []))
    if "http://mlcommons.org/croissant/1.1" not in conforms:
        raise ValueError("Croissant 1.1 conformance target is absent")

    record_sets = doc.get("cr:recordSet")
    if not isinstance(record_sets, list) or len(record_sets) != 1:
        raise ValueError("exactly one cr:RecordSet is required")
    record_set = record_sets[0]
    if record_set.get("@type") != "cr:RecordSet":
        raise ValueError("record set type is not cr:RecordSet")
    if record_set.get("cr:key", {}).get("@id") != "comparison_cases/comparison_id":
        raise ValueError("record set key drift")
    field_nodes = record_set.get("cr:field")
    if not isinstance(field_nodes, list):
        raise ValueError("cr:field must be an array")
    field_names = [item.get("name") for item in field_nodes]
    if set(field_names) != set(config["required_fields"]):
        raise ValueError("Croissant field set does not match the executable contract")
    for item in field_nodes:
        if item.get("@type") != "cr:Field" or item.get("cr:equivalentProperty") != FIELD_EQUIVALENT.get(item.get("name")):
            raise ValueError(f"invalid Croissant Field: {item.get('name')}")

    evidence_catalog = {}
    for node in doc.get("bla:evidenceCatalog", []):
        types = set(node.get("@type", []))
        if not {"prov:Entity", "bla:EvidenceArtifact"}.issubset(types):
            raise ValueError("evidence node lacks required JSON-LD types")
        artifact_id = node.get("bla:artifactId")
        if not isinstance(artifact_id, str) or artifact_id in evidence_catalog:
            raise ValueError("duplicate or invalid evidence artifact ID")
        if evidence_id_from_urn(node.get("@id", "")) != artifact_id:
            raise ValueError(f"evidence URI/ID mismatch: {artifact_id}")
        digest = node.get("bla:sha256")
        if not isinstance(digest, str) or not SHA256_RE.fullmatch(digest):
            raise ValueError(f"invalid evidence SHA-256: {artifact_id}")
        supports = node.get("bla:supportsField")
        if not isinstance(supports, list) or not supports:
            raise ValueError(f"missing field support declaration: {artifact_id}")
        evidence_catalog[artifact_id] = {
            "path": node.get("sc:contentUrl"),
            "sha256": digest,
            "supports_fields": supports,
        }

    claims = doc.get("bla:claimCatalog")
    if not isinstance(claims, list) or not claims:
        raise ValueError("claim catalog is absent")
    cases = []
    binding_count = 0
    seen_ids = set()
    for node in doc.get("bla:case", []):
        if set(node.get("@type", [])) != set(config["jsonld_contract"]["required_case_types"]):
            raise ValueError("case JSON-LD types do not match the profile")
        case = {key: node.get(term) for key, term in TERM.items()}
        comparison_id = case["comparison_id"]
        if not isinstance(comparison_id, str) or comparison_id in seen_ids:
            raise ValueError("duplicate or invalid comparison ID")
        seen_ids.add(comparison_id)
        declared_ids = node.get("bla:evidenceArtifactIds")
        derived_ids = [evidence_id_from_urn(item.get("@id", "")) for item in node.get("prov:wasDerivedFrom", [])]
        if not isinstance(declared_ids, list) or declared_ids != derived_ids:
            raise ValueError(f"case/evidence derivation mismatch: {comparison_id}")
        unknown_ids = set(declared_ids) - set(evidence_catalog)
        if unknown_ids:
            raise ValueError(f"unknown evidence IDs for {comparison_id}: {sorted(unknown_ids)}")

        bindings = {}
        for binding in node.get("bla:fieldEvidence", []):
            if binding.get("@type") != "bla:FieldEvidenceBinding":
                raise ValueError(f"invalid field-evidence type: {comparison_id}")
            field = binding.get("bla:field")
            if field in bindings:
                raise ValueError(f"duplicate field-evidence binding: {comparison_id}:{field}")
            ids = [evidence_id_from_urn(item.get("@id", "")) for item in binding.get("prov:wasDerivedFrom", [])]
            if not ids:
                raise ValueError(f"empty field-evidence binding: {comparison_id}:{field}")
            if not set(ids).issubset(set(declared_ids)):
                raise ValueError(f"field evidence is outside case evidence: {comparison_id}:{field}")
            for artifact_id in ids:
                if field not in evidence_catalog[artifact_id]["supports_fields"]:
                    raise ValueError(f"artifact does not support field: {comparison_id}:{field}:{artifact_id}")
            bindings[field] = ids
        if set(bindings) != set(config["decision_fields"]):
            raise ValueError(f"field-evidence closure failed: {comparison_id}")
        binding_count += len(bindings)
        case["evidence_artifact_ids"] = declared_ids
        case["field_evidence"] = bindings
        cases.append(case)

    if doc.get("bla:caseCount") != len(cases) or len(cases) != 18:
        raise ValueError("case count drift")
    carrier = {
        "passed": True,
        "root_type_count": len(doc["@type"]),
        "record_set_count": len(record_sets),
        "field_count": len(field_nodes),
        "case_count": len(cases),
        "field_evidence_binding_count": binding_count,
        "required_field_evidence_binding_count": 18 * len(config["decision_fields"]),
        "gold_fields_present": False,
    }
    return cases, evidence_catalog, claims, carrier


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases-jsonld", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--jsonld-diagnostic", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    cases_path = Path(args.cases_jsonld)
    config_path = Path(args.config)
    diagnostic_path = Path(args.jsonld_diagnostic)
    doc = json.loads(cases_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    input_hash = sha256(cases_path)
    if diagnostic.get("status") != "PASS":
        raise ValueError("standard JSON-LD processor pre-gate did not pass")
    if diagnostic.get("check_type") != "PRE_GATE_STANDARD_JSONLD_PROCESSOR_CHECK":
        raise ValueError("unexpected JSON-LD processor check type")
    if diagnostic.get("input_sha256") != input_hash:
        raise ValueError("JSON-LD processor input hash does not match classification input")
    if diagnostic.get("network_document_loader_used") is not False:
        raise ValueError("JSON-LD processor used a network document loader")
    if args.arm not in config["arms"]:
        raise ValueError(f"unknown arm: {args.arm}")
    cases, evidence_catalog, claims, carrier = validate_and_extract(doc, config)
    v1 = load_v1_classifier()

    decisions = []
    for case in cases:
        v1.validate_case(case, config, evidence_catalog)
        decision_class, trace = v1.classify(case, args.arm)
        allowed = set(config["classes"].get(decision_class, {}).get("allowed_claims", []))
        decisions.append(
            {
                "comparison_id": case["comparison_id"],
                "decision_class": decision_class,
                "gate_trace": trace,
                "claim_decisions": {claim: ("allowed" if claim in allowed else "forbidden") for claim in claims},
                "evidence_artifact_ids": case["evidence_artifact_ids"],
                "field_evidence_closure": {
                    "passed": True,
                    "valid_field_count": len(case["field_evidence"]),
                    "required_field_count": len(config["decision_fields"]),
                    "bindings": case["field_evidence"],
                },
            }
        )
    atomic_json(
        Path(args.output),
        {
            "arm": args.arm,
            "case_count": len(decisions),
            "input_sha256": input_hash,
            "config_sha256": sha256(config_path),
            "v1_classifier_sha256": FROZEN_V1_CLASSIFIER_SHA256,
            "jsonld_processor_validation": {
                "status": diagnostic["status"],
                "check_type": diagnostic["check_type"],
                "input_sha256": diagnostic["input_sha256"],
                "pyld_version": diagnostic["pyld_version"],
                "network_document_loader_used": diagnostic["network_document_loader_used"],
            },
            "carrier_validation": carrier,
            "decisions": decisions,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
