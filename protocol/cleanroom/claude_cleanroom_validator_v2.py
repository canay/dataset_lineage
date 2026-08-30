import sys
import re
import json

EVIDENCE_PREFIX = "urn:behavioral-lineage-audit:evidence:"

GOLD_KEYS = {"expected_class", "bla:expectedClass", "gold_label", "bla:goldLabel"}

EXPECTED_CONTEXT = {
    "@vocab": "http://schema.org/",
    "sc": "http://schema.org/",
    "cr": "http://mlcommons.org/croissant/",
    "prov": "http://www.w3.org/ns/prov#",
    "dct": "http://purl.org/dc/terms/",
    "bla": "urn:behavioral-lineage-audit:",
}

EXPECTED_FIELDS = [
    ("comparison_id", "bla:comparisonId"),
    ("dataset_family", "bla:datasetFamily"),
    ("left_state", "bla:leftState"),
    ("right_state", "bla:rightState"),
    ("provenance_status", "bla:provenanceStatus"),
    ("row_correspondence", "bla:rowCorrespondence"),
    ("target_semantics", "bla:targetSemantics"),
    ("evaluation_alignment", "bla:evaluationAlignment"),
    ("behavioral_execution", "bla:behavioralExecution"),
    ("evidence_artifact_ids", "bla:evidenceArtifactIds"),
    ("field_evidence", "bla:fieldEvidence"),
]

REQUIRED_DECISION_FIELDS = [
    "provenance_status",
    "row_correspondence",
    "target_semantics",
    "evaluation_alignment",
    "behavioral_execution",
]

CLAIM_ALLOWLIST = {
    "paired_behavior": {"lineage_fact", "paired_aggregate_effect", "individual_churn_same_task"},
    "target_sensitivity": {"lineage_fact", "target_sensitivity", "same_record_decision_difference_across_targets"},
    "own_test_only": {"lineage_fact", "own_test_difficulty"},
    "lineage_only": {"lineage_fact"},
    "not_identifiable": set(),
}

SHA256_RE = re.compile(r"^[0-9A-F]{64}$")


def contains_gold_keys(node):
    if isinstance(node, dict):
        for key, value in node.items():
            if key in GOLD_KEYS:
                return True
            if contains_gold_keys(value):
                return True
        return False
    if isinstance(node, list):
        return any(contains_gold_keys(item) for item in node)
    return False


def as_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def as_type_set(node):
    return set(as_list(node.get("@type")))


def strip_evidence_id(ref):
    rid = ref.get("@id", "") if isinstance(ref, dict) else ""
    if not rid.startswith(EVIDENCE_PREFIX):
        return None
    return rid[len(EVIDENCE_PREFIX):]


def validate_carrier(root):
    errors = []

    gold_present = contains_gold_keys(root)
    if gold_present:
        errors.append("gold_fields_present")

    context = root.get("@context")
    if not isinstance(context, dict) or context != EXPECTED_CONTEXT:
        errors.append("context")

    root_type_set = as_type_set(root)
    if root_type_set != {"sc:Dataset", "prov:Entity"}:
        errors.append("root_type")

    conforms_to = as_list(root.get("dct:conformsTo"))
    if "http://mlcommons.org/croissant/1.1" not in conforms_to:
        errors.append("conforms_to")

    record_sets = as_list(root.get("cr:recordSet"))
    record_set_count = len(record_sets)
    field_count = 0
    if record_set_count != 1:
        errors.append("record_set_count")
    else:
        rs = record_sets[0]
        if rs.get("@type") != "cr:RecordSet":
            errors.append("record_set_type")
        key = rs.get("cr:key")
        if not isinstance(key, dict) or key.get("@id") != "comparison_cases/comparison_id":
            errors.append("record_set_key")
        fields = as_list(rs.get("cr:field"))
        field_count = len(fields)
        if field_count != len(EXPECTED_FIELDS):
            errors.append("field_count")
        else:
            for field, (expected_name, expected_property) in zip(fields, EXPECTED_FIELDS):
                if field.get("@type") != "cr:Field":
                    errors.append("field_type:" + expected_name)
                if field.get("name") != expected_name:
                    errors.append("field_name:" + expected_name)
                if field.get("cr:equivalentProperty") != expected_property:
                    errors.append("field_property:" + expected_name)

    evidence_nodes = as_list(root.get("bla:evidenceCatalog"))
    catalog = {}
    for node in evidence_nodes:
        artifact_id = node.get("bla:artifactId")
        if as_type_set(node) != {"prov:Entity", "bla:EvidenceArtifact"}:
            errors.append("evidence_type:" + str(artifact_id))
        expected_id = EVIDENCE_PREFIX + str(artifact_id)
        if node.get("@id") != expected_id:
            errors.append("evidence_id:" + str(artifact_id))
        sha = node.get("bla:sha256", "")
        if not SHA256_RE.match(sha):
            errors.append("evidence_sha256:" + str(artifact_id))
        if not node.get("sc:contentUrl"):
            errors.append("evidence_content_url:" + str(artifact_id))
        supports = node.get("bla:supportsField")
        if not isinstance(supports, list) or not supports:
            errors.append("evidence_supports_field:" + str(artifact_id))
        if artifact_id is not None:
            catalog[artifact_id] = node

    cases_raw = as_list(root.get("bla:case"))
    case_count = len(cases_raw)
    if case_count != 18:
        errors.append("case_count")
    if root.get("bla:caseCount") != 18:
        errors.append("case_count_declared")

    parsed_cases = []
    field_evidence_binding_count = 0
    for case in cases_raw:
        comparison_id = case.get("bla:comparisonId")
        if as_type_set(case) != {"bla:ComparisonCase", "prov:Entity"}:
            errors.append("case_type:" + str(comparison_id))

        evidence_artifact_ids = case.get("bla:evidenceArtifactIds")
        if not isinstance(evidence_artifact_ids, list):
            evidence_artifact_ids = []
            errors.append("case_evidence_ids_missing:" + str(comparison_id))

        derived = as_list(case.get("prov:wasDerivedFrom"))
        derived_ids = []
        derived_ok = True
        for ref in derived:
            aid = strip_evidence_id(ref)
            if aid is None:
                derived_ok = False
                break
            derived_ids.append(aid)
        if not derived_ok or derived_ids != evidence_artifact_ids:
            errors.append("case_evidence_order:" + str(comparison_id))
        for aid in evidence_artifact_ids:
            if aid not in catalog:
                errors.append("case_evidence_unknown:" + str(comparison_id))

        case_evidence_set = set(evidence_artifact_ids)

        field_evidence = as_list(case.get("bla:fieldEvidence"))
        field_evidence_binding_count += len(field_evidence)
        bindings_by_field = {}
        seen_fields = set()
        if len(field_evidence) != len(REQUIRED_DECISION_FIELDS):
            errors.append("field_evidence_count:" + str(comparison_id))
        for binding in field_evidence:
            if binding.get("@type") != "bla:FieldEvidenceBinding":
                errors.append("field_evidence_type:" + str(comparison_id))
            field_name = binding.get("bla:field")
            if field_name not in REQUIRED_DECISION_FIELDS:
                errors.append("field_evidence_field:" + str(comparison_id))
                continue
            if field_name in seen_fields:
                errors.append("field_evidence_duplicate:" + str(comparison_id) + ":" + str(field_name))
                continue
            seen_fields.add(field_name)
            refs = as_list(binding.get("prov:wasDerivedFrom"))
            ids = []
            refs_ok = bool(refs)
            for ref in refs:
                aid = strip_evidence_id(ref)
                if aid is None:
                    refs_ok = False
                    break
                ids.append(aid)
            valid_subset = refs_ok and set(ids).issubset(case_evidence_set)
            valid_support = refs_ok and all(
                field_name in catalog.get(aid, {}).get("bla:supportsField", [])
                for aid in ids
            )
            if not (valid_subset and valid_support):
                errors.append("field_evidence_binding:" + str(comparison_id) + ":" + str(field_name))
            bindings_by_field[field_name] = ids
        if seen_fields != set(REQUIRED_DECISION_FIELDS):
            errors.append("field_evidence_missing:" + str(comparison_id))

        parsed_cases.append({
            "comparison_id": comparison_id,
            "provenance_status": case.get("bla:provenanceStatus"),
            "row_correspondence": case.get("bla:rowCorrespondence"),
            "target_semantics": case.get("bla:targetSemantics"),
            "evaluation_alignment": case.get("bla:evaluationAlignment"),
            "behavioral_execution": case.get("bla:behavioralExecution"),
            "evidence_artifact_ids": evidence_artifact_ids,
            "bindings_by_field": bindings_by_field,
        })

    required_field_evidence_binding_count = case_count * len(REQUIRED_DECISION_FIELDS)

    passed = not errors

    carrier_validation = {
        "passed": passed,
        "root_type_count": len(root_type_set),
        "record_set_count": record_set_count,
        "field_count": field_count,
        "case_count": case_count,
        "field_evidence_binding_count": field_evidence_binding_count,
        "required_field_evidence_binding_count": required_field_evidence_binding_count,
        "gold_fields_present": gold_present,
    }

    claim_catalog = root.get("bla:claimCatalog")
    if not isinstance(claim_catalog, list):
        claim_catalog = []

    return passed, carrier_validation, parsed_cases, claim_catalog


def finalize(case, claim_catalog, decision_class, gate_trace):
    allowed = CLAIM_ALLOWLIST.get(decision_class, set())
    claim_decisions = {
        claim: ("allowed" if claim in allowed else "forbidden")
        for claim in claim_catalog
    }

    bindings_by_field = case["bindings_by_field"]
    bindings = {field: bindings_by_field.get(field, []) for field in REQUIRED_DECISION_FIELDS}
    valid_field_count = sum(1 for field in REQUIRED_DECISION_FIELDS if bindings_by_field.get(field))
    field_evidence_closure = {
        "passed": valid_field_count == len(REQUIRED_DECISION_FIELDS),
        "valid_field_count": valid_field_count,
        "required_field_count": len(REQUIRED_DECISION_FIELDS),
        "bindings": bindings,
    }

    return {
        "comparison_id": case["comparison_id"],
        "decision_class": decision_class,
        "gate_trace": gate_trace,
        "claim_decisions": claim_decisions,
        "evidence_artifact_ids": case["evidence_artifact_ids"],
        "field_evidence_closure": field_evidence_closure,
    }


def decide_case(case, claim_catalog):
    gate_trace = []

    def add_gate(name, gate_passed, value):
        gate_trace.append({"gate": name, "passed": gate_passed, "value": value})

    provenance_status = case["provenance_status"]
    provenance_passed = provenance_status == "verified"
    add_gate("provenance", provenance_passed, provenance_status)
    if not provenance_passed:
        return finalize(case, claim_catalog, "not_identifiable", gate_trace)

    behavioral_execution = case["behavioral_execution"]
    execution_passed = behavioral_execution == "verified"
    add_gate("executed_evidence", execution_passed, behavioral_execution)

    if behavioral_execution == "unknown":
        return finalize(case, claim_catalog, "not_identifiable", gate_trace)

    if behavioral_execution == "absent":
        row_correspondence = case["row_correspondence"]
        target_semantics = case["target_semantics"]
        lineage_passed = target_semantics == "equivalent" and row_correspondence == "verified"
        add_gate("lineage_fact_only", lineage_passed, "identity_without_execution")
        decision_class = "lineage_only" if lineage_passed else "not_identifiable"
        return finalize(case, claim_catalog, decision_class, gate_trace)

    row_correspondence = case["row_correspondence"]
    evaluation_alignment = case["evaluation_alignment"]
    target_semantics = case["target_semantics"]

    pairing_passed = row_correspondence == "verified" and evaluation_alignment == "same_frozen_records"
    add_gate("pairing", pairing_passed, f"{row_correspondence}|{evaluation_alignment}")

    if pairing_passed:
        ts_passed = target_semantics in ("equivalent", "different")
        add_gate("target_semantics", ts_passed, target_semantics)
        if target_semantics == "equivalent":
            decision_class = "paired_behavior"
        elif target_semantics == "different":
            decision_class = "target_sensitivity"
        else:
            decision_class = "not_identifiable"
    else:
        otb_passed = (
            row_correspondence == "absent"
            and evaluation_alignment == "own_test_splits"
            and target_semantics == "equivalent"
        )
        add_gate(
            "own_test_boundary",
            otb_passed,
            f"{row_correspondence}|{evaluation_alignment}|{target_semantics}",
        )
        decision_class = "own_test_only" if otb_passed else "not_identifiable"

    return finalize(case, claim_catalog, decision_class, gate_trace)


def main():
    if len(sys.argv) != 3:
        sys.stderr.write("usage: claude_cleanroom_validator_v2.py <input.jsonld> <decision.json>\n")
        sys.exit(2)

    input_path, output_path = sys.argv[1], sys.argv[2]

    with open(input_path, "r", encoding="utf-8") as handle:
        root = json.load(handle)

    passed, carrier_validation, parsed_cases, claim_catalog = validate_carrier(root)

    decisions = []
    if passed:
        for case in parsed_cases:
            decisions.append(decide_case(case, claim_catalog))

    output = {
        "arm": "FULL",
        "case_count": carrier_validation["case_count"],
        "carrier_validation": carrier_validation,
        "decisions": decisions,
    }

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
        handle.write("\n")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
