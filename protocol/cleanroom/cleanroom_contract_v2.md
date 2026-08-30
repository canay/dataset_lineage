# Clean-room implementation contract v2

Date/time: 2026-08-26 17:14 +03:00  
Tool: Codex  
Model, if known: GPT-5  
Operation ID: `f04-behavioral-lineage-profile-validation-v2-20260826`

The independent implementer may read only:

1. `MD/02_design/comparison_admissibility_profile_candidate.md`;
2. `MD/02_design/comparison_admissibility_profile_candidate_v2.md`;
3. `data_inputs/cases_sanitized_v2.jsonld`;
4. this contract.

It must not read gold labels, configs, main source, v1/v2 outputs, aggregate
summaries, the manuscript, project state, or the web. It writes no files.

## CLI and output

Use only the Python standard library:

```text
python claude_cleanroom_validator_v2.py <input.jsonld> <decision.json>
```

The program accepts exactly those two positional arguments after the script
name. A carrier-validation failure is a rejected run: it writes the diagnostic
output but exits non-zero.

Output contains `arm: FULL`, `case_count`, `carrier_validation`, and decisions
in input order. Each decision contains exactly:

- `comparison_id`;
- `decision_class`;
- `gate_trace`;
- `claim_decisions`, a JSON object mapping every string in the input
  `bla:claimCatalog` to exactly `allowed` or `forbidden`;
- `evidence_artifact_ids`;
- `field_evidence_closure`, with `passed`, `valid_field_count`,
  `required_field_count`, and `bindings`. `bindings` is a JSON object whose
  keys are the five decision fields and whose values are the ordered evidence
  artifact-ID arrays extracted from that field's binding.

## Fail-closed JSON-LD carrier checks

- Reject recursively any key `expected_class`, `bla:expectedClass`,
  `gold_label`, or `bla:goldLabel`.
- `@context` must have exactly these six bindings and values:
  `@vocab=http://schema.org/`, `sc=http://schema.org/`,
  `cr=http://mlcommons.org/croissant/`,
  `prov=http://www.w3.org/ns/prov#`,
  `dct=http://purl.org/dc/terms/`, and
  `bla=urn:behavioral-lineage-audit:`. Checking only the key set or only that
  `bla` differs from the standard values is insufficient.
- Root type set is exactly `sc:Dataset` plus `prov:Entity` and
  `dct:conformsTo` contains `http://mlcommons.org/croissant/1.1`.
- Exactly one `cr:RecordSet` exists: the sole `cr:recordSet` node has
  `@type=cr:RecordSet`. Its key is
  `comparison_cases/comparison_id`. Its `cr:field` names are exactly the 11
  fields carried by the case contract and every field is a `cr:Field`. The
  `cr:equivalentProperty` values must exactly match the actual case properties:
  `bla:comparisonId`, `bla:datasetFamily`, `bla:leftState`, `bla:rightState`,
  `bla:provenanceStatus`, `bla:rowCorrespondence`, `bla:targetSemantics`,
  `bla:evaluationAlignment`, `bla:behavioralExecution`,
  `bla:evidenceArtifactIds`, and `bla:fieldEvidence`, respectively.
- Every `bla:evidenceCatalog` node is both `prov:Entity` and
  `bla:EvidenceArtifact`; its `@id` is exactly
  `urn:behavioral-lineage-audit:evidence:<bla:artifactId>`; SHA-256 is 64
  uppercase hexadecimal characters; `sc:contentUrl` is non-empty; and
  `bla:supportsField` is non-empty.
- Every `bla:case` is both `bla:ComparisonCase` and `prov:Entity`. Extract
  fields from the compact `bla:` properties used in the JSON-LD input.
- `bla:evidenceArtifactIds` must exactly equal the ordered artifact IDs in the
  case-level `prov:wasDerivedFrom`; every reference URI must use the exact
  `urn:behavioral-lineage-audit:evidence:` prefix and every ID must exist in the
  catalog. The same exact-prefix rule applies to field-evidence references.
- The five required decision fields are `provenance_status`,
  `row_correspondence`, `target_semantics`, `evaluation_alignment`, and
  `behavioral_execution`. Every case has exactly one
  `bla:FieldEvidenceBinding` per field. Each binding is non-empty, is a subset
  of case evidence, and points only to catalog artifacts whose
  `bla:supportsField` includes that field.
- `bla:caseCount` and actual case count must both be 18.

`carrier_validation` is exactly:

```json
{
  "passed": true,
  "root_type_count": 2,
  "record_set_count": 1,
  "field_count": 11,
  "case_count": 18,
  "field_evidence_binding_count": 90,
  "required_field_evidence_binding_count": 90,
  "gold_fields_present": false
}
```

## Canonical decision trace

- Append `provenance` first; passed iff status is `verified`, value is the
  source value. Failure returns `not_identifiable`.
- Append `executed_evidence` second; passed iff execution is `verified`, value
  is the source value. `unknown` returns `not_identifiable`.
- If execution is `absent`, append `lineage_fact_only`; passed iff target is
  `equivalent` and row correspondence is `verified`; value is
  `identity_without_execution`. Passed returns `lineage_only`, otherwise
  `not_identifiable`.
- With verified execution, append `pairing`; passed iff row correspondence is
  `verified` and evaluation alignment is `same_frozen_records`; value is
  `<row_correspondence>|<evaluation_alignment>`.
- If pairing passes, append `target_semantics`; passed iff target is
  `equivalent` or `different`, value is the source target. Equivalent returns
  `paired_behavior`; different returns `target_sensitivity`; unknown returns
  `not_identifiable`.
- If pairing fails, append `own_test_boundary`; passed iff row correspondence
  is `absent`, evaluation alignment is `own_test_splits`, and target is
  `equivalent`; value is
  `<row_correspondence>|<evaluation_alignment>|<target_semantics>`. Passed
  returns `own_test_only`; failure returns `not_identifiable`.

Allowed claims are exactly those in the frozen profile. In particular,
`lineage_fact` is allowed for `paired_behavior`, `target_sensitivity`,
`own_test_only`, and `lineage_only`, and forbidden only for
`not_identifiable`. The program must not contain a dataset-family name or
comparison ID in a decision branch.
