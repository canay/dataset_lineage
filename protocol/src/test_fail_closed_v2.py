from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path


def load_validator(path: Path) -> object:
    spec = importlib.util.spec_from_file_location("validator_v2_under_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases-jsonld", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    validator = load_validator(Path(__file__).with_name("validate_profile_v2.py"))
    source = json.loads(Path(args.cases_jsonld).read_text(encoding="utf-8"))
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))

    mutations = {}
    value = copy.deepcopy(source)
    value["@context"].pop("cr")
    mutations["missing_cr_namespace"] = value
    value = copy.deepcopy(source)
    value["bla:case"][0]["expected_class"] = "paired_behavior"
    mutations["gold_field_leak"] = value
    value = copy.deepcopy(source)
    value["bla:case"][0]["bla:fieldEvidence"].pop()
    mutations["missing_field_binding"] = value
    value = copy.deepcopy(source)
    value["bla:evidenceCatalog"][0]["bla:supportsField"] = ["provenance_status"]
    mutations["artifact_field_support_mismatch"] = value
    value = copy.deepcopy(source)
    value["bla:case"][0]["bla:evidenceArtifactIds"].append("unknown_artifact")
    mutations["unknown_evidence_id"] = value
    value = copy.deepcopy(source)
    value["cr:recordSet"][0]["cr:key"]["@id"] = "comparison_cases/wrong"
    mutations["recordset_key_drift"] = value
    value = copy.deepcopy(source)
    value["cr:recordSet"][0]["cr:field"][-1]["cr:equivalentProperty"] = "bla:field_evidence"
    mutations["field_property_mapping_drift"] = value

    results = {}
    for name, mutated in mutations.items():
        try:
            validator.validate_and_extract(mutated, config)
        except ValueError as exc:
            results[name] = {"rejected": True, "reason": str(exc)}
        else:
            results[name] = {"rejected": False, "reason": "accepted unexpectedly"}
    print(json.dumps(results, indent=2, ensure_ascii=False, sort_keys=True))
    return 0 if all(item["rejected"] for item in results.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
