from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = ROOT / "cleanroom" / "claude_cleanroom_validator_v2.py"
INPUT_PATH = ROOT / "data_inputs" / "cases_sanitized_v2.jsonld"


def load_validator() -> object:
    spec = importlib.util.spec_from_file_location("cleanroom_v2_under_test", VALIDATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load clean-room validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    validator = load_validator()
    source = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    mutations = {}

    value = copy.deepcopy(source)
    value["@context"]["cr"] = "https://example.invalid/croissant/"
    mutations["context_uri_drift"] = value

    value = copy.deepcopy(source)
    value["cr:recordSet"][0]["@type"] = "sc:Dataset"
    mutations["recordset_type_drift"] = value

    value = copy.deepcopy(source)
    value["cr:recordSet"][0]["cr:field"][-1]["cr:equivalentProperty"] = "bla:field_evidence"
    mutations["field_property_mapping_drift"] = value

    value = copy.deepcopy(source)
    value["bla:case"][0]["bla:fieldEvidence"].pop()
    mutations["missing_field_binding"] = value

    value = copy.deepcopy(source)
    value["bla:case"][0]["bla:fieldEvidence"][0]["prov:wasDerivedFrom"][0]["@id"] = "urn:wrong:evidence:x"
    mutations["evidence_prefix_drift"] = value

    value = copy.deepcopy(source)
    value["bla:case"][0]["expected_class"] = "paired_behavior"
    mutations["gold_field_leak"] = value

    results = {}
    with tempfile.TemporaryDirectory(prefix="f04-cleanroom-v2-") as tmp_dir:
        tmp = Path(tmp_dir)
        for name, mutated in mutations.items():
            passed, carrier, _, _ = validator.validate_carrier(mutated)
            input_path = tmp / f"{name}.jsonld"
            output_path = tmp / f"{name}.json"
            input_path.write_text(json.dumps(mutated), encoding="utf-8")
            completed = subprocess.run(
                [sys.executable, str(VALIDATOR_PATH), str(input_path), str(output_path)],
                check=False,
                capture_output=True,
                text=True,
            )
            output = json.loads(output_path.read_text(encoding="utf-8"))
            rejected = (
                not passed
                and not carrier["passed"]
                and completed.returncode != 0
                and not output["carrier_validation"]["passed"]
                and output["decisions"] == []
            )
            results[name] = {"rejected": rejected, "exit_code": completed.returncode}

    print(json.dumps(results, indent=2, sort_keys=True))
    return 0 if all(item["rejected"] for item in results.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
