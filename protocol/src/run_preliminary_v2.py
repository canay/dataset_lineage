from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def atomic_json(path: Path, value: object) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def main() -> int:
    run_dir = Path(__file__).resolve().parents[1]
    root = run_dir.parents[1]
    started = datetime.now(ZoneInfo("Europe/Istanbul"))
    config = run_dir / "configs" / "profile_config_v2.json"
    cases = run_dir / "data_inputs" / "cases_sanitized_v2.jsonld"
    gold = run_dir / "data_inputs" / "gold_labels_v2.json"
    manifest = run_dir / "data_inputs" / "input_manifest_v2.json"
    jsonld_diagnostic = run_dir / "metrics" / "jsonld_processor_validation.json"
    subprocess.run(
        [
            sys.executable,
            str(run_dir / "src" / "build_inputs_v2.py"),
            "--project-root",
            str(root),
            "--profile",
            "MD/02_design/comparison_admissibility_profile_candidate_v2.md",
            "--cases",
            "MD/02_design/comparison_cases_candidate_v2.json",
            "--output",
            str(cases),
            "--gold",
            str(gold),
            "--manifest",
            str(manifest),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(run_dir / "src" / "validate_jsonld_processor.py"),
            "--input",
            str(cases),
            "--packages",
            str(run_dir / "env" / "python_packages"),
            "--output",
            str(jsonld_diagnostic),
        ],
        check=True,
    )
    config_doc = json.loads(config.read_text(encoding="utf-8"))
    for arm in config_doc["arms"]:
        subprocess.run(
            [
                sys.executable,
                str(run_dir / "src" / "validate_profile_v2.py"),
                "--cases-jsonld",
                str(cases),
                "--config",
                str(config),
                "--jsonld-diagnostic",
                str(jsonld_diagnostic),
                "--arm",
                arm,
                "--output",
                str(run_dir / "raw_outputs" / f"{arm.lower()}.json"),
            ],
            check=True,
        )
    subprocess.run(
        [
            sys.executable,
            str(run_dir / "src" / "aggregate_v2.py"),
            "--gold",
            str(gold),
            "--config",
            str(config),
            "--decisions-dir",
            str(run_dir / "raw_outputs"),
            "--output",
            str(run_dir / "processed_outputs" / "preliminary_summary_v2.json"),
        ],
        check=True,
    )
    ended = datetime.now(ZoneInfo("Europe/Istanbul"))
    terminal = {
        "run_id": run_dir.name,
        "status": "preliminary_completed",
        "start_time": started.isoformat(),
        "end_time": ended.isoformat(),
        "elapsed_seconds": (ended - started).total_seconds(),
        "exit_code": 0,
        "planned_decisions": 108,
        "completed_decisions": 108,
        "python": sys.version,
        "platform": platform.platform(),
    }
    atomic_json(run_dir / "terminal_status.json", terminal)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
