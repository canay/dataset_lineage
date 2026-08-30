from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from importlib.metadata import version
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
    parser.add_argument("--input", required=True)
    parser.add_argument("--packages", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.packages).resolve()))
    from pyld import jsonld  # type: ignore[import-not-found]

    input_path = Path(args.input)
    doc = json.loads(input_path.read_text(encoding="utf-8"))

    def deny_network_loader(url: str, options: object | None = None) -> object:
        raise RuntimeError(f"network JSON-LD document loading is forbidden: {url}")

    processor_options = {"documentLoader": deny_network_loader}
    expanded = jsonld.expand(doc, options=processor_options)
    nquads = jsonld.to_rdf(
        doc,
        options={"format": "application/n-quads", "documentLoader": deny_network_loader},
    )
    nquad_lines = [line for line in nquads.splitlines() if line.strip()]
    report = {
        "status": "PASS",
        "check_type": "PRE_GATE_STANDARD_JSONLD_PROCESSOR_CHECK",
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest().upper(),
        "pyld_version": version("PyLD"),
        "expanded_top_level_node_count": len(expanded),
        "nquads_statement_count": len(nquad_lines),
        "network_document_loader_used": False,
        "interpretation_boundary": (
            "PyLD expansion and RDF conversion are a hash-bound pre-gate for JSON-LD syntax "
            "and local-context expansion. They are not Croissant certification or empirical "
            "metadata adjudication."
        ),
    }
    atomic_json(Path(args.output), report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
