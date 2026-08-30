# Dataset-lineage audit protocol

This repository is the curated replication package for **“An auditable dataset-lineage protocol for measuring model-behavior changes across reconstructed provenance and preprocessing decisions.”** It contains the analysis code, fixed configurations, aggregate outputs, source-provenance metadata, and the machine-readable comparison-admissibility profile used in the study.

The package supports two related tasks:

1. reconstructing documented dataset-lineage decisions and measuring their behavioral effects under fixed models and evaluation records; and
2. validating whether a proposed pairwise comparison has enough provenance, semantic, alignment, and execution evidence to support behavioral claims.

## Repository contents

- `code/` — data preparation, experiment, aggregation, robustness-audit, and plotting scripts.
- `results/` — aggregate result tables and machine-readable summaries reported by the study. Per-record predictions and raw benchmark data are intentionally excluded.
- `provenance/` — captured source metadata used to reconstruct dataset genealogy.
- `protocol/` — JSON-LD comparison cases, profile configuration, primary and clean-room validators, fail-closed tests, ablation outputs, and decision summaries.
- `requirements*.txt` — pinned environments for the primary analyses, revision-stage analyses, and Heart Disease extension.

## Data policy

Raw benchmark files are not redistributed. Obtain them from their original repositories and retain their original terms of use. The manuscript’s provenance table identifies the source artifacts and reports their SHA-256 digests. The analysis scripts expect source files under `data/`; see the path constants at the beginning of `code/prepare_data.py`, `code/run_compas_lineage.py`, and `code/run_heart_lineage_extension.py` for the required local layout.

The study uses Adult and German Credit source files from the UCI Machine Learning Repository, ACS PUMS files from the U.S. Census Bureau, the public COMPAS file released by ProPublica, and the UCI Heart Disease files. OpenML metadata captured for the lineage reconstruction is provided in `provenance/openml_lineage_meta.json`.

## Environments

Create separate environments because the analyses were frozen under different dependency sets:

```bash
python -m venv .venv-primary
.venv-primary/bin/python -m pip install -r requirements.txt

python -m venv .venv-revision
.venv-revision/bin/python -m pip install -r requirements-revision.txt
```

On Windows, activate the environment through `Scripts` or invoke its Python executable directly. The Heart Disease extension uses `requirements-heart.txt`. The JSON-LD standards check additionally uses `protocol/env/requirements-jsonld.txt`; install it into an isolated package directory with `python -m pip install --target protocol/env/jsonld-packages -r protocol/env/requirements-jsonld.txt`.

## Main analysis

After placing the source files in the paths expected by the scripts:

```bash
python code/prepare_data.py all
python code/run_units.py 0 1
python code/aggregate.py
python code/run_compas_lineage.py
python code/audit_extra_stats.py
python code/audit_allrows_eval.py
python code/audit_perm_stability.py
python code/figures.py all
```

`run_units.py` accepts a zero-based shard number and the total number of shards, so the experiment grid may be distributed by running `python code/run_units.py <shard> <n_shards>` for each shard. Existing unit outputs are reused.

The released aggregate files permit direct inspection of the reported results without redistributing the per-record benchmark inputs or prediction vectors. Scripts that compute new per-record statistics require those locally reconstructed inputs.

## Comparison-admissibility profile

The profile is fail-closed: a behavioral claim is allowed only when the required evidence bindings and comparison gates pass. From the repository root, validate the released JSON-LD carrier and profile with:

```bash
python protocol/src/validate_jsonld_processor.py \
  --input protocol/cases/cases_sanitized_v2.jsonld \
  --packages protocol/env/jsonld-packages \
  --output protocol/metrics/jsonld_processor_validation.reproduced.json

python protocol/src/validate_profile_v2.py \
  --cases-jsonld protocol/cases/cases_sanitized_v2.jsonld \
  --config protocol/configs/profile_config_v2.json \
  --jsonld-diagnostic protocol/metrics/jsonld_processor_validation.reproduced.json \
  --arm FULL \
  --output protocol/processed_outputs/full.reproduced.json

python protocol/src/test_fail_closed_v2.py \
  --cases-jsonld protocol/cases/cases_sanitized_v2.jsonld \
  --config protocol/configs/profile_config_v2.json
```

The v2 validator imports the hash-pinned frozen v1 classifier retained at `2026-08-26_codex_local_comparison_admissibility_validation/src/validate_profile.py`; the embedded digest prevents silent classifier drift. The clean-room contract and independent implementation are retained in `protocol/cleanroom/`. Gold labels are provided for test evaluation and are not read by the validators.

## Evidence boundary

The repository supports reproduction and audit of the released analyses; it does not establish that every dataset sharing a name or identifier is content-equivalent. Dataset identity and comparison admissibility remain evidence-bound to the source files, digests, semantic fields, alignment rules, and execution records specified by the protocol.

## Citation

Citation metadata is available in `CITATION.cff`. Until the article has a DOI, cite this repository and the manuscript title shown above.

## License

Code and original documentation in this repository are released under the MIT License. Third-party datasets are not included and remain governed by their source repositories’ terms.
