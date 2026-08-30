#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export PYTHONUNBUFFERED=1
export PYTHONPYCACHEPREFIX=/tmp/pyc

RUN_ID="heart_lineage_extension_20260615"
RESULT_DIR="$ROOT/results/heart_lineage_extension"
LOG_DIR="$ROOT/logs/vps_heart_lineage_extension_20260615"
FIGURE_DIR="$ROOT/figures"
mkdir -p "$RESULT_DIR" "$LOG_DIR" "$FIGURE_DIR"

{
  echo "===== FH4 HEART LINEAGE FULL RUN START $(date -Is) ====="
  echo "cwd=$ROOT"
  echo "host=$(hostname)"
  echo "kernel=$(uname -a)"
  echo "python=$(python3 --version 2>&1)"
  echo "threads OMP=$OMP_NUM_THREADS OPENBLAS=$OPENBLAS_NUM_THREADS MKL=$MKL_NUM_THREADS NUMEXPR=$NUMEXPR_NUM_THREADS"
  echo "--- resource start ---"
  (free -h || true)
  (uptime || true)
  echo "--- python packages ---"
  python3 - <<'PY'
import importlib, sys
mods = ["numpy", "pandas", "sklearn", "scipy", "matplotlib"]
print("python_executable", sys.executable)
for m in mods:
    mod = importlib.import_module(m)
    print(m, getattr(mod, "__version__", "unknown"))
PY
  echo "--- run command ---"
  echo "python3 code/run_heart_lineage_extension.py --result-dir '$RESULT_DIR'"
  python3 code/run_heart_lineage_extension.py --result-dir "$RESULT_DIR"
  echo "--- plot command ---"
  echo "python3 code/plot_heart_lineage_extension.py --result-dir '$RESULT_DIR' --figure-dir '$FIGURE_DIR'"
  python3 code/plot_heart_lineage_extension.py --result-dir "$RESULT_DIR" --figure-dir "$FIGURE_DIR"
  echo "--- artifact inventory ---"
  find "$RESULT_DIR" "$FIGURE_DIR" -maxdepth 2 -type f \( -name 'heart_*' -o -name 'fig_heart_lineage_extension.*' \) -printf '%p\t%s bytes\n' | sort
  echo "--- resource end ---"
  (free -h || true)
  (uptime || true)
  echo "===== FH4 HEART LINEAGE FULL RUN END $(date -Is) ====="
} 2>&1 | tee "$LOG_DIR/heart_lineage_cli_transcript.log"

tar -czf "$ROOT/fh4_heart_lineage_extension_outputs.tar.gz" \
  results/heart_lineage_extension \
  figures/fig_heart_lineage_extension.pdf \
  figures/fig_heart_lineage_extension.png \
  logs/vps_heart_lineage_extension_20260615

echo "packaged=$ROOT/fh4_heart_lineage_extension_outputs.tar.gz"
