#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source "$HOME/mi_entorno/venv/bin/activate"

python -m pip install -r requirements_v2.txt
mkdir -p logs_v2 experiments_v2

python -m compileall -q app_v2 scripts_v2 tests_v2

echo "Phase 2 installed."
echo "Run:"
echo "  python -m unittest discover -s tests_v2 -v"
echo "  bash scripts_v2/run_smoke_v2.sh"
