#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
source "$HOME/mi_entorno/venv/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r requirements_v2.txt

mkdir -p logs_v2 experiments_v2

echo "Phase 1 installed."
echo "Run:"
echo "  python -m unittest discover -s tests_v2 -v"
echo "  python scripts_v2/preflight_v2.py"
