#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source "$HOME/mi_entorno/venv/bin/activate"
export PYTHONPATH="$PWD"
python -m app_v2.experiment_runner --mode full
