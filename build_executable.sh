#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEMP_VENV="$(mktemp -d -t solver-build-XXXXXX)"

cleanup() {
    if [[ -d "$TEMP_VENV" ]]; then
        rm -rf "$TEMP_VENV"
    fi
}
trap cleanup EXIT

python3 -m venv "$TEMP_VENV"
source "$TEMP_VENV/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "$PROJECT_ROOT/requirements.txt"

cd "$PROJECT_ROOT"
pyinstaller solver.py --onefile --name solver-cli --clean

rm -rf build solver.spec

echo "Ejecutable disponible en $PROJECT_ROOT/dist/solver-cli"
