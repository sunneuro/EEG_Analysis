#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_pipeline.sh
# Usage:
#   ./scripts/run_pipeline.sh --task nback
#   ./scripts/run_pipeline.sh --task stroop P01
# ─────────────────────────────────────────────────────────────────────────────

if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
fi

python3 scripts/run_pipeline.py "$@"
