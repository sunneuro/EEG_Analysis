#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_nback_erp_pipeline.sh
#
# Runs the full N-back ERP analysis pipeline for all participants.
# All participants included — outlier flags recorded in output, not excluded.
#
# Pipeline:
#   Step 1: 06_erp_nback.py      — per-participant ERP extraction
#   Step 2: 07_group_erp_nback.py — group grand averages and plots
#   Step 3: 08_statistics_nback.py — between-group statistical comparisons
#
# Usage:
#   chmod +x scripts/run_nback_erp_pipeline.sh
#   ./scripts/run_nback_erp_pipeline.sh           # all participants
#   ./scripts/run_nback_erp_pipeline.sh P01       # single participant
#
# Requirements:
#   - Run from project root: /Users/SK/eeg_analysis/
#   - Virtual environment at .venv/
#   - Scripts 03, 04, 05 already run for all participants
#   - data/processed/P0X_nback_epo.fif exists for each participant
# ─────────────────────────────────────────────────────────────────────────────

set -e

# ── Colour codes ──────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m'

# ── Participants ──────────────────────────────────────────────────────────────
ALL_PARTICIPANTS=("P01" "P02" "P03" "P04" "P05" "P06" "P07" "P08")
# Note: P08 included — low target/hit (n=5) and target/miss (n=11) flagged
# Note: P04 included — low SNR noisy baseline flagged
# Note: P05, P06 target/hit included — low trial count flagged
# Note: P07, P03 target/miss included — high miss rate flagged

# ── Parse optional single-participant argument ────────────────────────────────
if [ $# -eq 1 ]; then
    PARTICIPANTS=("$1")
    echo -e "${YELLOW}Running for single participant: $1${NC}"
else
    PARTICIPANTS=("${ALL_PARTICIPANTS[@]}")
    echo -e "${YELLOW}Running for all participants: ${ALL_PARTICIPANTS[*]}${NC}"
fi

# ── Pre-flight: verify project root ──────────────────────────────────────────
if [ ! -d "scripts" ] || [ ! -d "data" ]; then
    echo -e "${RED}ERROR: Run this script from the project root directory.${NC}"
    echo -e "${RED}       Expected : /Users/SK/eeg_analysis/${NC}"
    echo -e "${RED}       Current  : $(pwd)${NC}"
    exit 1
fi

# ── Pre-flight: activate virtual environment ──────────────────────────────────
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo -e "${GREEN}Virtual environment activated.${NC}"
else
    echo -e "${YELLOW}WARNING: .venv not found — using system Python.${NC}"
fi

# ── Pipeline start ────────────────────────────────────────────────────────────
PIPELINE_START=$(date +%s)

echo ""
echo -e "${BOLD}============================================================${NC}"
echo -e "${BOLD} NBACK ERP PIPELINE — $(date '+%Y-%m-%d %H:%M:%S')${NC}"
echo -e "${BOLD}============================================================${NC}"
echo ""
echo -e "  Participants : ${PARTICIPANTS[*]}"
echo -e "  Outputs      : output/erp/  output/group/  output/stats/"
echo ""

# ── Tracking ──────────────────────────────────────────────────────────────────
PASSED=()
FAILED=()
SKIPPED=()

# ── STEP 1: Individual ERP extraction ────────────────────────────────────────
echo -e "${BOLD}STEP 1: Individual ERP extraction (06_erp_nback.py)${NC}"
echo "------------------------------------------------------------"

for PID in "${PARTICIPANTS[@]}"; do

    EPO_FILE="data/processed/${PID}_nback_epo.fif"

    if [ ! -f "$EPO_FILE" ]; then
        echo -e "  ${YELLOW}SKIP${NC} ${PID} — epoch file not found: ${EPO_FILE}"
        SKIPPED+=("$PID")
        continue
    fi

    echo -e "\n  ${BLUE}▶ ${PID}${NC}"
    STEP_START=$(date +%s)

    if python3 scripts/06_erp_nback.py "$PID" 2>&1 | \
        while IFS= read -r line; do echo "    $line"; done; then
        STEP_END=$(date +%s)
        echo -e "  ${GREEN}✓ ${PID} — completed in $((STEP_END - STEP_START))s${NC}"
        PASSED+=("$PID")
    else
        STEP_END=$(date +%s)
        echo -e "  ${RED}✗ ${PID} — FAILED after $((STEP_END - STEP_START))s${NC}"
        FAILED+=("$PID")
    fi

done

echo ""
echo "------------------------------------------------------------"
echo -e "  Passed  : ${GREEN}${#PASSED[@]} — ${PASSED[*]:-none}${NC}"
echo -e "  Skipped : ${YELLOW}${#SKIPPED[@]} — ${SKIPPED[*]:-none}${NC}"
echo -e "  Failed  : ${RED}${#FAILED[@]} — ${FAILED[*]:-none}${NC}"
echo ""

# ── Abort if nothing passed ───────────────────────────────────────────────────
if [ ${#PASSED[@]} -eq 0 ]; then
    echo -e "${RED}ERROR: No participants completed Step 1.${NC}"
    echo -e "${RED}       Skipping group and statistics steps.${NC}"
    exit 1
fi

if [ ${#FAILED[@]} -gt 0 ]; then
    echo -e "${YELLOW}WARNING: ${#FAILED[@]} participant(s) failed in Step 1.${NC}"
    echo -e "${YELLOW}         Continuing group analysis with: ${PASSED[*]}${NC}"
    echo ""
fi

# ── STEP 2: Group ERP analysis ────────────────────────────────────────────────
echo -e "${BOLD}STEP 2: Group ERP analysis (07_group_erp_nback.py)${NC}"
echo "------------------------------------------------------------"

STEP_START=$(date +%s)
set +e
python3 scripts/07_group_erp_nback.py 2>&1 | \
    while IFS= read -r line; do echo "  $line"; done
STEP_EXIT=${PIPESTATUS[0]}
set -e
STEP_END=$(date +%s)

if [ $STEP_EXIT -eq 0 ]; then
    echo -e "${GREEN}✓ Group ERP complete — $((STEP_END - STEP_START))s${NC}"
else
    echo -e "${RED}✗ Group ERP FAILED after $((STEP_END - STEP_START))s${NC}"
    echo -e "${RED}  Check output above. Continuing to statistics step.${NC}"
fi
echo ""

# ── STEP 3: Statistics ────────────────────────────────────────────────────────
echo -e "${BOLD}STEP 3: Statistical analysis (08_statistics_nback.py)${NC}"
echo "------------------------------------------------------------"

STEP_START=$(date +%s)
set +e
python3 scripts/08_statistics_nback.py 2>&1 | \
    while IFS= read -r line; do echo "  $line"; done
STEP_EXIT=${PIPESTATUS[0]}
set -e
STEP_END=$(date +%s)

if [ $STEP_EXIT -eq 0 ]; then
    echo -e "${GREEN}✓ Statistics complete — $((STEP_END - STEP_START))s${NC}"
else
    echo -e "${RED}✗ Statistics FAILED after $((STEP_END - STEP_START))s${NC}"
fi
echo ""

# ── Final summary ─────────────────────────────────────────────────────────────
PIPELINE_END=$(date +%s)
TOTAL=$((PIPELINE_END - PIPELINE_START))
MINS=$((TOTAL / 60))
SECS=$((TOTAL % 60))

echo -e "${BOLD}============================================================${NC}"
echo -e "${BOLD} NBACK PIPELINE COMPLETE — ${MINS}m ${SECS}s${NC}"
echo -e "${BOLD} Finished: $(date '+%Y-%m-%d %H:%M:%S')${NC}"
echo -e "${BOLD}============================================================${NC}"
echo ""
echo -e "  Participants passed  : ${GREEN}${#PASSED[@]} — ${PASSED[*]:-none}${NC}"
echo -e "  Participants skipped : ${YELLOW}${#SKIPPED[@]} — ${SKIPPED[*]:-none}${NC}"
echo -e "  Participants failed  : ${RED}${#FAILED[@]} — ${FAILED[*]:-none}${NC}"
echo ""
echo -e "  Individual ERPs : output/erp/P0X_nback_*.png"
echo -e "                    output/erp/P0X_nback_erp_components.csv"
echo -e "  Group plots     : output/group/group_nback_*.png"
echo -e "  Group CSVs      : output/group/group_nback_*.csv"
echo -e "  Statistics      : output/stats/nback_statistics_primary.csv"
echo -e "                    output/stats/nback_statistics_exploratory.csv"
echo -e "                    output/stats/nback_statistics_sensitivity.csv"
echo ""
echo -e "  Outlier notes:"
echo -e "    P08 — target/hit n=5, target/miss n=11   (included, flagged)"
echo -e "    P04 — low SNR noisy baseline              (included, flagged)"
echo -e "    P05 — target/hit n=33, reversed sens      (included, flagged)"
echo -e "    P06 — target/hit n=32                     (included, flagged)"
echo -e "    P07 — target/miss n=59 high miss rate     (included, flagged)"
echo -e "    P03 — target/miss n=81 high miss rate     (included, flagged)"
echo ""
