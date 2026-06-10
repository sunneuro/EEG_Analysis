#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_stroop_erp_pipeline.sh
#
# Runs the full Stroop ERP analysis pipeline for all participants.
# All participants included — outlier flags recorded in output, not excluded.
#
# Pipeline:
#   Step 1: 06_erp_stroop.py      — per-participant ERP extraction
#   Step 2: 07_group_erp_stroop.py — group grand averages and plots
#   Step 3: 08_statistics_stroop.py — between-group statistical comparisons
#
# Usage:
#   chmod +x scripts/run_stroop_erp_pipeline.sh
#   ./scripts/run_stroop_erp_pipeline.sh           # all participants
#   ./scripts/run_stroop_erp_pipeline.sh P01       # single participant
#
# Requirements:
#   - Run from project root: /Users/SK/eeg_analysis/
#   - Virtual environment at .venv/
#   - Scripts 03, 04, 05 already run for all participants
#   - data/processed/P0X_stroop_epo.fif exists for each participant
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
# Note: P03 included — high miss rate flagged in output, not excluded
# Note: P08 included — low congruent trials (n=6) flagged in output

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
echo -e "${BOLD} STROOP ERP PIPELINE — $(date '+%Y-%m-%d %H:%M:%S')${NC}"
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
echo -e "${BOLD}STEP 1: Individual ERP extraction (06_erp_stroop.py)${NC}"
echo "------------------------------------------------------------"

for PID in "${PARTICIPANTS[@]}"; do

    EPO_FILE="data/processed/${PID}_stroop_epo.fif"

    if [ ! -f "$EPO_FILE" ]; then
        echo -e "  ${YELLOW}SKIP${NC} ${PID} — epoch file not found: ${EPO_FILE}"
        SKIPPED+=("$PID")
        continue
    fi

    echo -e "\n  ${BLUE}▶ ${PID}${NC}"
    STEP_START=$(date +%s)

    if python3 scripts/06_erp_stroop.py "$PID" 2>&1 | \
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
echo -e "${BOLD}STEP 2: Group ERP analysis (07_group_erp_stroop.py)${NC}"
echo "------------------------------------------------------------"

STEP_START=$(date +%s)
set +e
python3 scripts/07_group_erp_stroop.py 2>&1 | \
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
echo -e "${BOLD}STEP 3: Statistical analysis (08_statistics_stroop.py)${NC}"
echo "------------------------------------------------------------"

STEP_START=$(date +%s)
set +e
python3 scripts/08_statistics_stroop.py 2>&1 | \
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
echo -e "${BOLD} STROOP PIPELINE COMPLETE — ${MINS}m ${SECS}s${NC}"
echo -e "${BOLD} Finished: $(date '+%Y-%m-%d %H:%M:%S')${NC}"
echo -e "${BOLD}============================================================${NC}"
echo ""
echo -e "  Participants passed  : ${GREEN}${#PASSED[@]} — ${PASSED[*]:-none}${NC}"
echo -e "  Participants skipped : ${YELLOW}${#SKIPPED[@]} — ${SKIPPED[*]:-none}${NC}"
echo -e "  Participants failed  : ${RED}${#FAILED[@]} — ${FAILED[*]:-none}${NC}"
echo ""
echo -e "  Individual ERPs : output/erp/P0X_stroop_*.png"
echo -e "                    output/erp/P0X_stroop_erp_components.csv"
echo -e "  Group plots     : output/group/group_stroop_*.png"
echo -e "  Group CSVs      : output/group/group_stroop_*.csv"
echo -e "  Statistics      : output/stats/stroop_statistics_primary.csv"
echo -e "                    output/stats/stroop_statistics_exploratory.csv"
echo -e "                    output/stats/stroop_statistics_sensitivity.csv"
echo ""
echo -e "  Outlier notes:"
echo -e "    P03 — high miss rate / reversed amplitude (included, flagged)"
echo -e "    P05 — low congruent trials n=25 (included, flagged)"
echo -e "    P08 — low congruent trials n=6  (included, flagged)"
echo ""
