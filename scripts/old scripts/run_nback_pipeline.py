#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
# run_nback_pipeline.py
#
# Cross-platform pipeline runner for the N-back ERP analysis.
# Works on Windows, Mac, and Linux without any bash dependency.
#
# Runs in sequence:
#   Step 1: 06_erp_nback.py       — per-participant ERP extraction
#   Step 2: 07_group_erp_nback.py  — group grand averages and plots
#   Step 3: 08_statistics_nback.py — statistical comparisons
#
# All participants included (P01–P08).
# Outlier flags recorded in output CSV — no participants excluded.
#
# Usage:
#   python run_nback_pipeline.py              # all participants
#   python run_nback_pipeline.py P01         # single participant
#   python run_nback_pipeline.py P01 P02 P04 # specific participants
#
# Requirements:
#   - Run from project root directory (where scripts/ and data/ folders are)
#   - Scripts 03, 04, 05 already run for all participants
#   - data/processed/P0X_nback_epo.fif exists for each participant
# ─────────────────────────────────────────────────────────────────────────────

import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime

# ── Colour codes (ANSI — work on Mac/Linux/Windows 10+) ──────────────────────
import os
os.system('')   # enables ANSI escape codes on Windows terminal

GREEN  = '\033[0;32m'
RED    = '\033[0;31m'
YELLOW = '\033[1;33m'
BLUE   = '\033[0;34m'
BOLD   = '\033[1m'
NC     = '\033[0m'

# ── All participants ──────────────────────────────────────────────────────────
ALL_PARTICIPANTS = ['P01','P02','P03','P04','P05','P06','P07','P08']
# P08 — target/hit n=5, target/miss n=11    (included, flagged)
# P04 — low SNR noisy baseline              (included, flagged)
# P05 — target/hit n=33 reversed sens       (included, flagged)
# P06 — target/hit n=32                     (included, flagged)
# P07 — target/miss n=59 high miss rate     (included, flagged)
# P03 — target/miss n=81 high miss rate     (included, flagged)

# ── Parse arguments ───────────────────────────────────────────────────────────
if len(sys.argv) > 1:
    PARTICIPANTS = sys.argv[1:]
    print(f"{YELLOW}Running for specified participants: {' '.join(PARTICIPANTS)}{NC}")
else:
    PARTICIPANTS = ALL_PARTICIPANTS
    print(f"{YELLOW}Running for all participants: {' '.join(PARTICIPANTS)}{NC}")

# ── Pre-flight: verify project root ──────────────────────────────────────────
project_root = Path.cwd()
if not (project_root / 'scripts').exists() or \
   not (project_root / 'data').exists():
    print(f"\n{RED}ERROR: Run this script from the project root directory.{NC}")
    print(f"{RED}       Expected location of scripts/ and data/ folders.{NC}")
    print(f"{RED}       Current directory: {project_root}{NC}")
    sys.exit(1)

scripts_dir = project_root / 'scripts'

# ── Helper: run a python script as subprocess ─────────────────────────────────
def run_script(script_name, args=None):
    """
    Run scripts/<script_name> with optional args.
    Streams output line by line to terminal.
    Returns (success: bool, elapsed_seconds: int).
    """
    script_path = scripts_dir / script_name
    if not script_path.exists():
        print(f"  {RED}ERROR: {script_path} not found{NC}")
        return False, 0

    cmd   = [sys.executable, str(script_path)] + (args or [])
    start = time.time()

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            cwd=str(project_root)
        )
        for line in proc.stdout:
            print(f"    {line}", end='')
        proc.wait()
        elapsed = int(time.time() - start)
        return proc.returncode == 0, elapsed

    except Exception as e:
        elapsed = int(time.time() - start)
        print(f"  {RED}EXCEPTION: {e}{NC}")
        return False, elapsed

# ── Pipeline start ────────────────────────────────────────────────────────────
pipeline_start = time.time()

print(f"\n{BOLD}============================================================{NC}")
print(f"{BOLD} NBACK ERP PIPELINE — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{NC}")
print(f"{BOLD}============================================================{NC}")
print(f"\n  Project root : {project_root}")
print(f"  Participants : {' '.join(PARTICIPANTS)}")
print(f"  Outputs      : output/erp/  output/group/  output/stats/")
print()

# ── Tracking ──────────────────────────────────────────────────────────────────
passed  = []
failed  = []
skipped = []

# ── STEP 1: Individual ERP extraction ────────────────────────────────────────
print(f"{BOLD}STEP 1: Individual ERP extraction (06_erp_nback.py){NC}")
print("------------------------------------------------------------")

for pid in PARTICIPANTS:

    epo_file = project_root / 'data' / 'processed' / f'{pid}_nback_epo.fif'

    if not epo_file.exists():
        print(f"  {YELLOW}SKIP{NC} {pid} — epoch file not found: {epo_file.name}")
        skipped.append(pid)
        continue

    print(f"\n  {BLUE}▶ {pid}{NC}")
    success, elapsed = run_script('06_erp_nback.py', args=[pid])

    if success:
        print(f"  {GREEN}✓ {pid} — completed in {elapsed}s{NC}")
        passed.append(pid)
    else:
        print(f"  {RED}✗ {pid} — FAILED after {elapsed}s{NC}")
        failed.append(pid)

print()
print("------------------------------------------------------------")
print(f"  Passed  : {GREEN}{len(passed)} — {' '.join(passed) or 'none'}{NC}")
print(f"  Skipped : {YELLOW}{len(skipped)} — {' '.join(skipped) or 'none'}{NC}")
print(f"  Failed  : {RED}{len(failed)} — {' '.join(failed) or 'none'}{NC}")
print()

if len(passed) == 0:
    print(f"{RED}ERROR: No participants completed Step 1.{NC}")
    print(f"{RED}       Skipping group and statistics steps.{NC}")
    sys.exit(1)

if failed:
    print(f"{YELLOW}WARNING: {len(failed)} participant(s) failed in Step 1.{NC}")
    print(f"{YELLOW}         Continuing group analysis with: {' '.join(passed)}{NC}")
    print()

# ── STEP 2: Group ERP analysis ────────────────────────────────────────────────
print(f"{BOLD}STEP 2: Group ERP analysis (07_group_erp_nback.py){NC}")
print("------------------------------------------------------------")

success, elapsed = run_script('07_group_erp_nback.py')

if success:
    print(f"{GREEN}✓ Group ERP complete — {elapsed}s{NC}")
else:
    print(f"{RED}✗ Group ERP FAILED after {elapsed}s{NC}")
    print(f"{RED}  Check output above. Continuing to statistics step.{NC}")
print()

# ── STEP 3: Statistics ────────────────────────────────────────────────────────
print(f"{BOLD}STEP 3: Statistical analysis (08_statistics_nback.py){NC}")
print("------------------------------------------------------------")

success, elapsed = run_script('08_statistics_nback.py')

if success:
    print(f"{GREEN}✓ Statistics complete — {elapsed}s{NC}")
else:
    print(f"{RED}✗ Statistics FAILED after {elapsed}s{NC}")
print()

# ── Final summary ─────────────────────────────────────────────────────────────
total = int(time.time() - pipeline_start)
mins  = total // 60
secs  = total  % 60

print(f"{BOLD}============================================================{NC}")
print(f"{BOLD} NBACK PIPELINE COMPLETE — {mins}m {secs}s{NC}")
print(f"{BOLD} Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{NC}")
print(f"{BOLD}============================================================{NC}")
print()
print(f"  Participants passed  : {GREEN}{len(passed)} — {' '.join(passed) or 'none'}{NC}")
print(f"  Participants skipped : {YELLOW}{len(skipped)} — {' '.join(skipped) or 'none'}{NC}")
print(f"  Participants failed  : {RED}{len(failed)} — {' '.join(failed) or 'none'}{NC}")
print()
print(f"  Individual ERPs : output/erp/P0X_nback_*.png")
print(f"                    output/erp/P0X_nback_erp_components.csv")
print(f"  Group plots     : output/group/group_nback_*.png")
print(f"  Group CSVs      : output/group/group_nback_*.csv")
print(f"  Statistics      : output/stats/nback_statistics_primary.csv")
print(f"                    output/stats/nback_statistics_exploratory.csv")
print(f"                    output/stats/nback_statistics_sensitivity.csv")
print()
print(f"  Outlier notes (all included, flagged in CSV):")
print(f"    P08 — target/hit n=5, target/miss n=11")
print(f"    P04 — low SNR noisy baseline")
print(f"    P05 — target/hit n=33, reversed condition sensitivity")
print(f"    P06 — target/hit n=32")
print(f"    P07 — target/miss n=59 high miss rate")
print(f"    P03 — target/miss n=81 high miss rate")
print()
