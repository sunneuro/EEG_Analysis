#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
# run_pipeline.py
#
# Cross-platform pipeline runner for the ERP analysis.
#
# Runs in sequence:
#   Step 1: 05_erp.py       — per-participant ERP extraction
#   Step 2: 06_group_erp.py  — group grand averages and plots
#   Step 3: 07_statistics.py — statistical comparisons
#   Step 4: 08_EEG_variation.py — variation analysis
#   Step 5: 09_behavioural.py — behavioral analysis
#
# Usage:
#   python run_pipeline.py --task nback              # all participants
#   python run_pipeline.py --task stroop             # all participants
#   python run_pipeline.py --task nback P01          # single participant
# ─────────────────────────────────────────────────────────────────────────────

import sys
import argparse
import subprocess
import time
from pathlib import Path
from datetime import datetime
import os

os.system('')   # enables ANSI escape codes on Windows terminal

GREEN  = '\033[0;32m'
RED    = '\033[0;31m'
YELLOW = '\033[1;33m'
BLUE   = '\033[0;34m'
BOLD   = '\033[1m'
NC     = '\033[0m'

ALL_PARTICIPANTS = ['P01','P02','P03','P04','P05','P06','P07','P08']

def run_script(script_name, args=None):
    script_path = Path('scripts') / script_name
    if not script_path.exists():
        print(f"  {RED}ERROR: {script_path} not found{NC}")
        return False, 0
    cmd = [sys.executable, str(script_path)] + (args or [])
    start = time.time()
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
        for line in proc.stdout:
            print(f"    {line}", end='')
        proc.wait()
        return proc.returncode == 0, int(time.time() - start)
    except Exception as e:
        print(f"  {RED}EXCEPTION: {e}{NC}")
        return False, int(time.time() - start)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['nback', 'stroop'], required=True)
    parser.add_argument('participants', nargs='*', default=ALL_PARTICIPANTS)
    args = parser.parse_args()

    task = args.task
    participants = args.participants

    print(f"\n{BOLD}============================================================{NC}")
    print(f"{BOLD} {task.upper()} ERP PIPELINE — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{NC}")
    print(f"{BOLD}============================================================{NC}\n")

    passed, failed, skipped = [], [], []

    # Step 1
    print(f"{BOLD}STEP 1: Individual ERP extraction (05_erp.py){NC}")
    print("------------------------------------------------------------")
    for pid in participants:
        epo_file = Path('data') / 'processed' / f'{pid}_{task}_epo.fif'
        if not epo_file.exists():
            print(f"  {YELLOW}SKIP{NC} {pid} — epoch file not found: {epo_file.name}")
            skipped.append(pid)
            continue
        print(f"\n  {BLUE}▶ {pid}{NC}")
        success, elapsed = run_script('05_erp.py', args=['--task', task, pid])
        if success:
            print(f"  {GREEN}✓ {pid} — completed in {elapsed}s{NC}")
            passed.append(pid)
        else:
            print(f"  {RED}✗ {pid} — FAILED after {elapsed}s{NC}")
            failed.append(pid)

    if not passed:
        print(f"{RED}ERROR: No participants completed Step 1.{NC}")
        sys.exit(1)

    # Step 2
    print(f"\n{BOLD}STEP 2: Group ERP analysis (06_group_erp.py){NC}")
    print("------------------------------------------------------------")
    success, elapsed = run_script('06_group_erp.py', args=['--task', task])
    if success: print(f"{GREEN}✓ Group ERP complete — {elapsed}s{NC}")
    else: print(f"{RED}✗ Group ERP FAILED after {elapsed}s{NC}")

    # Step 3
    print(f"\n{BOLD}STEP 3: Statistical analysis (07_statistics.py){NC}")
    print("------------------------------------------------------------")
    success, elapsed = run_script('07_statistics.py', args=['--task', task])
    if success: print(f"{GREEN}✓ Statistics complete — {elapsed}s{NC}")
    else: print(f"{RED}✗ Statistics FAILED after {elapsed}s{NC}")

    # Step 4
    print(f"\n{BOLD}STEP 4: EEG Variation (08_EEG_variation.py){NC}")
    print("------------------------------------------------------------")
    success, elapsed = run_script('08_EEG_variation.py', args=['--task', task])
    if success: print(f"{GREEN}✓ Variation complete — {elapsed}s{NC}")
    else: print(f"{RED}✗ Variation FAILED after {elapsed}s{NC}")

    # Step 5
    print(f"\n{BOLD}STEP 5: Behavioural (09_behavioural.py){NC}")
    print("------------------------------------------------------------")
    success, elapsed = run_script('09_behavioural.py', args=['--task', task])
    if success: print(f"{GREEN}✓ Behavioural complete — {elapsed}s{NC}")
    else: print(f"{RED}✗ Behavioural FAILED after {elapsed}s{NC}")

    print(f"\n{BOLD}============================================================{NC}")
    print(f"{BOLD} PIPELINE COMPLETE {NC}")
    print(f"{BOLD}============================================================{NC}\n")


if __name__ == '__main__':
    main()
