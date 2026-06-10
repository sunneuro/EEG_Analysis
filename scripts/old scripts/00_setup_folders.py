#!/usr/bin/env python3
# ─────────────────────────────────────────────────────────────────────────────
# 00_setup_folders.py
#
# Cross-platform project folder setup and validation tool.
# Works on Windows, Mac, and Linux.
#
# Three modes:
#   [1] Create new project folder structure at a chosen location
#   [2] Check an existing project folder structure for completeness
#   [3] Exit
#
# IMPORTANT — output folders are NOT created here.
# They are created automatically by the pipeline scripts:
#   output/epochs/  — created by 05_epochs.py
#   output/erp/     — created by 06_erp_stroop.py / 06_erp_nback.py
#   output/group/   — created by 07_group_erp_stroop.py / 07_group_erp_nback.py
#   output/stats/   — created by 08_statistics_stroop.py / 08_statistics_nback.py
#
# Usage:
#   python 00_setup_folders.py
# ─────────────────────────────────────────────────────────────────────────────

import os
import sys
from pathlib import Path
from datetime import datetime

# ── Colour codes (ANSI — Windows 10+, Mac, Linux) ────────────────────────────
os.system('')   # enables ANSI on Windows terminal

GREEN  = '\033[0;32m'
RED    = '\033[0;31m'
YELLOW = '\033[1;33m'
BLUE   = '\033[0;34m'
CYAN   = '\033[0;36m'
BOLD   = '\033[1m'
NC     = '\033[0m'

# ── Input folder structure ────────────────────────────────────────────────────
# These are the folders this script creates and validates.
# Output folders (output/erp etc.) are created by later pipeline scripts.

INPUT_FOLDERS = [
    'scripts',
    'data',
    'data/raw',
    'data/behavioural',
    'data/behavioural/stroop',
    'data/behavioural/nback',
    'data/behavioural/edat_backup',
]

# Output folders — checked for existence in mode 2 but NEVER created here
OUTPUT_FOLDERS = [
    'data/processed',
    'output',
    'output/epochs',
    'output/erp',
    'output/group',
    'output/stats',
]

# Expected pipeline scripts
EXPECTED_SCRIPTS = [
    '01_parse_eprime_nback.py',
    '02_parse_eprime_stroop.py',
    '03_import_filter_eeg.py',
    '04_ica.py',
    '05_epochs.py',
    '06_erp_stroop.py',
    '06_erp_nback.py',
    '07_group_erp_stroop.py',
    '07_group_erp_nback.py',
    '08_statistics_stroop.py',
    '08_statistics_nback.py',
    'run_stroop_pipeline.py',
    'run_nback_pipeline.py',
]

# participants.csv required columns
PARTICIPANTS_CSV_COLUMNS = [
    'participant_id', 'name', 'group', 'age', 'sex'
]

PARTICIPANTS_CSV_TEMPLATE = \
"""participant_id,name,group,age,sex
P01,Participant One,creatine,24,M
P02,Participant Two,control,22,F
P03,Participant Three,creatine,23,F
P04,Participant Four,creatine,25,M
P05,Participant Five,control,24,F
P06,Participant Six,control,23,M
P07,Participant Seven,control,22,F
P08,Participant Eight,creatine,24,M
"""

# ── File type instructions per folder ─────────────────────────────────────────
FOLDER_INSTRUCTIONS = {
    'data/raw': {
        'title': 'Raw EEG files — BrainVision format',
        'types': ['.vhdr', '.vmrk', '.eeg'],
        'instructions': [
            'Each recording session produces EXACTLY THREE files:',
            '  .vhdr  — header file (text, contains recording parameters)',
            '  .vmrk  — marker file (text, contains trigger codes from E-Prime)',
            '  .eeg   — binary data file (large, contains raw EEG signal)',
            '',
            'All three files must have IDENTICAL base names, e.g.:',
            '  JD_Stroop_22_05_2026.vhdr',
            '  JD_Stroop_22_05_2026.vmrk',
            '  JD_Stroop_22_05_2026.eeg',
            '',
            'GOLDEN RULE: NEVER rename .vhdr, .vmrk, or .eeg files after',
            'recording. The .vhdr file contains internal references to the',
            'other two files by name. Renaming breaks these references and',
            'causes subtle, hard-to-detect errors.',
            '',
            'One set of three files per participant per task:',
            '  P01 stroop → JD_Stroop_22_05_2026.vhdr/.vmrk/.eeg',
            '  P01 nback  → JD_Nback_22_05_2026.vhdr/.vmrk/.eeg',
            '  P02 stroop → AB_Stroop_25_05_2026.vhdr/.vmrk/.eeg',
            '  ... and so on for all participants',
            '',
            'The .vhdr filename is what you pass to script 03:',
            '  python scripts/03_import_filter_eeg.py P01 stroop',
            '  JD_Stroop_22_05_2026.vhdr',
        ],
    },
    'data/behavioural/stroop': {
        'title': 'E-Prime Stroop behavioural files',
        'types': ['.txt'],
        'instructions': [
            'WHAT THE PIPELINE READS:',
            '  .txt files only — one per participant, named:',
            '  P01_stroop.txt',
            '  P02_stroop.txt',
            '  ... P08_stroop.txt',
            '',
            'UNDERSTANDING E-PRIME FILE TYPES:',
            '  E-Prime produces up to three file types per session:',
            '  .edat2  — binary proprietary format (E-Prime raw data)',
            '            Cannot be read directly by this pipeline.',
            '            Keep as backup — do not place in this folder.',
            '  .txt    — tab-separated text export, UTF-16 encoded.',
            '            THIS is what the pipeline reads.',
            '  .xlsx   — optional Excel export. Not used by this pipeline.',
            '',
            'DO I ALREADY HAVE A .txt FILE?',
            '  E-Prime can be configured to create a .txt file automatically',
            '  alongside the .edat2 file during the experiment session.',
            '  Check whether a .txt file already exists alongside your .edat2.',
            '  If yes — rename it to P0X_stroop.txt and place it here.',
            '  If no  — you must export it from E-DataAid (see below).',
            '',
            'HOW TO EXPORT .txt FROM E-DataAid (if not auto-created):',
            '  1. Open E-DataAid (installed with E-Prime)',
            '  2. File → Open → select your .edat2 file',
            '  3. File → Export → tab-delimited text',
            '  4. In the export dialog:',
            '       Format   : Tab-delimited text (.txt)',
            '       Encoding : Unicode (UTF-16)',
            '       Include  : All variables (do not filter)',
            '  5. Save as P0X_stroop.txt',
            '  6. Place in data/behavioural/stroop/',
            '',
            'WHERE TO KEEP .edat2 FILES:',
            '  Keep original .edat2 files as backup.',
            '  Recommended location: data/behavioural/edat_backup/',
            '  Do NOT place .edat2 files in data/behavioural/stroop/ —',
            '  the pipeline will ignore them but it creates confusion.',
            '',
            'REQUIRED E-PRIME COLUMNS (must be present in .txt):',
            '  Stimulus3.RT    — response time in milliseconds',
            '  Stimulus3.ACC   — accuracy (1=correct, 0=incorrect)',
            '  Congruency      — trial type (Congruent / Incongruent)',
            '  StimWord        — colour word presented',
            '  StimColor       — ink colour of word',
            '',
            'FILENAME RULES:',
            '  Use underscores only — no spaces in filenames.',
            '  P01_stroop.txt ✓     P01 stroop.txt ✗',
            '',
            'Run: python scripts/02_parse_eprime_stroop.py',
        ],
    },
    'data/behavioural/nback': {
        'title': 'E-Prime N-back behavioural files',
        'types': ['.txt'],
        'instructions': [
            'WHAT THE PIPELINE READS:',
            '  .txt files only — one per participant, named:',
            '  P01_nback.txt',
            '  P02_nback.txt',
            '  ... P08_nback.txt',
            '',
            'UNDERSTANDING E-PRIME FILE TYPES:',
            '  E-Prime produces up to three file types per session:',
            '  .edat2  — binary proprietary format (E-Prime raw data)',
            '            Cannot be read directly by this pipeline.',
            '            Keep as backup — do not place in this folder.',
            '  .txt    — tab-separated text export, UTF-16 encoded.',
            '            THIS is what the pipeline reads.',
            '  .xlsx   — optional Excel export. Not used by this pipeline.',
            '',
            'DO I ALREADY HAVE A .txt FILE?',
            '  E-Prime can be configured to create a .txt file automatically',
            '  alongside the .edat2 file during the experiment session.',
            '  Check whether a .txt file already exists alongside your .edat2.',
            '  If yes — rename it to P0X_nback.txt and place it here.',
            '  If no  — you must export it from E-DataAid (see below).',
            '',
            'HOW TO EXPORT .txt FROM E-DataAid (if not auto-created):',
            '  1. Open E-DataAid (installed with E-Prime)',
            '  2. File → Open → select your .edat2 file',
            '  3. File → Export → tab-delimited text',
            '  4. In the export dialog:',
            '       Format   : Tab-delimited text (.txt)',
            '       Encoding : Unicode (UTF-16)',
            '       Include  : All variables (do not filter)',
            '  5. Save as P0X_nback.txt',
            '  6. Place in data/behavioural/nback/',
            '',
            'WHERE TO KEEP .edat2 FILES:',
            '  Keep original .edat2 files as backup.',
            '  Recommended location: data/behavioural/edat_backup/',
            '  Do NOT place .edat2 files in data/behavioural/nback/ —',
            '  the pipeline will ignore them but it creates confusion.',
            '',
            'REQUIRED E-PRIME COLUMNS (must be present in .txt):',
            '  Stimulus.RT     — response time (note: NOT Stimulus3.RT)',
            '  Stimulus.ACC    — accuracy (1=correct, 0=incorrect)',
            '  ListName        — block type (ZeroList/OneList/TwoList/',
            '                    Prac0List/Prac1List/Prac2List)',
            '  TrialType       — Target or Non-target',
            '  Letter          — stimulus letter presented',
            '  CorrectAnswers  — correct response key',
            '',
            'FILENAME RULES:',
            '  Use underscores only — no spaces in filenames.',
            '  P01_nback.txt ✓     P01 nback.txt ✗',
            '',
            'Run: python scripts/01_parse_eprime_nback.py',
        ],
    },
    'data/participants.csv': {
        'title': 'Participants metadata file',
        'types': ['.csv'],
        'instructions': [
            'One CSV file listing all participants.',
            'Required columns:',
            '  participant_id  — P01, P02 ... P08',
            '  name            — participant name or initials',
            '  group           — creatine or control (lowercase)',
            '  age             — age in years',
            '  sex             — M or F',
            '',
            'Group assignments for this study:',
            '  Creatine : P01, P02, P03, P04',
            '  Control  : P05, P06, P07, P08',
            '',
            'A template file is created automatically in mode [1].',
            'Edit it with the correct participant details before running',
            'scripts 01 or 02.',
        ],
    },
    'scripts': {
        'title': 'Pipeline scripts',
        'types': ['.py'],
        'instructions': [
            'All pipeline scripts should be placed here.',
            'Required scripts:',
            '  01_parse_eprime_nback.py',
            '  02_parse_eprime_stroop.py',
            '  03_import_filter_eeg.py',
            '  04_ica.py',
            '  05_epochs.py',
            '  06_erp_stroop.py',
            '  06_erp_nback.py',
            '  07_group_erp_stroop.py',
            '  07_group_erp_nback.py',
            '  08_statistics_stroop.py',
            '  08_statistics_nback.py',
            '  run_stroop_pipeline.py',
            '  run_nback_pipeline.py',
            '',
            'Pipeline runner scripts (run_*.py) should be placed in the',
            'PROJECT ROOT — not inside scripts/ folder.',
        ],
    },
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def tick(msg):   print(f"  {GREEN}✓{NC} {msg}")
def cross(msg):  print(f"  {RED}✗{NC} {msg}")
def warn(msg):   print(f"  {YELLOW}⚠{NC} {msg}")
def info(msg):   print(f"  {BLUE}•{NC} {msg}")
def header(msg): print(f"\n{BOLD}{msg}{NC}")
def rule():      print("─" * 60)

def ask(prompt, default=None):
    """Ask user for input with optional default."""
    if default:
        full_prompt = f"{prompt}\n  [{default}] > "
    else:
        full_prompt = f"{prompt}\n  > "
    response = input(full_prompt).strip()
    return response if response else (default or '')

def ask_path(prompt):
    """Ask user for a folder path, handle both Windows and Unix styles."""
    raw = ask(prompt, default=str(Path.cwd()))
    # Normalise path separators
    p = Path(raw.strip('"').strip("'"))
    return p

def print_folder_instructions(folder_key):
    """Print detailed file type instructions for a folder."""
    if folder_key not in FOLDER_INSTRUCTIONS:
        return
    cfg = FOLDER_INSTRUCTIONS[folder_key]
    print(f"\n  {CYAN}{cfg['title']}{NC}")
    print(f"  Expected file types: {BOLD}{', '.join(cfg['types'])}{NC}")
    for line in cfg['instructions']:
        if line == '':
            print()
        else:
            print(f"    {line}")

# ── Mode 1: Create new folder structure ──────────────────────────────────────
def mode_create():
    header("MODE 1 — Create new project folder structure")
    rule()

    # Ask for location
    print(f"\nWhere would you like to create the project folder?")
    print(f"Examples:")
    print(f"  Mac/Linux : /Users/SK/eeg_analysis")
    print(f"  Windows   : C:\\Users\\SK\\eeg_analysis")
    project_root = ask_path(
        "\nEnter full path for new project folder",
    )

    print(f"\n  Project will be created at: {BOLD}{project_root}{NC}")
    confirm = ask("Proceed? (y/n)", default='y').lower()
    if confirm != 'y':
        print(f"\n{YELLOW}Cancelled.{NC}")
        return

    print()
    header("Creating folder structure")
    rule()

    # Create input folders
    all_ok = True
    for folder in INPUT_FOLDERS:
        target = project_root / folder
        try:
            target.mkdir(parents=True, exist_ok=True)
            tick(f"Created : {folder}/")
        except Exception as e:
            cross(f"FAILED  : {folder}/  — {e}")
            all_ok = False

    # Create participants.csv template if missing
    csv_path = project_root / 'data' / 'participants.csv'
    if not csv_path.exists():
        try:
            csv_path.write_text(PARTICIPANTS_CSV_TEMPLATE, encoding='utf-8')
            tick(f"Created : data/participants.csv  (template — edit before running)")
        except Exception as e:
            cross(f"FAILED  : data/participants.csv — {e}")
    else:
        warn(f"Exists  : data/participants.csv — not overwritten")

    # Note about output folders
    print()
    warn("Output folders are NOT created here.")
    info("output/epochs/  — created automatically by 05_epochs.py")
    info("output/erp/     — created automatically by 06_erp_stroop/nback.py")
    info("output/group/   — created automatically by 07_group_erp_stroop/nback.py")
    info("output/stats/   — created automatically by 08_statistics_stroop/nback.py")
    info("data/processed/ — created automatically by 03_import_filter_eeg.py")

    # ── Script copying ────────────────────────────────────────────────────────
    # 00_setup_folders.py is always run from the folder that contains it.
    # That folder may also contain the other pipeline scripts.
    # If it does — offer to copy them automatically.
    # If it does not — tell the user explicitly what they need to copy.
    print()
    header("Pipeline scripts")
    rule()

    import shutil

    # The folder containing THIS script is where we look for pipeline scripts
    this_script_dir = Path(__file__).resolve().parent

    # Scripts that go into scripts/ subfolder
    scripts_to_copy = [s for s in EXPECTED_SCRIPTS
                       if not s.startswith('run_')]
    # Runner scripts that go into project root
    runners_to_copy = [s for s in EXPECTED_SCRIPTS
                       if s.startswith('run_')]

    # Check which scripts are present alongside 00_setup_folders.py
    found_scripts  = [s for s in scripts_to_copy
                      if (this_script_dir / s).exists()]
    found_runners  = [s for s in runners_to_copy
                      if (this_script_dir / s).exists()]
    found_setup    = (this_script_dir / '00_setup_folders.py').exists()

    all_found      = found_scripts + found_runners
    missing        = [s for s in scripts_to_copy  if s not in found_scripts] + \
                     [s for s in runners_to_copy if s not in found_runners]

    if all_found:
        print(f"\n  {GREEN}Found {len(all_found)} pipeline script(s) alongside "
              f"00_setup_folders.py{NC}")
        print(f"  Source: {this_script_dir}")
        print()

        # Ask user if they want to copy automatically
        copy_choice = ask(
            f"  Copy them to the correct locations automatically? (y/n)",
            default='y'
        ).lower()

        if copy_choice == 'y':
            scripts_dest = project_root / 'scripts'
            root_dest    = project_root
            copied       = []
            copy_failed  = []

            for script in found_scripts:
                src = this_script_dir / script
                dst = scripts_dest / script
                # Don't overwrite if already identical location
                if src.resolve() == dst.resolve():
                    tick(f"Already in place : scripts/{script}")
                    continue
                try:
                    shutil.copy2(str(src), str(dst))
                    tick(f"Copied → scripts/{script}")
                    copied.append(script)
                except Exception as e:
                    cross(f"FAILED  : {script} — {e}")
                    copy_failed.append(script)

            for runner in found_runners:
                src = this_script_dir / runner
                dst = root_dest / runner
                if src.resolve() == dst.resolve():
                    tick(f"Already in place : {runner}  (project root)")
                    continue
                try:
                    shutil.copy2(str(src), str(dst))
                    tick(f"Copied → {runner}  (project root)")
                    copied.append(runner)
                except Exception as e:
                    cross(f"FAILED  : {runner} — {e}")
                    copy_failed.append(runner)

            # Copy 00_setup_folders.py itself to root if not already there
            setup_src = this_script_dir / '00_setup_folders.py'
            setup_dst = root_dest / '00_setup_folders.py'
            if setup_src.resolve() != setup_dst.resolve():
                try:
                    shutil.copy2(str(setup_src), str(setup_dst))
                    tick(f"Copied → 00_setup_folders.py  (project root)")
                except Exception as e:
                    warn(f"Could not copy 00_setup_folders.py: {e}")

            print()
            if copied:
                print(f"  {GREEN}✓ {len(copied)} script(s) copied successfully.{NC}")
            if copy_failed:
                print(f"  {RED}✗ {len(copy_failed)} script(s) failed to copy — "
                      f"copy manually:{NC}")
                for f in copy_failed:
                    print(f"    {RED}•{NC} {f}")

            if missing:
                print()
                warn(f"{len(missing)} expected script(s) not found alongside "
                     f"00_setup_folders.py:")
                for m in missing:
                    dest = 'scripts/' if not m.startswith('run_') else 'project root'
                    warn(f"  {m}  →  {dest}")
                print(f"\n  {YELLOW}Copy these manually when you have them.{NC}")
        else:
            # User declined auto-copy — print manual instructions
            print()
            info("Manual copy instructions:")
            print(f"\n  Copy these into {BOLD}{project_root / 'scripts'}/{NC}:")
            for s in scripts_to_copy:
                marker = f"  {GREEN}✓{NC}" if s in found_scripts else f"  {YELLOW}?{NC}"
                print(f"  {marker} {s}")
            print(f"\n  Copy these into {BOLD}{project_root}/{NC} (project root):")
            for r in runners_to_copy:
                marker = f"  {GREEN}✓{NC}" if r in found_runners else f"  {YELLOW}?{NC}"
                print(f"  {marker} {r}")
            print(f"    {GREEN}✓{NC} 00_setup_folders.py")

    else:
        # No scripts found alongside 00_setup_folders.py
        # User must have downloaded/copied only this one file
        print(f"\n  {YELLOW}No pipeline scripts found alongside 00_setup_folders.py.{NC}")
        print(f"  This is fine if you only copied this setup script so far.")
        print(f"\n  {BOLD}You will need to copy the following scripts manually:{NC}")
        print()
        print(f"  {BOLD}Into: {project_root / 'scripts'}/ {NC}")
        for s in scripts_to_copy:
            print(f"    • {s}")
        print()
        print(f"  {BOLD}Into: {project_root}/  (project root){NC}")
        for r in runners_to_copy:
            print(f"    • {r}")
        print(f"    • 00_setup_folders.py  (this file)")
        print()
        print(f"  {CYAN}Tip: If you downloaded from GitHub, the repository contains{NC}")
        print(f"  {CYAN}all scripts. Copy the entire repository contents into:{NC}")
        print(f"  {CYAN}{project_root}{NC}")
        print(f"  {CYAN}then re-run: python 00_setup_folders.py{NC}")


    print(f"""
  {BOLD}1. Pipeline scripts{NC}
     If auto-copy ran successfully above — already done.
     If any scripts were missing or copy failed, copy them manually:
       Scripts 01–08 : {project_root / 'scripts'}/
       Runner scripts: {project_root}/run_stroop_pipeline.py
                       {project_root}/run_nback_pipeline.py
                       {project_root}/00_setup_folders.py

  {BOLD}2. Copy raw EEG files into:{NC}
     {project_root / 'data' / 'raw'}/
     — Three files per participant per task:
       .vhdr  .vmrk  .eeg  (all same base name, never rename)

  {BOLD}3. Copy E-Prime behavioural files into:{NC}

     {BOLD}Step 3a — Check if .txt files already exist:{NC}
     E-Prime may have auto-created a .txt file alongside the .edat2
     during the experiment session. Check the folder where your
     .edat2 files are stored. If a matching .txt exists — use it.

     {BOLD}Step 3b — If .txt files do not exist, export from E-DataAid:{NC}
     1. Open E-DataAid (installed with E-Prime)
     2. File → Open → select the participant's .edat2 file
     3. File → Export → Tab-delimited text
     4. Settings: Format=Tab-delimited, Encoding=Unicode (UTF-16),
        Include=All variables
     5. Save as P0X_stroop.txt or P0X_nback.txt

     {BOLD}Step 3c — Place .txt files here (one per participant):{NC}
     {project_root / 'data' / 'behavioural' / 'stroop'}/
       P01_stroop.txt  P02_stroop.txt  ...  P08_stroop.txt

     {project_root / 'data' / 'behavioural' / 'nback'}/
       P01_nback.txt   P02_nback.txt   ...  P08_nback.txt

     {BOLD}Step 3d — Store .edat2 backup files here (do not mix with .txt):{NC}
     {project_root / 'data' / 'behavioural' / 'edat_backup'}/
       All original .edat2 files — pipeline does not read these
       but keep them as backup in case re-export is needed

  {BOLD}4. Edit participants.csv:{NC}
     {csv_path}
     — Fill in correct participant names (group assignments already set)

  {BOLD}5. Run the pipeline from the project root:{NC}
     cd {project_root}
     python run_stroop_pipeline.py
     python run_nback_pipeline.py

  {BOLD}6. For detailed file format instructions, run:{NC}
     python 00_setup_folders.py  → choose option [3] Instructions
""")

    if all_ok:
        print(f"{GREEN}{BOLD}✓ Folder structure created successfully.{NC}")
    else:
        print(f"{YELLOW}{BOLD}⚠ Some folders could not be created — check errors above.{NC}")

# ── Mode 2: Check existing folder structure ───────────────────────────────────
def mode_check():
    header("MODE 2 — Check existing project folder structure")
    rule()

    print(f"\nWhere is your existing project folder?")
    print(f"Examples:")
    print(f"  Mac/Linux : /Users/SK/eeg_analysis")
    print(f"  Windows   : C:\\Users\\SK\\eeg_analysis")
    project_root = ask_path(
        "\nEnter full path of existing project folder",
    )

    if not project_root.exists():
        cross(f"Path does not exist: {project_root}")
        return

    print(f"\n  Checking: {BOLD}{project_root}{NC}\n")

    issues   = []
    warnings = []

    # ── 1. Input folders ──────────────────────────────────────────────────────
    header("1. Input folders")
    rule()
    for folder in INPUT_FOLDERS:
        target = project_root / folder
        if target.exists():
            n_files = len(list(target.iterdir()))
            if n_files == 0:
                warn(f"Empty   : {folder}/  (no files yet)")
                warnings.append(f"{folder}/ is empty")
            else:
                tick(f"Found   : {folder}/  ({n_files} items)")
        else:
            cross(f"Missing : {folder}/")
            issues.append(f"Missing input folder: {folder}/")

    # ── 2. Output folders ─────────────────────────────────────────────────────
    header("2. Output folders (created by pipeline scripts)")
    rule()
    for folder in OUTPUT_FOLDERS:
        target = project_root / folder
        if target.exists():
            n_files = len(list(target.iterdir()))
            tick(f"Found   : {folder}/  ({n_files} items)")
        else:
            info(f"Not yet : {folder}/  — will be created by pipeline")

    # ── 3. participants.csv ───────────────────────────────────────────────────
    header("3. participants.csv")
    rule()
    csv_path = project_root / 'data' / 'participants.csv'
    if csv_path.exists():
        tick(f"Found   : data/participants.csv")
        try:
            import csv
            with open(csv_path, encoding='utf-8') as f:
                reader    = csv.DictReader(f)
                rows      = list(reader)
                cols      = reader.fieldnames or []
                missing_cols = [c for c in PARTICIPANTS_CSV_COLUMNS if c not in cols]
                if missing_cols:
                    cross(f"Missing columns: {missing_cols}")
                    issues.append(f"participants.csv missing columns: {missing_cols}")
                else:
                    tick(f"Columns : {', '.join(PARTICIPANTS_CSV_COLUMNS)}")
                print(f"\n  Participants found ({len(rows)}):")
                for row in rows:
                    grp = row.get('group','?')
                    grp_col = GREEN if grp=='creatine' else BLUE if grp=='control' else YELLOW
                    print(f"    {grp_col}●{NC} {row.get('participant_id','?'):5s}  "
                          f"{row.get('name','?'):20s}  "
                          f"group={grp_col}{grp}{NC}  "
                          f"age={row.get('age','?')}  "
                          f"sex={row.get('sex','?')}")
        except Exception as e:
            warn(f"Could not parse participants.csv: {e}")
    else:
        cross(f"Missing : data/participants.csv")
        issues.append("data/participants.csv not found")

    # ── 4. Raw EEG files ──────────────────────────────────────────────────────
    header("4. Raw EEG files (data/raw/)")
    rule()
    raw_dir = project_root / 'data' / 'raw'
    if raw_dir.exists():
        vhdr_files = sorted(raw_dir.glob('*.vhdr'))
        vmrk_files = sorted(raw_dir.glob('*.vmrk'))
        eeg_files  = sorted(raw_dir.glob('*.eeg'))

        if not vhdr_files:
            warn("No .vhdr files found in data/raw/")
            warnings.append("No raw EEG files found")
        else:
            print(f"\n  {len(vhdr_files)} recording(s) found:\n")
            for vhdr in vhdr_files:
                stem  = vhdr.stem
                has_vmrk = (raw_dir / f"{stem}.vmrk").exists()
                has_eeg  = (raw_dir / f"{stem}.eeg").exists()

                if has_vmrk and has_eeg:
                    tick(f"{stem}")
                    print(f"       .vhdr ✓   .vmrk ✓   .eeg ✓")
                else:
                    cross(f"{stem}  — INCOMPLETE SET")
                    if not has_vmrk:
                        print(f"       .vhdr ✓   {RED}.vmrk ✗{NC}   ", end='')
                        issues.append(f"Missing .vmrk for {stem}")
                    else:
                        print(f"       .vhdr ✓   .vmrk ✓   ", end='')
                    if not has_eeg:
                        print(f"{RED}.eeg ✗{NC}")
                        issues.append(f"Missing .eeg for {stem}")
                    else:
                        print(f".eeg ✓")

        # Check for orphan .vmrk or .eeg without matching .vhdr
        vhdr_stems = {v.stem for v in vhdr_files}
        for vmrk in vmrk_files:
            if vmrk.stem not in vhdr_stems:
                warn(f"Orphan .vmrk (no matching .vhdr): {vmrk.name}")
                warnings.append(f"Orphan .vmrk: {vmrk.name}")
        for eeg in eeg_files:
            if eeg.stem not in vhdr_stems:
                warn(f"Orphan .eeg (no matching .vhdr): {eeg.name}")
                warnings.append(f"Orphan .eeg: {eeg.name}")

        # File type reminder
        print(f"""
  {CYAN}Raw EEG file instructions:{NC}
    Each recording requires THREE files with identical base names:
      .vhdr — header (recording parameters and channel info)
      .vmrk — markers (trigger codes from E-Prime, stimulus timings)
      .eeg  — binary data (the actual EEG signal — usually largest file)

    {YELLOW}GOLDEN RULE: Never rename these files after recording.{NC}
    The .vhdr contains internal references to .vmrk and .eeg by name.
    Renaming any file breaks these references and corrupts the import.

    Naming convention used in this study:
      Initials_Task_DD_MM_YYYY.vhdr/.vmrk/.eeg
      e.g. JD_Stroop_22_05_2026.vhdr

    Pass the .vhdr filename to script 03:
      python scripts/03_import_filter_eeg.py P01 stroop JD_Stroop_22_05_2026.vhdr
""")
    else:
        cross("data/raw/ folder missing")
        issues.append("data/raw/ folder missing")

    # ── 5. Behavioural files ──────────────────────────────────────────────────
    header("5. Behavioural files")
    rule()

    # Check edat_backup folder
    edat_backup = project_root / 'data' / 'behavioural' / 'edat_backup'
    print(f"\n  {BOLD}data/behavioural/edat_backup/{NC}  (recommended backup location)")
    if edat_backup.exists():
        edat_files = sorted(edat_backup.glob('*.edat2')) + \
                     sorted(edat_backup.glob('*.edat'))
        if edat_files:
            tick(f"Found {len(edat_files)} .edat/.edat2 backup file(s)")
            for ef in edat_files:
                info(f"  {ef.name}")
        else:
            warn("edat_backup/ exists but no .edat/.edat2 files found")
    else:
        info("edat_backup/ not created yet — recommended for storing")
        info("original .edat2 files as backup alongside exported .txt files")

    for task in ['stroop', 'nback']:
        behav_dir = project_root / 'data' / 'behavioural' / task
        ext       = task
        script_n  = '02' if task == 'stroop' else '01'
        rt_col    = 'Stimulus3.RT' if task == 'stroop' else 'Stimulus.RT'
        key_cols  = ('Congruency, StimWord, StimColor' if task == 'stroop'
                     else 'ListName, TrialType, Letter')

        print(f"\n  {BOLD}data/behavioural/{task}/{NC}")

        if behav_dir.exists():

            txt_files  = sorted(behav_dir.glob('*.txt'))
            edat_files = sorted(behav_dir.glob('*.edat2')) + \
                         sorted(behav_dir.glob('*.edat'))

            # Warn if .edat files placed here instead of backup folder
            if edat_files:
                warn(f"{len(edat_files)} .edat/.edat2 file(s) found here — "
                     f"move to data/behavioural/edat_backup/ (pipeline cannot read them)")
                for ef in edat_files:
                    warn(f"  {ef.name}")
                warnings.append(
                    f".edat files in {task}/ — move to edat_backup/")

            # Show .txt files
            raw_txts = [t for t in txt_files if 'clean' not in t.stem]
            if not raw_txts:
                cross(f"No .txt files found")
                issues.append(
                    f"No .txt files in data/behavioural/{task}/")
                print(f"""
    {YELLOW}No .txt files found. You need one per participant.{NC}

    {CYAN}Do you already have a .txt file?{NC}
      E-Prime may have created a .txt file automatically alongside
      the .edat2 file during the experiment session.
      Check the folder where your .edat2 files are stored — a matching
      .txt file may already exist there.
      If yes: rename it to P0X_{ext}.txt and copy it here.

    {CYAN}If no .txt file exists — export from E-DataAid:{NC}
      1. Open E-DataAid (installed with E-Prime)
      2. File → Open → select your .edat2 file
      3. File → Export → Tab-delimited text
      4. Export dialog settings:
           Format   : Tab-delimited text (.txt)
           Encoding : Unicode (UTF-16)  ← critical — wrong encoding breaks import
           Include  : All variables (do not filter columns)
      5. Save as P0X_{ext}.txt
      6. Copy to data/behavioural/{task}/

    {CYAN}Required columns in the .txt file:{NC}
      {rt_col}, {'Stimulus3' if task=='stroop' else 'Stimulus'}.ACC,
      {key_cols}
""")
            else:
                for txt in raw_txts:
                    has_clean = (behav_dir / f"{txt.stem}_clean.csv").exists()
                    status    = "(parsed ✓)" if has_clean else "(not yet parsed)"
                    col       = GREEN if has_clean else YELLOW
                    print(f"    {col}●{NC}  {txt.name}  {col}{status}{NC}")

                # Check for any expected participant files that are missing
                expected = [f"P0{i}_{ext}.txt" for i in range(1, 9)]
                missing  = [e for e in expected
                            if not (behav_dir / e).exists()]
                if missing:
                    warn(f"Missing expected files: {', '.join(missing)}")
                    warnings.append(
                        f"Missing {task} .txt files: {', '.join(missing)}")

                print(f"""
    {CYAN}E-Prime {task} file notes:{NC}
      Pipeline reads  : .txt files only (NOT .edat/.edat2)
      File naming     : P01_{ext}.txt ... P08_{ext}.txt (underscores, no spaces)
      Format required : Tab-delimited, UTF-16 encoding

      {CYAN}If a .txt is missing for any participant:{NC}
        Check if E-Prime auto-created one alongside the .edat2 file.
        If not, export from E-DataAid:
          File → Export → Tab-delimited text → Encoding: Unicode (UTF-16)

      .edat2 backup location : data/behavioural/edat_backup/
      Run script             : python scripts/{script_n}_parse_eprime_{ext}.py
""")
        else:
            cross(f"data/behavioural/{task}/ folder missing")
            issues.append(f"data/behavioural/{task}/ folder missing")

    # ── 6. Processed files ────────────────────────────────────────────────────
    header("6. Processed files (data/processed/)")
    rule()
    proc_dir = project_root / 'data' / 'processed'
    if proc_dir.exists():
        all_pids = [f'P0{i}' for i in range(1, 9)]
        print(f"\n  {'PID':5s}  {'stroop_filtered':>18}  {'stroop_ica':>12}  "
              f"{'stroop_epo':>12}  {'nback_filtered':>16}  "
              f"{'nback_ica':>11}  {'nback_epo':>11}")
        print(f"  {'─'*110}")

        for pid in all_pids:
            def chk(fname):
                return f"{GREEN}✓{NC}" if (proc_dir/fname).exists() else f"{RED}✗{NC}"
            print(f"  {pid:5s}  "
                  f"{'stroop_filtered':>18} {chk(f'{pid}_stroop_filtered_raw.fif')}  "
                  f"{'stroop_ica':>12} {chk(f'{pid}_stroop_ica_raw.fif')}  "
                  f"{'stroop_epo':>12} {chk(f'{pid}_stroop_epo.fif')}  "
                  f"{'nback_filtered':>16} {chk(f'{pid}_nback_filtered_raw.fif')}  "
                  f"{'nback_ica':>11} {chk(f'{pid}_nback_ica_raw.fif')}  "
                  f"{'nback_epo':>11} {chk(f'{pid}_nback_epo.fif')}")
    else:
        info("data/processed/ not yet created — will be created by script 03")

    # ── 7. Scripts ────────────────────────────────────────────────────────────
    header("7. Pipeline scripts (scripts/)")
    rule()
    scripts_dir = project_root / 'scripts'
    if scripts_dir.exists():
        print()
        for script in EXPECTED_SCRIPTS:
            spath = scripts_dir / script
            if spath.exists():
                tick(f"Found   : scripts/{script}")
            else:
                cross(f"Missing : scripts/{script}")
                issues.append(f"Missing script: {script}")

        # Check runner scripts in root
        print()
        for runner in ['run_stroop_pipeline.py', 'run_nback_pipeline.py']:
            rpath = project_root / runner
            if rpath.exists():
                tick(f"Found   : {runner}  (project root)")
            else:
                warn(f"Missing : {runner}  (should be in project root)")
                warnings.append(f"Runner script {runner} not in project root")
    else:
        cross("scripts/ folder missing")
        issues.append("scripts/ folder missing")

    # ── 8. Summary ────────────────────────────────────────────────────────────
    header("SUMMARY")
    rule()
    print()

    if not issues and not warnings:
        print(f"  {GREEN}{BOLD}✓ READY — all folders and files present.{NC}")
        print(f"  Pipeline can be run from: {project_root}")
        print(f"\n  Start with:")
        print(f"    cd {project_root}")
        print(f"    python run_stroop_pipeline.py")
        print(f"    python run_nback_pipeline.py")

    elif issues:
        print(f"  {RED}{BOLD}✗ NOT READY — {len(issues)} issue(s) must be resolved:{NC}")
        for i, issue in enumerate(issues, 1):
            print(f"    {RED}{i}.{NC} {issue}")
        if warnings:
            print(f"\n  {YELLOW}{BOLD}⚠ {len(warnings)} warning(s):{NC}")
            for w in warnings:
                print(f"    {YELLOW}•{NC} {w}")

    else:
        print(f"  {YELLOW}{BOLD}⚠ PARTIAL — folders present but {len(warnings)} warning(s):{NC}")
        for w in warnings:
            print(f"    {YELLOW}•{NC} {w}")
        print(f"\n  Pipeline may be partially runnable.")
        print(f"  Resolve warnings above before running affected steps.")

    print()

# ── Main menu ─────────────────────────────────────────────────────────────────
def main():
    print(f"\n{BOLD}{'='*60}{NC}")
    print(f"{BOLD} EEG ANALYSIS PIPELINE — Folder Setup{NC}")
    print(f"{BOLD} {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{NC}")
    print(f"{BOLD}{'='*60}{NC}")
    print(f"""
  What would you like to do?

  {BOLD}[1]{NC} Create new project folder structure at a location I choose
  {BOLD}[2]{NC} Check an existing project folder structure
  {BOLD}[3]{NC} Show file type instructions for all folders
  {BOLD}[4]{NC} Exit
""")

    choice = ask("Enter choice (1/2/3/4)", default='1')

    if choice == '1':
        mode_create()

    elif choice == '2':
        mode_check()

    elif choice == '3':
        header("File type instructions — all input folders")
        rule()
        for folder_key in FOLDER_INSTRUCTIONS:
            print_folder_instructions(folder_key)
            print()

    elif choice == '4':
        print(f"\n{YELLOW}Exited.{NC}\n")
        sys.exit(0)

    else:
        print(f"\n{RED}Invalid choice. Run again and enter 1, 2, 3, or 4.{NC}\n")
        sys.exit(1)

    print()

if __name__ == '__main__':
    main()
