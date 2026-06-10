# scripts/01_parse_eprime_nback.py
# ─────────────────────────────────────────────────────────────────────────────
# STEP 0a OF EEG PIPELINE: Parse E-Prime N-back behavioural data
#
# Reads ALL E-Prime N-back .txt files in data/behavioural/nback/
# Joins participant metadata (group, age, sex) from data/participants.csv
# Saves one clean CSV per participant into data/behavioural/nback/
#
# E-Prime .txt file structure (4 header rows before data):
#   Row 0: data types     (STRING, INTEGER, ...)
#   Row 1: E-Prime roles  (EXPNAME, VARIABLE, ...)
#   Row 2: log levels     (1, 1, 2, 3, ...)
#   Row 3: column names   (ExperimentName, Subject, ...)  ← real header
#   Row 4+: data rows
#
# N-back specific columns:
#   Stimulus.RT  / Stimulus.ACC  (note: Stimulus not Stimulus3 unlike Stroop)
#   ListName     : ZeroList, OneList, TwoList, Prac0List, Prac1List, Prac2List
#   TrialType    : Target or Non-target
#   Letter       : stimulus letter presented
#   CorrectAnswers: correct response key
#
# File naming convention expected:
#   {participant_id}_nback.txt  e.g. P01_nback.txt
#
# Usage:
#   python3 scripts/01_parse_eprime_nback.py
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
nback_folder      = Path('data/behavioural/nback')
participants_file = Path('data/participants.csv')
# ─────────────────────────────────────────────────────────────────────────────

# ── Step 1: Load participants metadata ────────────────────────────────────────
if not participants_file.exists():
    print(f"ERROR: {participants_file} not found.")
    print("Create data/participants.csv first.")
    exit()

participants = pd.read_csv(participants_file)
print(f"Loaded participants.csv: {len(participants)} participants")
print(participants[['participant_id', 'name', 'group']].to_string(index=False))
print()

# ── Step 2: Find all N-back .txt files ───────────────────────────────────────
txt_files = sorted(nback_folder.glob('*.txt'))

if len(txt_files) == 0:
    print(f"No .txt files found in {nback_folder}")
    print("Make sure your E-Prime N-back files are copied there.")
    exit()

print(f"Found {len(txt_files)} file(s) to process:\n")

# ── Step 3: Loop over every file ──────────────────────────────────────────────
for input_file in txt_files:

    # Build output filename: P01_nback.txt → P01_nback_clean.csv
    output_file = nback_folder / (input_file.stem + '_clean.csv')

    # Skip already processed files
    if output_file.exists():
        print(f"  SKIP (already exists): {output_file.name}")
        continue

    print(f"  Processing: {input_file.name}")

    try:
        # ── Parse E-Prime file ────────────────────────────────────────────────

        # Read raw — no automatic header
        df_raw = pd.read_csv(
            input_file,
            sep='\t',
            encoding='utf-16',
            header=None,
            low_memory=False
        )

        # Row 3 = real column names, rows 4+ = data
        col_names = list(df_raw.iloc[3])
        df = df_raw.iloc[4:].copy()
        df.columns = col_names
        df = df.reset_index(drop=True)

        # ── Keep only trial-level rows ────────────────────────────────────────
        # Stimulus.RT is null/empty on block-level rows
        # RT = 0 means no response (omission) — keep, different from missing
        df = df[
            df['Stimulus.RT'].notna() &
            (df['Stimulus.RT'] != 'NULL') &
            (df['Stimulus.RT'] != '')
        ].copy().reset_index(drop=True)

        # ── Keep useful columns ───────────────────────────────────────────────
        keep = [
            'ExperimentName', 'Subject', 'Session',
            'SessionDate', 'SessionTime',
            'Running[Block]', 'Running[Trial]',
            'BlockList.Cycle', 'BlockList.Sample',
            'TrialList.Cycle', 'TrialList.Sample',
            'ListName',
            'Letter',
            'TrialType',
            'CorrectAnswers',
            'Stimulus.ACC',
            'Stimulus.CRESP',
            'Stimulus.RESP',
            'Stimulus.RT',
            'Stimulus.RTTime',
            'Stimulus.OnsetTime',
            'Stimulus.OnsetDelay',
            'Stimulus.OnsetToOnsetTime',
            'Procedure[Trial]',
            'Procedure[Block]',
        ]
        df = df[[c for c in keep if c in df.columns]]

        # ── Convert numeric columns ───────────────────────────────────────────
        df['Stimulus.RT']  = pd.to_numeric(df['Stimulus.RT'],  errors='coerce')
        df['Stimulus.ACC'] = pd.to_numeric(df['Stimulus.ACC'], errors='coerce')

        # ── Add missed response flag ──────────────────────────────────────────
        # RT = 0 means no key pressed within response window (omission)
        # Different from wrong key press (ACC=0 but RT > 0)
        df['missed'] = (df['Stimulus.RT'] == 0).astype(int)

        # ── Separate practice from experimental trials ────────────────────────
        practice_lists = ['Prac0List', 'Prac1List', 'Prac2List']
        df['is_practice'] = df['ListName'].isin(practice_lists).astype(int)

        # ── Join participant metadata from participants.csv ───────────────────
        # Extract participant_id from filename: P01_nback.txt → 'P01'
        participant_id = input_file.stem.split('_')[0]
        task           = 'nback'

        participant_row = participants[
            participants['participant_id'] == participant_id
        ]

        if len(participant_row) == 0:
            print(f"    WARNING: {participant_id} not found in participants.csv")
            print(f"    Group will be missing — add to participants.csv")
            df.insert(0, 'task',           task)
            df.insert(0, 'participant_id', participant_id)
        else:
            meta = participant_row.iloc[0]
            df.insert(0, 'task',           task)
            df.insert(0, 'group',          meta['group'])
            df.insert(0, 'participant_id', participant_id)
            if 'age' in meta.index and pd.notna(meta['age']):
                df.insert(3, 'age', meta['age'])
            if 'sex' in meta.index and pd.notna(meta['sex']):
                df.insert(4, 'sex', meta['sex'])

        # ── Save ──────────────────────────────────────────────────────────────
        df.to_csv(output_file, index=False)

        # ── Summary ───────────────────────────────────────────────────────────
        group_val = df['group'].iloc[0] if 'group' in df.columns else 'unknown'
        exp_trials = df[df['is_practice'] == 0]
        print(f"    participant_id : {participant_id}")
        print(f"    group          : {group_val}")
        print(f"    total rows     : {len(df)} (incl. practice)")
        print(f"    practice trials: {df['is_practice'].sum()}")
        print(f"    exp trials     : {len(exp_trials)}")
        print(f"    missed (exp)   : {exp_trials['missed'].sum()} "
              f"({exp_trials['missed'].mean():.1%})")
        print(f"    accuracy (exp) : {exp_trials['Stimulus.ACC'].mean():.1%}")
        rt_correct = exp_trials[
            (exp_trials['Stimulus.ACC'] == 1) & (exp_trials['missed'] == 0)
        ]['Stimulus.RT']
        print(f"    mean RT (exp)  : {rt_correct.mean():.1f} ms")

        # N-back level breakdown (experimental trials only)
        print(f"\n    N-back level breakdown (experimental trials):")
        exp_lists = [l for l in exp_trials['ListName'].unique()
                     if l not in practice_lists]
        for lst in sorted(exp_lists):
            sub = exp_trials[exp_trials['ListName'] == lst]
            acc = sub['Stimulus.ACC'].mean()
            rt  = sub[
                (sub['Stimulus.ACC'] == 1) & (sub['missed'] == 0)
            ]['Stimulus.RT'].mean()
            print(f"      {lst:12s}: n={len(sub):3d}  ACC={acc:.1%}  "
                  f"RT={rt:.1f}ms")

        print(f"\n    saved to: {output_file.name}\n")

    except Exception as e:
        print(f"    ERROR processing {input_file.name}: {e}\n")

print("Done.")
