# scripts/02_parse_eprime_stroop.py
# ─────────────────────────────────────────────────────────────────────────────
# Reads ALL E-Prime Stroop .txt files in data/behavioural/stroop/
# Joins participant metadata from data/participants.csv
# Saves one clean CSV per participant into data/behavioural/stroop/
#
# Key Stroop-specific columns:
#   Stimulus3.RT  / Stimulus3.ACC  (note: NOT Stimulus.RT like N-back)
#   Congruency    : 'Incongruent' or 'congruent' (standardised to title case)
#   StimWord      : the written colour word (Red, Green, Blue, Yellow)
#   StimColor     : the ink colour the word is printed in
#   Stimulus3.CRESP : correct response key
#   Stimulus3.RESP  : participant's actual response key
#
# Response keys (Chronos box):
#   1 = Red, 2 = Green, 3 = Blue (infer from CRESP values in your data)
#
# Usage:
#   python3 scripts/02_parse_eprime_stroop.py
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
stroop_folder     = Path('data/behavioural/stroop')
participants_file = Path('data/participants.csv')
# ─────────────────────────────────────────────────────────────────────────────

# ── Step 1: load participants metadata ───────────────────────────────────────
if not participants_file.exists():
    print(f"ERROR: {participants_file} not found.")
    exit()

participants = pd.read_csv(participants_file)
print(f"Loaded participants.csv: {len(participants)} participants")
print(participants[['participant_id', 'name', 'group']].to_string(index=False))
print()

# ── Step 2: find all Stroop .txt files ───────────────────────────────────────
txt_files = sorted(stroop_folder.glob('*.txt'))

if len(txt_files) == 0:
    print(f"No .txt files found in {stroop_folder}")
    exit()

print(f"Found {len(txt_files)} file(s) to process:\n")

# ── Step 3: loop over every file ─────────────────────────────────────────────
for input_file in txt_files:

    output_file = stroop_folder / (input_file.stem + '_clean.csv')

    if output_file.exists():
        print(f"  SKIP (already exists): {output_file.name}")
        continue

    print(f"  Processing: {input_file.name}")

    try:
        # ── Parse E-Prime file ────────────────────────────────────────────────

        # Same 4-row header structure as N-back
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

        # ── Keep trial-level rows ─────────────────────────────────────────────
        # Stroop uses Stimulus3.RT (not Stimulus.RT)
        # RT = 0 means no response — keep these, they are missed trials
        # We only drop rows where Stimulus3.RT is completely absent (block rows)
        df = df[
            df['Stimulus3.RT'].notna() &
            (df['Stimulus3.RT'] != 'NULL') &
            (df['Stimulus3.RT'] != '')
        ].copy().reset_index(drop=True)

        # ── Keep useful columns ───────────────────────────────────────────────
        keep = [
            'ExperimentName', 'Subject', 'Session',
            'SessionDate', 'SessionTime',
            'Block',
            'Running',
            'StroopBlockList.Cycle',
            'StroopBlockList.Sample',
            'Congruency',
            'StimWord',
            'StimColor',
            'Correct',
            'Stimulus3.ACC',
            'Stimulus3.CRESP',
            'Stimulus3.RESP',
            'Stimulus3.RT',
            'Stimulus3.RTTime',
            'Stimulus3.OnsetTime',
            'Stimulus3.OnsetDelay',
            'Stimulus3.OnsetToOnsetTime',
            'Procedure',
        ]
        df = df[[c for c in keep if c in df.columns]]

        # ── Convert numeric columns ───────────────────────────────────────────
        df['Stimulus3.RT']  = pd.to_numeric(df['Stimulus3.RT'],  errors='coerce')
        df['Stimulus3.ACC'] = pd.to_numeric(df['Stimulus3.ACC'], errors='coerce')

        # ── Standardise Congruency capitalisation ─────────────────────────────
        # Raw data has 'Incongruent' and 'congruent' — inconsistent capitalisation
        # Standardise to title case: 'Incongruent' and 'Congruent'
        if 'Congruency' in df.columns:
            df['Congruency'] = df['Congruency'].str.strip().str.title()
            # Result: 'Incongruent' and 'Congruent' — consistent

        # ── Add missed response flag ──────────────────────────────────────────
        # RT = 0 means no key was pressed within the response window
        # This is different from a wrong key press (ACC=0 but RT > 0)
        # Distinguishing these matters for signal detection theory analysis
        df['missed'] = (df['Stimulus3.RT'] == 0).astype(int)

        # ── Join participant metadata ─────────────────────────────────────────
        participant_id = input_file.stem.split('_')[0]   # e.g. 'P01'
        task           = 'stroop'

        participant_row = participants[
            participants['participant_id'] == participant_id
        ]

        if len(participant_row) == 0:
            print(f"    WARNING: {participant_id} not in participants.csv")
            df.insert(0, 'task',           task)
            df.insert(0, 'participant_id', participant_id)
        else:
            meta = participant_row.iloc[0]
            df.insert(0, 'task',           task)
            df.insert(0, 'group',          meta['group'])
            df.insert(0, 'participant_id', participant_id)
            if 'age' in meta.index:
                df.insert(3, 'age', meta['age'])
            if 'sex' in meta.index:
                df.insert(4, 'sex', meta['sex'])

        # ── Save ─────────────────────────────────────────────────────────────
        df.to_csv(output_file, index=False)

        # ── Summary ──────────────────────────────────────────────────────────
        group_val = df['group'].iloc[0] if 'group' in df.columns else 'unknown'
        print(f"    participant_id : {participant_id}")
        print(f"    group          : {group_val}")
        print(f"    total trials   : {len(df)}")
        print(f"    missed (RT=0)  : {df['missed'].sum()} ({df['missed'].mean():.1%})")

        # Accuracy and RT by congruency (excluding missed trials)
        df_resp = df[df['missed'] == 0]
        print(f"\n    Congruency breakdown (responded trials only):")
        for cond in sorted(df['Congruency'].dropna().unique()):
            sub = df_resp[df_resp['Congruency'] == cond]
            if len(sub) == 0:
                continue
            acc = sub['Stimulus3.ACC'].mean()
            rt  = sub[sub['Stimulus3.ACC']==1]['Stimulus3.RT'].mean()
            print(f"      {cond:15s}: n={len(sub):3d}  ACC={acc:.1%}  "
                  f"RT={rt:.1f}ms")

        print(f"\n    saved to: {output_file.name}\n")

    except Exception as e:
        print(f"    ERROR processing {input_file.name}: {e}\n")

print("Done.")