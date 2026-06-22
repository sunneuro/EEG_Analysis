import pandas as pd
import argparse
from pathlib import Path
import sys

def main():
    parser = argparse.ArgumentParser(description="Parse E-Prime behavioural data.")
    parser.add_argument('--task', choices=['nback', 'stroop'], required=True, help="Task type")
    args = parser.parse_args()

    task = args.task
    task_folder = Path(f'data/behavioural/{task}')
    participants_file = Path('data/participants.csv')

    if not participants_file.exists():
        print(f"ERROR: {participants_file} not found.")
        print("Create data/participants.csv first.")
        sys.exit(1)

    participants = pd.read_csv(participants_file)
    print(f"Loaded participants.csv: {len(participants)} participants")
    print(participants[['participant_id', 'name', 'group']].to_string(index=False))
    print()

    txt_files = sorted(task_folder.glob('*.txt'))

    if len(txt_files) == 0:
        print(f"No .txt files found in {task_folder}")
        print(f"Make sure your E-Prime {task} files are copied there.")
        sys.exit(1)

    print(f"Found {len(txt_files)} file(s) to process:\n")

    for input_file in txt_files:
        output_file = task_folder / (input_file.stem + '_clean.csv')

        if output_file.exists():
            print(f"  SKIP (already exists): {output_file.name}")
            continue

        print(f"  Processing: {input_file.name}")

        try:
            df_raw = pd.read_csv(
                input_file,
                sep='\t',
                encoding='utf-16',
                header=None,
                low_memory=False
            )

            col_names = list(df_raw.iloc[3])
            df = df_raw.iloc[4:].copy()
            df.columns = col_names
            df = df.reset_index(drop=True)

            if task == 'nback':
                df = df[
                    df['Stimulus.RT'].notna() &
                    (df['Stimulus.RT'] != 'NULL') &
                    (df['Stimulus.RT'] != '')
                ].copy().reset_index(drop=True)
                
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

                df['Stimulus.RT']  = pd.to_numeric(df['Stimulus.RT'],  errors='coerce')
                df['Stimulus.ACC'] = pd.to_numeric(df['Stimulus.ACC'], errors='coerce')

                df['missed'] = (df['Stimulus.RT'] == 0).astype(int)

                practice_lists = ['Prac0List', 'Prac1List', 'Prac2List']
                df['is_practice'] = df['ListName'].isin(practice_lists).astype(int)

            elif task == 'stroop':
                df = df[
                    df['Stimulus3.RT'].notna() &
                    (df['Stimulus3.RT'] != 'NULL') &
                    (df['Stimulus3.RT'] != '')
                ].copy().reset_index(drop=True)

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

                df['Stimulus3.RT']  = pd.to_numeric(df['Stimulus3.RT'],  errors='coerce')
                df['Stimulus3.ACC'] = pd.to_numeric(df['Stimulus3.ACC'], errors='coerce')

                if 'Congruency' in df.columns:
                    df['Congruency'] = df['Congruency'].str.strip().str.title()

                df['missed'] = (df['Stimulus3.RT'] == 0).astype(int)

            participant_id = input_file.stem.split('_')[0]

            participant_row = participants[
                participants['participant_id'] == participant_id
            ]

            if len(participant_row) == 0:
                print(f"    WARNING: {participant_id} not found in participants.csv")
                print(f"    Skipping this file.")
                continue
            else:
                meta = participant_row.iloc[0]
                df.insert(0, 'task',           task)
                df.insert(0, 'group',          meta['group'])
                df.insert(0, 'participant_id', participant_id)
                if 'age' in meta.index and pd.notna(meta['age']):
                    df.insert(3, 'age', meta['age'])
                if 'sex' in meta.index and pd.notna(meta['sex']):
                    df.insert(4, 'sex', meta['sex'])

            df.to_csv(output_file, index=False)

            group_val = df['group'].iloc[0] if 'group' in df.columns else 'unknown'
            print(f"    participant_id : {participant_id}")
            print(f"    group          : {group_val}")
            
            if task == 'nback':
                exp_trials = df[df['is_practice'] == 0]
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
            elif task == 'stroop':
                print(f"    total trials   : {len(df)}")
                print(f"    missed (RT=0)  : {df['missed'].sum()} ({df['missed'].mean():.1%})")

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

if __name__ == '__main__':
    main()
