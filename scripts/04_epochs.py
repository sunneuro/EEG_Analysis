# scripts/05_epochs.py
# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 OF EEG PREPROCESSING: Epoching
#
# What this script does:
#   1. Loads ICA-cleaned data from script 04
#   2. Extracts events from annotations
#   3. Creates stimulus-locked epochs for all three Stroop conditions
#   4. Applies baseline correction (−200 to 0 ms)
#   5. Rejects epochs exceeding ±75 µV peak-to-peak (artefact rejection)
#   6. Prints rejection summary — how many trials lost per condition
#   7. Saves epochs object for ERP analysis in script 06
#
# Epoch window: −200 to +800 ms relative to stimulus onset (S3)
# Baseline    : −200 to 0 ms
# Rejection   : ±75 µV peak-to-peak across all channels
#
# Stroop trigger codes:
#   S3 (code 3) = stimulus onset — all trials
#   S5 (code 5) = correct response, Incongruent
#   S6 (code 6) = correct response, Congruent
#   S7 (code 7) = no response / timeout
#
# Epoching strategy:
#   Time-lock to S3 (stimulus onset)
#   Label each epoch by what followed (S5, S6, or S7)
#   This gives condition labels at the stimulus level
#
# Usage:
#   python3 scripts/05_epochs.py P01 stroop
# ─────────────────────────────────────────────────────────────────────────────

import sys
import mne
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# ── Command-line arguments ────────────────────────────────────────────────────
if len(sys.argv) != 3:
    print("Usage: python3 scripts/05_epochs.py <participant_id> <task>")
    print("Example: python3 scripts/05_epochs.py P01 stroop")
    sys.exit(1)

participant_id = sys.argv[1]
task           = sys.argv[2]

print("=" * 60)
print(f"EPOCHING")
print(f"Participant : {participant_id}")
print(f"Task        : {task}")
print(f"Started     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

# ── Paths ─────────────────────────────────────────────────────────────────────
input_file   = Path(f'data/processed/{participant_id}_{task}_ica_raw.fif')
output_file  = Path(f'data/processed/{participant_id}_{task}_epo.fif')
report_file  = Path(f'output/epochs/{participant_id}_{task}_epoch_report.csv')

output_file.parent.mkdir(parents=True, exist_ok=True)
report_file.parent.mkdir(parents=True, exist_ok=True)

# ── Epoching parameters ───────────────────────────────────────────────────────
# Time window relative to stimulus onset
TMIN        = -0.200    # −200 ms pre-stimulus baseline start
TMAX        =  0.800    # +800 ms post-stimulus
BASELINE    = (-0.200, 0.000)   # baseline correction window
REJECT_UV   = 75e-6     # ±75 µV rejection threshold (in Volts — MNE uses Volts)

# ── Step 1: Load ICA-cleaned data ─────────────────────────────────────────────
print("\nSTEP 1: Loading ICA-cleaned data")

if not input_file.exists():
    print(f"ERROR: {input_file} not found.")
    print(f"Run script 04 first for {participant_id} {task}.")
    sys.exit(1)

raw = mne.io.read_raw_fif(str(input_file), preload=True, verbose=False)
print(f"Loaded  : {input_file.name}")
print(f"Channels: {len(raw.ch_names)}")
print(f"Duration: {raw.times[-1]:.1f}s")
print(f"Sample rate: {raw.info['sfreq']:.0f} Hz")

# ── Step 2: Extract events ────────────────────────────────────────────────────
print("\nSTEP 2: Extracting events from annotations")

events, event_id_all = mne.events_from_annotations(raw, verbose=False)
print(f"Total events found: {len(events)}")
print(f"Event types: {event_id_all}")

# ── Step 3: Define condition labels ───────────────────────────────────────────
# For Stroop: epoch around S3 (stimulus onset) and label by response code
# Strategy: find each S3 event, look at what follows, assign condition label
#
# For Nback: epoch around S2 (stimulus onset) and label by response code
# We build this logic for both tasks

print("\nSTEP 3: Assigning condition labels to stimulus events")

if task == 'stroop':
    stimulus_code  = 3   # S3 = stimulus onset
    condition_map  = {
        5: 'incongruent/correct',   # S5 followed = incongruent correct
        6: 'congruent/correct',     # S6 followed = congruent correct
        7: 'no_response',           # S7 followed = no response (congruent or incongruent)
        # Note: S7 follows both congruent and incongruent no-response trials
        # We use a single no_response label here
        # Can be split further using behavioural data in script 08
    }
elif task == 'nback':
    stimulus_code  = 2   # S2 = stimulus onset
    condition_map  = {
        7: 'nontarget/correct',   # S7 = correct rejection
        8: 'target/hit',          # S8 = target hit
        9: 'target/miss',         # S9 = target miss / timeout
    }
else:
    print(f"ERROR: Unknown task '{task}'. Expected 'stroop' or 'nback'.")
    sys.exit(1)

# Build new event array with condition codes
# We assign a new event code to each S3/S2 based on what follows it
# New codes: 101 = condition 1, 102 = condition 2, 103 = condition 3

condition_codes = {}   # label → new integer code
new_events_list = []   # list of [sample, 0, new_code] arrays
skipped         = 0    # events with no following response code

# Map condition labels to new integer codes
for idx, (response_code, label) in enumerate(condition_map.items()):
    condition_codes[label] = 100 + idx + 1

print(f"Condition code mapping:")
for label, code in condition_codes.items():
    print(f"  {code} = {label}")

# Loop through all events and find stimulus onsets
for i, event in enumerate(events):
    if event[2] == stimulus_code:
        # Look at the next event to determine condition
        if i + 1 < len(events):
            next_code = events[i + 1, 2]
            if next_code in condition_map:
                label    = condition_map[next_code]
                new_code = condition_codes[label]
                new_events_list.append([event[0], 0, new_code])
            else:
                skipped += 1
        else:
            skipped += 1

new_events = np.array(new_events_list)
print(f"\nLabelled events: {len(new_events)}")
print(f"Skipped (no following response code): {skipped}")

# Print count per condition
print(f"\nTrials per condition (before rejection):")
for label, code in condition_codes.items():
    n = (new_events[:, 2] == code).sum()
    print(f"  {label:35s}: {n} trials")

# ── Step 4: Create epochs ─────────────────────────────────────────────────────
print(f"\nSTEP 4: Creating epochs")
print(f"  Window   : {TMIN*1000:.0f} to {TMAX*1000:.0f} ms")
print(f"  Baseline : {BASELINE[0]*1000:.0f} to {BASELINE[1]*1000:.0f} ms")
print(f"  Rejection: ±{REJECT_UV*1e6:.0f} µV")

# MNE handles boundary checking internally
# Use parameters directly
tmin_use = TMIN
tmax_use = TMAX

epochs = mne.Epochs(
    raw,
    new_events,
    event_id=condition_codes,
    tmin=tmin_use,
    tmax=tmax_use,
    baseline=BASELINE,
    reject=dict(eeg=REJECT_UV),
    preload=True,
    verbose=False,
    on_missing='warn'        # warn instead of crash if a condition has no events
)

print(f"\nEpochs created.")
print(f"  Total epochs before rejection: {len(new_events)}")
print(f"  Total epochs after  rejection: {len(epochs)}")
print(f"  Rejected: {len(new_events) - len(epochs)} "
      f"({(len(new_events)-len(epochs))/len(new_events)*100:.1f}%)")

# ── Step 5: Rejection summary per condition ───────────────────────────────────
print(f"\nSTEP 5: Rejection summary per condition")
print(f"{'Condition':35s} {'Before':>8} {'After':>8} {'Rejected':>10}")
print("-" * 65)

report_rows = []
for label, code in condition_codes.items():
    n_before = (new_events[:, 2] == code).sum()
    n_after  = len(epochs[label])
    n_reject = n_before - n_after
    pct      = n_reject / n_before * 100 if n_before > 0 else 0
    print(f"  {label:33s} {n_before:>8} {n_after:>8} "
          f"{n_reject:>6} ({pct:.1f}%)")
    report_rows.append({
        'participant_id': participant_id,
        'task':           task,
        'condition':      label,
        'n_before':       n_before,
        'n_after':        n_after,
        'n_rejected':     n_reject,
        'pct_rejected':   round(pct, 1),
        'threshold_uv':   REJECT_UV * 1e6
    })

# Overall rejection rate
total_before = len(new_events)
total_after  = len(epochs)
total_reject = total_before - total_after
print("-" * 65)
print(f"  {'TOTAL':33s} {total_before:>8} {total_after:>8} "
      f"{total_reject:>6} ({total_reject/total_before*100:.1f}%)")

# Flag if any condition has fewer than 20 trials after rejection
# 20 is the generally accepted minimum for stable ERP averages
print(f"\nTrial adequacy check (minimum 20 trials per condition):")
for row in report_rows:
    if row['n_after'] == 0:
        status = '✗ CRITICAL — no trials, condition excluded from ERP'
    elif row['n_after'] < 20:
        status = '✗ WARNING — too few trials for stable ERP'
    else:
        status = '✓'
    print(f"  {row['condition']:35s}: {row['n_after']} trials {status}")

# ── Step 6: Save epoch report ─────────────────────────────────────────────────
print(f"\nSTEP 6: Saving epoch report")

df_report = pd.DataFrame(report_rows)
df_report.to_csv(str(report_file), index=False)
print(f"Epoch report saved: {report_file}")

# ── Step 7: Save epochs ───────────────────────────────────────────────────────
print(f"\nSTEP 7: Saving epochs")

epochs.save(str(output_file), overwrite=True, verbose=False)
print(f"Epochs saved: {output_file.name}")

# Summary
print("\n" + "=" * 60)
print(f"COMPLETE — {participant_id} {task}")
print(f"Finished : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)
print(f"\nOutputs:")
print(f"  Epochs      : {output_file}")
print(f"  Epoch report: {report_file}")
print(f"\nEpoch info:")
print(f"  Window      : {tmin_use*1000:.0f} to {tmax_use*1000:.0f} ms")
print(f"  Conditions  : {list(condition_codes.keys())}")
print(f"  Channels    : {len(epochs.ch_names)}")
print(f"  Timepoints  : {len(epochs.times)}")
print(f"\nNext: python3 scripts/06_erp.py {participant_id} {task}")