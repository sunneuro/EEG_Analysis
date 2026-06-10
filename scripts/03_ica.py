# scripts/04_ica.py
# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 OF EEG PREPROCESSING: ICA artefact removal
#
# Usage:
#   python3 scripts/04_ica.py P01 stroop
#   python3 scripts/04_ica.py P01 nback
# ─────────────────────────────────────────────────────────────────────────────

import sys
import mne
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mne.preprocessing import ICA
from mne_icalabel import label_components
from pathlib import Path
from datetime import datetime

# ── Command-line arguments ────────────────────────────────────────────────────
if len(sys.argv) != 3:
    print("Usage: python3 scripts/04_ica.py <participant_id> <task>")
    print("Example: python3 scripts/04_ica.py P01 stroop")
    sys.exit(1)

participant_id = sys.argv[1]
task           = sys.argv[2]

print("=" * 60)
print(f"ICA PREPROCESSING")
print(f"Participant : {participant_id}")
print(f"Task        : {task}")
print(f"Started     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

# ── Paths ─────────────────────────────────────────────────────────────────────
input_file     = Path(f'data/processed/{participant_id}_{task}_filtered_raw.fif')
output_clean   = Path(f'data/processed/{participant_id}_{task}_ica_raw.fif')
output_ica     = Path(f'data/processed/{participant_id}_{task}-ica.fif')
ica_folder     = Path('output/ica')
decisions_file = Path('data/ica_decisions.csv')

ica_folder.mkdir(parents=True, exist_ok=True)
report_csv = ica_folder / f'{participant_id}_{task}_ica_report.csv'

# ── Step 1: Load filtered data ────────────────────────────────────────────────
print("\nSTEP 1: Loading filtered data")

if not input_file.exists():
    print(f"ERROR: {input_file} not found.")
    print(f"Run script 03 first for {participant_id} {task}.")
    sys.exit(1)

raw = mne.io.read_raw_fif(str(input_file), preload=True, verbose=False)
print(f"Loaded  : {input_file.name}")
print(f"Channels: {len(raw.ch_names)}  Duration: {raw.times[-1]:.1f}s")

montage = mne.channels.make_standard_montage('standard_1020')
raw.set_montage(montage, match_case=False, on_missing='warn')

# ── Step 2: Average reference ─────────────────────────────────────────────────
print("\nSTEP 2: Applying average reference")
raw.set_eeg_reference('average', projection=True)
raw.apply_proj()
print("Average reference applied.")

# ── Step 3: Prepare 1-100 Hz data for ICA fitting ────────────────────────────
print("\nSTEP 3: Preparing 1-100 Hz data for ICA fitting")

# Search for original .vhdr matching BOTH participant_id AND task
# Never fall back to another participant's file
all_vhdrs      = sorted(Path('data/raw').glob('*.vhdr'), key=lambda f: f.name)
task_matches   = [f for f in all_vhdrs if task.lower() in f.name.lower()]
pid_matches    = [f for f in task_matches if participant_id.lower() in f.name.lower()]

if pid_matches:
    vhdr_file = pid_matches[0]
    print(f"Using original raw file: {vhdr_file.name}")
    raw_orig = mne.io.read_raw_brainvision(
        str(vhdr_file), preload=True, verbose=False
    )
    raw_orig.set_montage(montage, match_case=False, on_missing='warn')
    raw_orig = mne.add_reference_channels(raw_orig, ref_channels=['Cz'])
    raw_orig.set_montage(montage, match_case=False, on_missing='warn')
    raw_orig.set_eeg_reference('average', projection=True)
    raw_orig.apply_proj()
    raw_for_ica = raw_orig.copy().filter(
        l_freq=1.0, h_freq=100.0,
        method='fir', fir_window='hamming', verbose=False
    )
else:
    print(f"WARNING: No .vhdr found matching both '{participant_id}' and '{task}'")
    print("Falling back to filtered .fif copy for ICA fitting")
    raw_for_ica = raw.copy().filter(
        l_freq=1.0, h_freq=100.0,
        method='fir', fir_window='hamming', verbose=False
    )
    raw_for_ica.set_eeg_reference('average', projection=True)
    raw_for_ica.apply_proj()

print("ICA fitting data ready.")

# ── Step 4: Fit ICA ───────────────────────────────────────────────────────────
print("\nSTEP 4: Fitting ICA — please wait (2-5 minutes)")

n_components = 25

ica = ICA(
    n_components=n_components,
    method='infomax',
    fit_params=dict(extended=True),
    random_state=42,
    max_iter=500,
    verbose=False
)

ica.fit(raw_for_ica, verbose=False)
print(f"ICA fitted — {n_components} components extracted.")

# ── Step 5: ICLabel automatic classification ──────────────────────────────────
print("\nSTEP 5: ICLabel automatic classification")

ic_labels = label_components(raw_for_ica, ica, method='iclabel')
labels    = ic_labels['labels']
probs     = ic_labels['y_pred_proba']

mixing = ica.get_components()

report_rows  = []
auto_exclude = []

# NOTE: muscle label is 'muscle artifact' not 'muscle'
ARTEFACT_LABELS = [
    'eye blink',
    'eye movement',
    'muscle artifact',
    'heart',
    'line noise',
    'channel noise',
]

for i, (label, prob) in enumerate(zip(labels, probs)):
    conf    = prob.max()
    top_idx = np.argsort(np.abs(mixing[:, i]))[::-1][:5]
    top_ch  = ', '.join([ica.ch_names[j] for j in top_idx])

    is_artefact = label in ARTEFACT_LABELS
    if is_artefact and conf > 0.70:
        auto_exclude.append(i)
        auto_decision = 'REMOVE'
    elif label == 'other':
        auto_decision = 'inspect'
    else:
        auto_decision = 'keep'

    report_rows.append({
        'component':      f'IC{i:02d}',
        'label':          label,
        'confidence':     round(conf, 4),
        'auto_decision':  auto_decision,
        'top_channels':   top_ch,
        'final_decision': ''
    })

print(f"\n{'IC':>5}  {'Label':>16}  {'Conf':>7}  {'Auto':>8}  Top channels")
print("-" * 80)
for r in report_rows:
    print(f"  {r['component']}  {r['label']:>16}  "
          f"{r['confidence']:>7.1%}  {r['auto_decision']:>8}  "
          f"{r['top_channels']}")

print(f"\nICLabel suggests removing: {auto_exclude}")

# ── Step 6: Save combined component figures ───────────────────────────────────
print(f"\nSTEP 6: Saving component figures (6 per figure)")

events_raw, event_id_raw = mne.events_from_annotations(raw_for_ica, verbose=False)
practice_end_code = 4 if task == 'stroop' else 6
practice_events   = events_raw[events_raw[:, 2] == practice_end_code]
practice_end_time = (
    practice_events[0, 0] / raw_for_ica.info['sfreq']
    if len(practice_events) > 0 else None
)

sources  = ica.get_sources(raw_for_ica)
times    = sources.times
all_data = sources.get_data()

decision_colours = {
    'keep':    '#1D9E75',
    'REMOVE':  '#A32D2D',
    'inspect': '#BA7517',
}

components_per_fig = 6
all_ics   = list(range(n_components))
ic_groups = [all_ics[i:i+components_per_fig] for i in range(0, len(all_ics), components_per_fig)]

figure_paths = []

for fig_num, ic_group in enumerate(ic_groups):
    n_rows = len(ic_group)
    fig, axes = plt.subplots(
        n_rows, 2,
        figsize=(16, n_rows * 2.5),
        gridspec_kw={'width_ratios': [1, 3]}
    )
    if n_rows == 1:
        axes = [axes]

    for row_idx, comp_num in enumerate(ic_group):
        ax_topo = axes[row_idx][0]
        ax_ts   = axes[row_idx][1]

        r            = report_rows[comp_num]
        label_str    = r['label']
        decision_str = r['auto_decision']
        conf_str     = f"{r['confidence']:.0%}"
        top_ch_str   = r['top_channels']
        colour       = decision_colours.get(decision_str, '#534AB7')

        if decision_str == 'keep':
            tag = 'BRAIN - keep'
        elif decision_str == 'REMOVE':
            tag = 'ARTEFACT - remove'
        else:
            tag = 'INSPECT'

        comp_vector = mixing[:, comp_num]
        abs_max     = np.abs(comp_vector).max()

        im, _ = mne.viz.plot_topomap(
            comp_vector, raw_for_ica.info,
            axes=ax_topo, show=False,
            contours=4, cmap='RdBu_r',
            vlim=(-abs_max, abs_max)
        )
        plt.colorbar(im, ax=ax_topo, fraction=0.046, pad=0.04)
        ax_topo.set_title(f'IC{comp_num:02d}', fontsize=10, fontweight='bold', color=colour, pad=3)

        signal = all_data[comp_num]
        ax_ts.plot(times, signal, color=colour, linewidth=0.4, alpha=0.85)
        ax_ts.axhline(0, color='#cccccc', linewidth=0.5, zorder=0)

        if practice_end_time is not None:
            ax_ts.axvline(x=practice_end_time, color='#185FA5', linewidth=1.0, linestyle='--', alpha=0.8, zorder=2)
            if fig_num == 0 and row_idx == 0:
                ax_ts.text(practice_end_time + 3, signal.max() * 0.7,
                           'Practice end\nMain task ->', fontsize=7, color='#185FA5', va='center', zorder=3)

        ax_ts.set_title(
            f'IC{comp_num:02d} - {label_str} ({conf_str}) - {tag}\nTop channels: {top_ch_str}',
            fontsize=8, loc='left', pad=2, color=colour
        )
        ax_ts.spines['top'].set_visible(False)
        ax_ts.spines['right'].set_visible(False)
        ax_ts.set_xlim(times[0], times[-1])

        if row_idx == n_rows - 1:
            ax_ts.set_xlabel('Time (s)', fontsize=9)
        else:
            ax_ts.tick_params(labelbottom=False)

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='#185FA5', linewidth=1.0, linestyle='--', label='Practice end / Main task begins'),
        Line2D([0], [0], color='#1D9E75', linewidth=1.5, label='Brain (keep)'),
        Line2D([0], [0], color='#A32D2D', linewidth=1.5, label='Artefact (remove)'),
        Line2D([0], [0], color='#BA7517', linewidth=1.5, label='Inspect'),
    ]
    axes[0][1].legend(handles=legend_elements, loc='upper right', fontsize=8, framealpha=0.7)

    ic_range = f'IC{ic_group[0]:02d}-IC{ic_group[-1]:02d}'
    plt.suptitle(
        f'{participant_id} {task.upper()} - ICA components {ic_range}\n'
        f'Left = scalp topography  Right = full time series  Dashed blue = practice end',
        fontsize=10, y=1.01
    )
    plt.tight_layout()

    fig_path = ica_folder / f'{participant_id}_{task}_ica_components_{ic_group[0]:02d}_{ic_group[-1]:02d}.png'
    fig.savefig(str(fig_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    figure_paths.append(fig_path)
    print(f"  Saved: {fig_path.name}")

print(f"\nAll component figures saved to: {ica_folder}/")

# ── Step 7: Experimenter decision ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: YOUR DECISION")
print("=" * 60)
print(f"\nOpen component figures in Finder to inspect:")
for fp in figure_paths:
    print(f"  {fp}")
print(f"\nICLabel automatic suggestion: remove {auto_exclude}")
print(f"\nReminder - what to look for:")
print(f"  Frontal bilateral blob (Fp1/Fp2) = eye blink    -> remove")
print(f"  Asymmetric frontal               = eye movement -> remove")
print(f"  Focal edge (temporal/frontal)    = muscle       -> consider removing")
print(f"  Regular rhythmic pulses          = heartbeat    -> remove")
print(f"  Smooth dipolar gradient          = brain        -> keep")
print(f"  When in doubt                                   -> keep")
print()

user_input = input(
    "Enter component numbers to remove (comma-separated),\n"
    "or press Enter to accept ICLabel suggestion,\n"
    "or type 'none' to remove nothing: "
).strip()

if user_input.lower() == 'none':
    final_exclude = []
elif user_input == '':
    final_exclude = auto_exclude
    print(f"Accepted ICLabel suggestion: {final_exclude}")
else:
    try:
        final_exclude = [int(x.strip()) for x in user_input.split(',')]
        print(f"Experimenter override: removing {final_exclude}")
    except ValueError:
        print("Could not parse input - using ICLabel suggestion.")
        final_exclude = auto_exclude

print(f"\nFinal components removed: {final_exclude}")

for r in report_rows:
    ic_num = int(r['component'].replace('IC', ''))
    r['final_decision'] = 'removed' if ic_num in final_exclude else 'kept'

# ── Step 8: Save report and log ───────────────────────────────────────────────
print(f"\nSTEP 8: Saving report and logging decision")

df_report = pd.DataFrame(report_rows)
df_report.to_csv(str(report_csv), index=False)
print(f"Component report saved : {report_csv.name}")

log_row = {
    'participant_id':     participant_id,
    'task':               task,
    'date':               datetime.now().strftime('%Y-%m-%d %H:%M'),
    'components_removed': str(final_exclude),
    'n_removed':          len(final_exclude),
    'auto_suggestion':    str(auto_exclude),
    'notes':              ''
}

if decisions_file.exists():
    df_log = pd.read_csv(decisions_file)
    df_log = df_log[
        ~((df_log['participant_id'] == participant_id) &
          (df_log['task'] == task))
    ]
    df_log = pd.concat([df_log, pd.DataFrame([log_row])], ignore_index=True)
else:
    df_log = pd.DataFrame([log_row])

df_log.to_csv(str(decisions_file), index=False)
print(f"Master log updated     : {decisions_file}")
print(f"  Removed : {final_exclude} ({len(final_exclude)} components)")
print(f"  Auto was: {auto_exclude}")

# ── Step 9: Apply ICA and save ────────────────────────────────────────────────
print(f"\nSTEP 9: Applying ICA to 0.1-40 Hz filtered data")

ica.exclude = final_exclude
raw_clean   = ica.apply(raw.copy(), verbose=False)
print(f"Removed {len(final_exclude)} component(s).")

raw_clean.save(str(output_clean), overwrite=True, verbose=False)
print(f"Cleaned data saved : {output_clean.name}")

ica.save(str(output_ica), overwrite=True, verbose=False)
print(f"ICA solution saved : {output_ica.name}")

print("\n" + "=" * 60)
print(f"COMPLETE - {participant_id} {task}")
print(f"Finished : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)
print(f"\nOutputs:")
print(f"  Cleaned EEG    : {output_clean}")
print(f"  ICA solution   : {output_ica}")
print(f"  Component figs : {ica_folder}/{participant_id}_{task}_ica_components_*.png")
print(f"  Report CSV     : {report_csv}")
print(f"  Master log     : {decisions_file}")
print(f"\nNext: python3 scripts/05_epochs.py {participant_id} {task}")