import sys
import argparse
import mne
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec
from pathlib import Path
from datetime import datetime

# ── Config ──────────────────────────────────────────────────────────────────
def load_outlier_flags(task):
    flags = {}
    outliers_file = Path('data/outliers.csv')
    if outliers_file.exists():
        try:
            df = pd.read_csv(outliers_file)
            if 'task' in df.columns:
                df = df[df['task'] == task]
            for _, row in df.iterrows():
                pid = row['participant_id']
                cond = row.get('condition', 'all')
                flag = row.get('flag', 'outlier_from_csv')
                flags[(pid, cond)] = flag
        except Exception as e:
            print(f"  ⚠ Could not load outliers.csv: {e}")
    return flags

primary_components_nback = {
    'N200': {'tmin':0.200,'tmax':0.350,'electrode':'FC1+FC2','polarity':-1,
             'colour':'#185FA5','measures':['mean_amp','peak_amp','peak_lat'],
             'reference':'Jonides et al. (1997); Gevins & Smith (2000)'},
    'P300': {'tmin':0.300,'tmax':0.600,'electrode':'Pz','polarity':1,
             'colour':'#A32D2D','measures':['mean_amp','peak_amp','peak_lat'],
             'reference':'Polich (2007); Kok (2001)'},
    'P3b':  {'tmin':0.300,'tmax':0.600,'electrode':'Pz','polarity':1,
             'colour':'#A32D2D','measures':['peak_lat'],
             'reference':'Polich (2007)'},
}

exploratory_components_nback = {
    'N1':  {'tmin':0.080,'tmax':0.160,'electrode':'FC1+FC2','polarity':-1,
            'colour':'#6A0DAD','measures':['mean_amp'],
            'reference':'Luck & Kappenman (2012); Hillyard et al. (1973)'},
    'P2':  {'tmin':0.150,'tmax':0.250,'electrode':'Pz','polarity':1,
            'colour':'#E88C2A','measures':['mean_amp'],
            'reference':'Potts (2004); Wijers et al. (1989)'},
    'FSW': {'tmin':0.200,'tmax':0.500,'electrode':'Fz','polarity':-1,
            'colour':'#2E8B57','measures':['mean_amp'],
            'reference':'Ruchkin et al. (1992); Gevins et al. (1996)',
            'note':'Frontal slow wave — target/hit focus'},
}

primary_components_stroop = {
    'N200': {'tmin':0.200,'tmax':0.350,'electrode':'FC1+FC2','polarity':-1,
             'colour':'#185FA5','measures':['mean_amp','peak_amp','peak_lat'],
             'reference':'Folstein & Van Petten (2008); Luck & Kappenman (2012)'},
    'P300': {'tmin':0.300,'tmax':0.600,'electrode':'Pz','polarity':1,
             'colour':'#A32D2D','measures':['mean_amp','peak_amp','peak_lat'],
             'reference':'Polich (2007); Luck & Kappenman (2012)'},
    'P3b':  {'tmin':0.300,'tmax':0.600,'electrode':'Pz','polarity':1,
             'colour':'#A32D2D','measures':['peak_lat'],
             'reference':'Polich (2007)'},
}

exploratory_components_stroop = {
    'N1':  {'tmin':0.080,'tmax':0.160,'electrode':'FC1+FC2','polarity':-1,
            'colour':'#6A0DAD','measures':['mean_amp'],
            'reference':'Luck & Kappenman (2012); Hillyard et al. (1973)'},
    'CSW': {'tmin':0.400,'tmax':0.700,'electrode':'FC1+FC2','polarity':-1,
            'colour':'#1D7A6A','measures':['mean_amp'],
            'reference':'Larson et al. (2014); West & Alain (2000)'},
}

conditions_nback = ['nontarget/correct','target/hit','target/miss']
condition_colours_nback = {
    'nontarget/correct': '#1D9E75',
    'target/hit':        '#185FA5',
    'target/miss':       '#A32D2D',
}
condition_labels_nback = {
    'nontarget/correct': 'Non-target correct',
    'target/hit':        'Target hit',
    'target/miss':       'Target miss',
}

conditions_stroop = ['congruent/correct','incongruent/correct','no_response']
condition_colours_stroop = {
    'congruent/correct':   '#1D9E75',
    'incongruent/correct': '#A32D2D',
    'no_response':         '#888780',
}
condition_labels_stroop = {
    'congruent/correct':   'Congruent correct',
    'incongruent/correct': 'Incongruent correct',
    'no_response':         'No response',
}

def main():
    parser = argparse.ArgumentParser(description="ERP averaging and component extraction.")
    parser.add_argument('participant_id', help="Participant ID (e.g. P01)")
    parser.add_argument('--task', choices=['nback', 'stroop'], required=True, help="Task type")
    args = parser.parse_args()

    participant_id = args.participant_id
    task = args.task

    print("=" * 60)
    print(f"ERP EXTRACTION — {task.upper()}  |  {participant_id}")
    print(f"Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    input_file  = Path(f'data/processed/{participant_id}_{task}_epo.fif')
    erp_folder  = Path('output/erp')
    report_file = Path(f'output/erp/{participant_id}_{task}_erp_components.csv')
    erp_folder.mkdir(parents=True, exist_ok=True)

    MIN_TRIALS = 20

    outlier_flags = load_outlier_flags(task)

    def get_outlier_flag(pid, cond, n):
        if (pid, 'all')  in outlier_flags: return outlier_flags[(pid, 'all')]
        if (pid, cond)   in outlier_flags: return outlier_flags[(pid, cond)]
        if n < MIN_TRIALS:                 return f'low_trials_n{n}'
        return ''

    if task == 'nback':
        primary_components = primary_components_nback
        exploratory_components = exploratory_components_nback
        conditions = conditions_nback
        condition_colours = condition_colours_nback
        condition_labels = condition_labels_nback
        fc1fc2_windows = {
            'N1':   exploratory_components['N1'],
            'N200': primary_components['N200'],
        }
        pz_windows = {
            'P2':        exploratory_components['P2'],
            'P300 / P3b': primary_components['P300'],
        }
        title_fc1fc2 = f'{participant_id} {task.upper()} — ERP at FC1+FC2 (virtual FCz)\nPurple=N1 (80–160ms) · Blue=N200 (200–350ms) · Negative up'
        title_pz = f'{participant_id} {task.upper()} — ERP at Pz\nOrange=P2 (150–250ms) · Red=P300/P3b (300–600ms) · Negative up'
        title_fz = f'{participant_id} {task.upper()} — ERP at Fz (display only)\nFrontal slow wave / sustained negativity · No component extraction · Negative up'
    else:
        primary_components = primary_components_stroop
        exploratory_components = exploratory_components_stroop
        conditions = conditions_stroop
        condition_colours = condition_colours_stroop
        condition_labels = condition_labels_stroop
        fc1fc2_windows = {
            'N1':   exploratory_components['N1'],
            'N200': primary_components['N200'],
            'CSW':  exploratory_components['CSW'],
        }
        pz_windows = {'P300 / P3b': primary_components['P300']}
        title_fc1fc2 = f'{participant_id} {task.upper()} — ERP at FC1+FC2 (virtual FCz)\nPurple=N1 (80–160ms) · Blue=N200 (200–350ms) · Teal=CSW (400–700ms) · Negative up'
        title_pz = f'{participant_id} {task.upper()} — ERP at Pz\nRed = P300 / P3b (300–600 ms) · Negative up'
        title_fz = f'{participant_id} {task.upper()} — ERP at Fz (display only)\nNo component extraction · Negative up'

    all_components = {**primary_components, **exploratory_components}
    display_order = conditions[:]

    topomap_times = [0.120, 0.220, 0.350, 0.450, 0.550]
    FCZ_COLOUR    = '#185FA5'
    PZ_COLOUR     = '#A32D2D'

    def get_ch(evoked, elec):
        if '+' in elec:
            chs = elec.split('+')
            idx = [evoked.ch_names.index(c) for c in chs if c in evoked.ch_names]
            return np.mean([evoked.data[i] for i in idx], axis=0)*1e6 if idx else None
        return evoked.data[evoked.ch_names.index(elec)]*1e6 \
               if elec in evoked.ch_names else None

    def lighten(hex_col, amount=0.50):
        c = mcolors.to_rgb(hex_col)
        return tuple(c[i]+(1.0-c[i])*amount for i in range(3))

    def ticks(ax, lo=-200, hi=800, step=100):
        t = np.arange(lo, hi+step, step)
        ax.set_xticks(t)
        ax.set_xticklabels([str(int(x)) for x in t], fontsize=8)
        ax.set_xlim(lo, hi)

    def style(ax):
        ax.axhline(0, color='#cccccc', lw=0.8)
        ax.axvline(0, color='#cccccc', lw=0.8, ls='--')
        ax.invert_yaxis()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    def shade(ax, wdict, y=0):
        for lbl, comp in wdict.items():
            ax.axvspan(comp['tmin']*1000, comp['tmax']*1000,
                       alpha=0.07, color=comp['colour'])
            ax.text((comp['tmin']+comp['tmax'])/2*1000, y, lbl,
                    ha='center', va='center', fontsize=9,
                    color=comp['colour'], fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    print("\nSTEP 1: Loading epochs")
    if not input_file.exists():
        print(f"ERROR: {input_file} not found. Run script 05 first.")
        sys.exit(1)

    epochs = mne.read_epochs(str(input_file), verbose=False)
    print(f"Epochs: {len(epochs)}  |  Window: "
          f"{epochs.tmin*1000:.0f}–{epochs.tmax*1000:.0f} ms")
    tmin_ms = int(epochs.tmin*1000)
    tmax_ms = int(epochs.tmax*1000)

    print("\nSTEP 2: Computing evoked responses")
    evokeds = {}
    for cond in conditions:
        if cond in epochs.event_id:
            ev            = epochs[cond].average()
            evokeds[cond] = ev
            flag          = get_outlier_flag(participant_id, cond, ev.nave)
            fstr          = f'  ⚠ {flag}' if flag else ''
            print(f"  {cond:35s}: {ev.nave:4d} trials{fstr}")
        else:
            print(f"  {cond:35s}: NOT FOUND")

    if not evokeds:
        print("ERROR: No conditions found.")
        sys.exit(1)

    ordered = [(c, evokeds[c]) for c in display_order if c in evokeds]

    print("\nSTEP 3: Butterfly plots")

    fig, axes = plt.subplots(len(ordered), 1,
                             figsize=(14, len(ordered)*3), sharex=True)
    if len(ordered) == 1: axes = [axes]

    for ax, (cond, ev) in zip(axes, ordered):
        times = ev.times*1000
        data  = ev.data*1e6
        for ci, ch in enumerate(ev.ch_names):
            if ch in ['FC1','FC2']:
                ax.plot(times, data[ci], color=FCZ_COLOUR,
                        lw=0.8, ls='--', alpha=0.5, zorder=2)
            elif ch == 'Pz':
                ax.plot(times, data[ci], color=condition_colours[cond],
                        lw=0.8, ls='-', alpha=0.5, zorder=2)
            else:
                ax.plot(times, data[ci], color=condition_colours[cond],
                        lw=0.3, alpha=0.25, zorder=1)
        vf = get_ch(ev, 'FC1+FC2')
        if vf is not None:
            ax.plot(times, vf, color=FCZ_COLOUR, lw=2.0,
                    ls='--', label='FC1+FC2 (virtual FCz)', zorder=5)
        pz = get_ch(ev, 'Pz')
        if pz is not None:
            ax.plot(times, pz, color=condition_colours[cond],
                    lw=2.0, ls='-', label='Pz', zorder=5)
        style(ax)
        ax.set_ylabel('Amplitude (µV)', fontsize=11)
        ax.set_title(f"{condition_labels.get(cond,cond)} (n={ev.nave})",
                     fontsize=13, loc='left', color=condition_colours[cond])
        ax.legend(fontsize=8, loc='upper right')
        ticks(ax, tmin_ms, tmax_ms)

    axes[-1].set_xlabel('Time (ms)', fontsize=12)
    plt.suptitle(
        f'{participant_id} {task.upper()} — Butterfly plots (negative up)\n'
        f'Blue dashed = FC1, FC2 + virtual FCz · Solid = Pz · Faded = other',
        fontsize=11, y=1.01)
    plt.tight_layout()
    fig.savefig(str(erp_folder/f'{participant_id}_{task}_butterfly.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {participant_id}_{task}_butterfly.png")

    print("\nSTEP 4: ERP plots")

    # 4a: FC1+FC2
    fig, ax = plt.subplots(figsize=(12, 4))
    for cond, ev in ordered:
        d = get_ch(ev, 'FC1+FC2')
        if d is None: continue
        ax.plot(ev.times*1000, d, color=condition_colours[cond],
                lw=2, label=f"{condition_labels.get(cond,cond)} (n={ev.nave})")
    shade(ax, fc1fc2_windows)
    style(ax)
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('Amplitude (µV)', fontsize=12)
    ax.legend(fontsize=9, loc='upper right')
    ax.set_title(title_fc1fc2, fontsize=10)
    ticks(ax, tmin_ms, tmax_ms)
    plt.tight_layout()
    fig.savefig(str(erp_folder/f'{participant_id}_{task}_erp_FC1_FC2.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {participant_id}_{task}_erp_FC1_FC2.png")

    # 4b: Pz
    fig, ax = plt.subplots(figsize=(12, 4))
    for cond, ev in ordered:
        d = get_ch(ev, 'Pz')
        if d is None: continue
        ax.plot(ev.times*1000, d, color=condition_colours[cond],
                lw=2, label=f"{condition_labels.get(cond,cond)} (n={ev.nave})")
    shade(ax, pz_windows)
    style(ax)
    ax.set_xlabel('Time (ms)', fontsize=12)
    ax.set_ylabel('Amplitude (µV)', fontsize=12)
    ax.legend(fontsize=9, loc='upper right')
    ax.set_title(title_pz, fontsize=10)
    ticks(ax, tmin_ms, tmax_ms)
    plt.tight_layout()
    fig.savefig(str(erp_folder/f'{participant_id}_{task}_erp_Pz.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {participant_id}_{task}_erp_Pz.png")

    # 4c: Fz
    fig, ax = plt.subplots(figsize=(12, 4))
    plotted = False
    for cond, ev in ordered:
        d = get_ch(ev, 'Fz')
        if d is None: continue
        ax.plot(ev.times*1000, d, color=condition_colours[cond],
                lw=2, label=f"{condition_labels.get(cond,cond)} (n={ev.nave})")
        plotted = True
    if plotted:
        style(ax)
        ax.set_xlabel('Time (ms)', fontsize=12)
        ax.set_ylabel('Amplitude (µV)', fontsize=12)
        ax.legend(fontsize=9, loc='upper right')
        ax.set_title(title_fz, fontsize=10)
        ticks(ax, tmin_ms, tmax_ms)
        plt.tight_layout()
        fig.savefig(str(erp_folder/f'{participant_id}_{task}_erp_Fz.png'),
                    dpi=150, bbox_inches='tight')
        print(f"  Saved: {participant_id}_{task}_erp_Fz.png")
    plt.close(fig)

    print("\nSTEP 5: Topomaps")

    n_conds = len(ordered)
    n_times = len(topomap_times)
    all_vals = []
    for cond, ev in ordered:
        for t in topomap_times:
            i0 = np.searchsorted(ev.times, t-0.025)
            i1 = np.searchsorted(ev.times, t+0.025)
            all_vals.extend(np.abs(ev.data[:,i0:i1].mean(1)*1e6).tolist())
    gmax = np.percentile(all_vals, 95)

    fig      = plt.figure(figsize=(n_times*2.8, n_conds*2.8+0.9))
    tt       = 0.88
    gs       = GridSpec(n_conds, n_times, figure=fig,
                        top=tt-0.02, bottom=0.03,
                        left=0.10, right=0.97,
                        hspace=0.15, wspace=0.05)
    last_im  = None
    for ri, (cond, ev) in enumerate(ordered):
        for ci, t in enumerate(topomap_times):
            ax  = fig.add_subplot(gs[ri, ci])
            i0  = np.searchsorted(ev.times, t-0.025)
            i1  = np.searchsorted(ev.times, t+0.025)
            avg = ev.data[:,i0:i1].mean(1)*1e6
            im, _ = mne.viz.plot_topomap(avg, ev.info, axes=ax, show=False,
                                         cmap='RdBu_r', contours=4,
                                         vlim=(-gmax, gmax))
            last_im = im
            ax.set_title(f'{int(t*1000)} ms', fontsize=8, pad=2)
            if ci == 0:
                ax.set_ylabel(condition_labels.get(cond,cond),
                              fontsize=13, labelpad=10)

    cbar_ax = fig.add_axes([0.60, tt+0.03, 0.22, 0.018])
    cbar    = fig.colorbar(last_im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('µV', fontsize=9, labelpad=2)
    cbar.ax.tick_params(labelsize=8)
    fig.text(0.02, tt+0.04,
             f'{participant_id} {task.upper()} — Topographic maps\n'
             f'120–550 ms (±25 ms avg) · Shared scale ±{gmax:.1f} µV',
             fontsize=13, va='bottom', ha='left')
    fig.savefig(str(erp_folder/f'{participant_id}_{task}_topomaps_combined.png'),
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {participant_id}_{task}_topomaps_combined.png")

    print("\nSTEP 6: Extracting component values")

    rows = []
    for cond, ev in ordered:
        flag = get_outlier_flag(participant_id, cond, ev.nave)
        for comp_name, comp in all_components.items():
            d = get_ch(ev, comp['electrode'])
            if d is None: continue

            t     = ev.times
            i0    = np.searchsorted(t, comp['tmin'])
            i1    = np.searchsorted(t, comp['tmax'])
            win   = d[i0:i1]

            mean_amp = float(np.mean(win)) if 'mean_amp' in comp['measures'] else np.nan
            peak_amp = np.nan
            peak_lat = np.nan

            if 'peak_amp' in comp['measures'] or 'peak_lat' in comp['measures']:
                pi       = np.argmin(win) if comp['polarity']==-1 else np.argmax(win)
                peak_amp = float(win[pi])
                peak_lat = float(t[i0+pi]*1000)

            ctype = 'primary' if comp_name in primary_components else 'exploratory'

            comp_note = ''
            if task == 'nback':
                if comp_name == 'FSW' and cond != 'target/hit':
                    comp_note = 'non_primary_condition'
                if comp_name == 'P3b':
                    comp_note = 'latency_only'

            r = {
                'participant_id': participant_id,
                'task':           task,
                'condition':      cond,
                'n_trials':       ev.nave,
                'component':      comp_name,
                'electrode':      comp['electrode'],
                'tmin_ms':        comp['tmin']*1000,
                'tmax_ms':        comp['tmax']*1000,
                'mean_amp_uv':    round(mean_amp,4) if not np.isnan(mean_amp) else np.nan,
                'peak_amp_uv':    round(peak_amp,4) if not np.isnan(peak_amp) else np.nan,
                'peak_lat_ms':    round(peak_lat,1) if not np.isnan(peak_lat) else np.nan,
                'component_type': ctype,
                'outlier_flag':   flag,
            }
            if task == 'nback':
                r['component_note'] = comp_note
            rows.append(r)

    print("\nSTEP 7: Saving CSV")
    df = pd.DataFrame(rows)
    df.to_csv(str(report_file), index=False)
    print(f"Saved: {report_file.name}")
    
    disp_cols = ['condition','component','mean_amp_uv','peak_amp_uv','peak_lat_ms','component_type','outlier_flag']
    print(df[disp_cols].to_string(index=False))

    print("\n" + "="*60)
    print(f"COMPLETE — {participant_id} {task.upper()}  |  "
          f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    print(f"\nNext: python3 scripts/07_group_erp.py --task {task}")

if __name__ == '__main__':
    main()
