# scripts/07_group_erp_nback.py
# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 OF EEG PIPELINE: Group-level ERP analysis — NBACK
#
# All participants included. Outlier flags preserved from script 06.
# Group order: Control (left) | Creatine (right) throughout all figures.
#
# PRIMARY components: N200, P300, P3b (latency only)
# EXPLORATORY components: N1, P2, FSW (frontal slow wave)
#
# Outputs — output/group/:
#   group_nback_erp_FC1_FC2.png / _by_condition.pngroup_nback_erp_FC1_FC2_by_conditiong
#   group_nback_erp_Pz.png     / _by_condition.png
#   group_nback_erp_Fz.png     / _by_condition.png
#   group_nback_N200_amplitude_bars.png     (primary, p-value annotated)
#   group_nback_N200_latency_bars.png
#   group_nback_P300_amplitude_bars.png
#   group_nback_P300_latency_bars.png
#   group_nback_P3b_latency_bars.png
#   group_nback_N1_amplitude_bars.png       (exploratory, d only)
#   group_nback_P2_amplitude_bars.png
#   group_nback_FSW_amplitude_bars.png      (target/hit only)
#   group_nback_target_effect.png
#   group_nback_P3b_vs_P300_latency.png
#   group_nback_summary.csv
#   group_nback_individual_components.csv
#
# Usage:
#   python3 scripts/07_group_erp_nback.py
# ─────────────────────────────────────────────────────────────────────────────

import mne
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats
from statsmodels.stats.multitest import multipletests
from pathlib import Path
from datetime import datetime

task = 'nback'

print("=" * 60)
print(f"GROUP ERP ANALYSIS — NBACK")
print(f"Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)

# ── Paths ─────────────────────────────────────────────────────────────────────
erp_folder   = Path('output/erp')
group_folder = Path('output/group')
group_folder.mkdir(parents=True, exist_ok=True)

participants_file = Path('data/participants.csv')
participants      = pd.read_csv(participants_file)
group_map         = dict(zip(participants['participant_id'], participants['group']))

excluded = []

# ── Component definitions ─────────────────────────────────────────────────────
primary_components = {
    'N200': {'tmin':0.200,'tmax':0.350,'electrode':'FC1+FC2',
             'polarity':-1,'colour':'#185FA5'},
    'P300': {'tmin':0.300,'tmax':0.600,'electrode':'Pz',
             'polarity':1,'colour':'#A32D2D'},
    'P3b':  {'tmin':0.300,'tmax':0.600,'electrode':'Pz',
             'polarity':1,'colour':'#A32D2D'},
}
exploratory_components = {
    'N1':  {'tmin':0.080,'tmax':0.160,'electrode':'FC1+FC2',
            'polarity':-1,'colour':'#6A0DAD'},
    'P2':  {'tmin':0.150,'tmax':0.250,'electrode':'Pz',
            'polarity':1,'colour':'#E88C2A'},
    'FSW': {'tmin':0.200,'tmax':0.500,'electrode':'Fz',
            'polarity':-1,'colour':'#2E8B57'},
}
all_components = {**primary_components, **exploratory_components}

fc1fc2_windows = {
    'N1':   exploratory_components['N1'],
    'N200': primary_components['N200'],
}
pz_windows = {
    'P2':         exploratory_components['P2'],
    'P300 / P3b': primary_components['P300'],
}

# ── Condition / group settings ────────────────────────────────────────────────
conditions  = ['nontarget/correct','target/hit','target/miss']
cond_labels = {
    'nontarget/correct': 'Non-target',
    'target/hit':        'Target hit',
    'target/miss':       'Target miss',
}
cond_colours = {
    'nontarget/correct': '#1D9E75',
    'target/hit':        '#185FA5',
    'target/miss':       '#A32D2D',
}

# Control LEFT, Creatine RIGHT
groups           = ['control','creatine']
group_colours    = {'control':'#A32D2D','creatine':'#185FA5'}
group_linestyles = {'control':':','creatine':'-'}
group_labels     = {'control':'Control','creatine':'Creatine'}

bar_styles = {
    'control':  {'facecolor':'#DDDDDD','edgecolor':'#333333',
                 'linewidth':2.5,'linestyle':':'},
    'creatine': {'facecolor':'#AAAAAA','edgecolor':'#333333',
                 'linewidth':2.5,'linestyle':'-'},
}

IND_LW    = 1.2
IND_ALPHA = 0.45
GA_LW     = 2.8
GA_ALPHA  = 1.0

# ── Helpers ───────────────────────────────────────────────────────────────────
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

def ticks(ax):
    t = np.arange(-200, 900, 100)
    ax.set_xticks(t)
    ax.set_xticklabels([str(int(x)) for x in t], fontsize=12)
    ax.set_xlim(-200, 800)

def style(ax):
    ax.axhline(0, color='#cccccc', lw=0.8)
    ax.axvline(0, color='#cccccc', lw=0.8, ls='--')
    ax.invert_yaxis()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=12)

def shade(ax, wdict, y=0):
    for lbl, comp in wdict.items():
        ax.axvspan(comp['tmin']*1000, comp['tmax']*1000,
                   alpha=0.07, color=comp['colour'])
        ax.text((comp['tmin']+comp['tmax'])/2*1000, y, lbl,
                ha='center', va='center', fontsize=13,
                color=comp['colour'], fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'),
                zorder=10)

def cohens_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2: return np.nan
    s = np.sqrt(((na-1)*np.std(a,ddof=1)**2+(nb-1)*np.std(b,ddof=1)**2)/(na+nb-2))
    return np.nan if s == 0 else (np.mean(a)-np.mean(b))/s

def run_tests(a, b):
    if len(a) >= 2 and len(b) >= 2:
        t, tp = stats.ttest_ind(a, b, equal_var=False)
    else:
        t, tp = np.nan, np.nan
    try:
        u, up = stats.mannwhitneyu(a, b, alternative='two-sided') \
                if len(a)>=1 and len(b)>=1 else (np.nan, np.nan)
    except ValueError:
        u, up = np.nan, np.nan
    d = cohens_d(a, b)
    return t, tp, u, up, d

def stars(p):
    if np.isnan(p): return 'n/a'
    if p < 0.001:   return '***'
    if p < 0.01:    return '**'
    if p < 0.05:    return '*'
    if p < 0.10:    return '†'
    return 'ns'



# ── Step 1: Load CSVs ─────────────────────────────────────────────────────────
print(f"\nSTEP 1: Loading individual ERP component files")

csv_files = sorted(erp_folder.glob(f'P*_{task}_erp_components.csv'))
if not csv_files:
    print("ERROR: No component CSVs found. Run 06_erp_nback.py first.")
    import sys; sys.exit(1)

all_dfs = []
for f in csv_files:
    pid = f.stem.split('_')[0]
    df  = pd.read_csv(f)
    df['group'] = group_map.get(pid, 'unknown')
    all_dfs.append(df)
    print(f"  Loaded: {f.name}  (group={group_map.get(pid,'?')})")

combined = pd.concat(all_dfs, ignore_index=True)
print(f"\nParticipants: {sorted(combined['participant_id'].unique())}")

# ── Step 2: Group statistics ──────────────────────────────────────────────────
print(f"\nSTEP 2: Group statistics")

stats_rows = []
for comp in all_components:
    for cond in conditions:
        for grp in groups:
            sub = combined[(combined['component']==comp) &
                           (combined['condition']==cond) &
                           (combined['group']==grp)]
            if sub.empty: continue
            n = len(sub)
            stats_rows.append({
                'task':task,'component':comp,'condition':cond,'group':grp,'n':n,
                'mean_amp_uv':   round(sub['mean_amp_uv'].mean(),4)
                                 if sub['mean_amp_uv'].notna().any() else np.nan,
                'sd_amp_uv':     round(sub['mean_amp_uv'].std(),4)
                                 if sub['mean_amp_uv'].notna().any() else np.nan,
                'sem_amp_uv':    round(sub['mean_amp_uv'].std()/np.sqrt(n),4)
                                 if sub['mean_amp_uv'].notna().any() else np.nan,
                'mean_peak_amp': round(sub['peak_amp_uv'].mean(),4)
                                 if sub['peak_amp_uv'].notna().any() else np.nan,
                'mean_lat_ms':   round(sub['peak_lat_ms'].mean(),1)
                                 if sub['peak_lat_ms'].notna().any() else np.nan,
                'sd_lat_ms':     round(sub['peak_lat_ms'].std(),1)
                                 if sub['peak_lat_ms'].notna().any() else np.nan,
            })

df_stats = pd.DataFrame(stats_rows)

# ── Step 3: Load epochs ───────────────────────────────────────────────────────
print(f"\nSTEP 3: Loading epochs for grand averages")

group_evokeds = {grp: [] for grp in groups}
ind_evokeds   = {(grp, cond): [] for grp in groups for cond in conditions}

for pid in sorted(combined['participant_id'].unique()):
    epo_file = Path(f'data/processed/{pid}_{task}_epo.fif')
    if not epo_file.exists():
        print(f"  WARNING: {epo_file.name} not found"); continue
    grp    = group_map.get(pid,'unknown')
    epochs = mne.read_epochs(str(epo_file), verbose=False)
    for cond in conditions:
        if cond in epochs.event_id:
            ev = epochs[cond].average()
            ev.comment = cond
            group_evokeds[grp].append((cond, ev))
            ind_evokeds[(grp, cond)].append((pid, ev))
    print(f"  Loaded: {pid} ({grp})")

grand_avgs = {}
for grp in groups:
    for cond in conditions:
        evlist = [ev for c, ev in group_evokeds[grp] if c == cond]
        if evlist:
            grand_avgs[(grp, cond)] = mne.grand_average(evlist)

# ── Step 4: Side-by-side waveform plots ───────────────────────────────────────
print(f"\nSTEP 4: Waveform plots — side-by-side")

elec_cfgs = [
    {'elec':'FC1+FC2','label':'FC1+FC2 (virtual FCz)','fname':'FC1_FC2',
     'windows':fc1fc2_windows,
     'subtitle':'Purple=N1 (80–160ms) · Blue=N200 (200–350ms)'},
    {'elec':'Pz','label':'Pz','fname':'Pz',
     'windows':pz_windows,
     'subtitle':'Orange=P2 (150–250ms) · Red=P300/P3b (300–600ms)'},
    {'elec':'Fz','label':'Fz (display only)','fname':'Fz',
     'windows':{},
     'subtitle':'Frontal slow wave — no component extraction'},
]

for ecfg in elec_cfgs:
    elec = ecfg['elec']
    fig, axes = plt.subplots(1, 2, figsize=(14.4, 4.5), sharey=False)

    for ax, grp in zip(axes, groups):  # control left, creatine right
        for cond in conditions:
            light_col = lighten(cond_colours[cond], 0.50)
            for pid, ev in ind_evokeds.get((grp, cond), []):
                d = get_ch(ev, elec)
                if d is None: continue
                ax.plot(ev.times*1000, d, color=light_col,
                        lw=IND_LW, alpha=IND_ALPHA, zorder=1)

        for cond in conditions:
            if (grp, cond) not in grand_avgs: continue
            ev = grand_avgs[(grp, cond)]
            d  = get_ch(ev, elec)
            if d is None: continue
            n_p = len(ind_evokeds.get((grp, cond), []))
            ax.plot(ev.times*1000, d, color=cond_colours[cond],
                    lw=GA_LW, alpha=GA_ALPHA, zorder=5,
                    label=f"{cond_labels[cond]} (N={n_p})")

        shade(ax, ecfg['windows'])
        style(ax)
        ax.set_xlabel('Time (ms)', fontsize=15)
        ax.set_ylabel('Amplitude (µV)', fontsize=16)
        ax.legend(fontsize=13, loc='upper right')
        grp_n = len(combined[combined['group']==grp]['participant_id'].unique())
        ax.set_title(f"{group_labels[grp]} group (N={grp_n})",
                     fontsize=16, fontweight='bold', color=group_colours[grp])
        ticks(ax)

    # Synchronize y-limits across subplots for fair visual comparison (dynamic scale)
    ylims = [ax.get_ylim() for ax in axes]
    new_bottom = max(lim[0] for lim in ylims)
    new_top = min(lim[1] for lim in ylims)
    for ax in axes:
        ax.set_ylim(new_bottom, new_top)

    plt.suptitle(
        f"NBACK — Grand average ERPs at {ecfg['label']}\n"
        f"Green=non-target · Blue=target hit · Red=target miss\n"
        f"Dotted=control · Solid=creatine · "
        f"Bold=grand average · Faint=individual participants · Negative up",
        fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(str(group_folder/f"group_nback_erp_{ecfg['fname']}.png"),
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: group_nback_erp_{ecfg['fname']}.png")

# ── Step 5: By-condition waveform plots ───────────────────────────────────────
print(f"\nSTEP 5: Waveform plots — by condition")

for ecfg in elec_cfgs:
    elec = ecfg['elec']
    fig, axes = plt.subplots(len(conditions), 1,
                             figsize=(12.6, len(conditions)*3.42), sharex=True)

    for ax, cond in zip(axes, conditions):
        for grp in groups:
            light_col = lighten(cond_colours[cond], 0.50)
            for pid, ev in ind_evokeds.get((grp, cond), []):
                d = get_ch(ev, elec)
                if d is None: continue
                ax.plot(ev.times*1000, d, color=light_col,
                        lw=IND_LW, ls=group_linestyles[grp],
                        alpha=IND_ALPHA, zorder=1)

            if (grp, cond) not in grand_avgs: continue
            ev = grand_avgs[(grp, cond)]
            d  = get_ch(ev, elec)
            if d is None: continue
            grp_n = len(ind_evokeds.get((grp, cond), []))
            ax.plot(ev.times*1000, d, color=cond_colours[cond],
                    lw=GA_LW, ls=group_linestyles[grp], alpha=GA_ALPHA,
                    zorder=5,
                    label=f"{group_labels[grp]} grand avg (N={grp_n})")

        shade(ax, ecfg['windows'])
        style(ax)
        ax.set_ylabel('Amplitude (µV)', fontsize=16)
        ax.set_title(cond_labels[cond], fontsize=17, loc='left',
                     color=cond_colours[cond])
        ax.legend(fontsize=13, loc='upper right')
        ticks(ax)

    # Synchronize y-limits across condition subplots for fair visual comparison (dynamic scale)
    ylims = [ax.get_ylim() for ax in axes]
    new_bottom = max(lim[0] for lim in ylims)
    new_top = min(lim[1] for lim in ylims)
    for ax in axes:
        ax.set_ylim(new_bottom, new_top)

    axes[-1].set_xlabel('Time (ms)', fontsize=15)
    plt.suptitle(
        f"NBACK — Control vs Creatine at {ecfg['label']}\n"
        f"Green=non-target · Blue=target hit · Red=target miss\n"
        f"Dotted=control · Solid=creatine · "
        f"Bold=grand average · Faint=individual participants · Negative up",
        fontsize=13, y=1.01)
    plt.tight_layout()
    fig.savefig(str(group_folder/
                    f"group_nback_erp_{ecfg['fname']}_by_condition.png"),
                dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: group_nback_erp_{ecfg['fname']}_by_condition.png")

# ── Step 6: Bar charts ────────────────────────────────────────────────────────
print(f"\nSTEP 6: Bar charts")

def make_bar_chart(comp_name, measure_col, ylabel, fname,
                   is_primary=True, invert_y=False, cond_subset=None):
    conds_to_plot = cond_subset if cond_subset else conditions
    w = max(6.5, 3.5 * len(conds_to_plot))
    h = 7.0
    fig, axes = plt.subplots(1, len(conds_to_plot),
                             figsize=(w, h), sharey=True)
    if len(conds_to_plot) == 1: axes = [axes]

    for ax, cond in zip(axes, conds_to_plot):
        ctrl_vals  = combined[(combined['component']==comp_name) &
                              (combined['condition']==cond) &
                              (combined['group']=='control')][measure_col].dropna().values
        creat_vals = combined[(combined['component']==comp_name) &
                              (combined['condition']==cond) &
                              (combined['group']=='creatine')][measure_col].dropna().values
        
        t, tp, u, up, d = run_tests(ctrl_vals, creat_vals)
        
        # Plot bars
        for gi, (grp, vals) in enumerate([('control', ctrl_vals), ('creatine', creat_vals)]):
            m   = np.mean(vals) if len(vals) > 0 else np.nan
            sem = np.std(vals, ddof=1)/np.sqrt(len(vals)) if len(vals) > 1 else 0
            sty = bar_styles[grp]
            if not np.isnan(m):
                ax.bar(gi, m, width=0.35,
                       facecolor=sty['facecolor'], edgecolor=sty['edgecolor'],
                       linewidth=sty['linewidth'], linestyle=sty['linestyle'],
                       label=f"{group_labels[grp]} (N={len(vals)})")
                ax.errorbar(gi, m, yerr=sem, fmt='none',
                            color='black', capsize=5, linewidth=1.5)
            # Add jittered individual points
            jitter = np.random.default_rng(42).uniform(-0.07, 0.07, len(vals))
            ax.scatter(gi+jitter, vals, color='#333333', s=40, zorder=5, alpha=0.85)

        ax.axhline(0, color='#cccccc', lw=0.8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([group_labels[g] for g in groups], fontsize=14)
        ax.set_xlim([-0.5, 1.5])
        ax.set_title(cond_labels.get(cond, cond), fontsize=13, pad=60)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Build stats text block (uncorrected)
        t_stars = f" ({stars(tp)})" if not np.isnan(tp) else ""
        u_stars = f" ({stars(up)})" if not np.isnan(up) else ""
        tp_str = f"{tp:.3f}" if not np.isnan(tp) else "n/a"
        up_str = f"{up:.3f}" if not np.isnan(up) else "n/a"
        d_str = f"{d:.2f}" if not np.isnan(d) else "n/a"
        
        stats_text = f"t-test: p = {tp_str}{t_stars}\nU-test: p = {up_str}{u_stars}\nd = {d_str}"
        ax.text(0.5, 1.02, stats_text, transform=ax.transAxes,
                ha='center', va='bottom', fontsize=11.5, color='#333333',
                fontweight='normal', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))

    # If y-axis is inverted
    if invert_y:
        axes[0].invert_yaxis()
    axes[0].set_ylabel(ylabel, fontsize=16)
    axes[0].tick_params(axis='y', which='major', labelsize=13)

    # Adjust y-limits with standard 10% visual margin at the top (no large 55% expansion)
    ylim = axes[0].get_ylim()
    v_bottom, v_top = ylim[0], ylim[1]
    span = v_top - v_bottom
    new_top = v_top + 0.10 * span
    new_bottom = v_bottom - 0.05 * span
    axes[0].set_ylim(new_bottom, new_top)

    type_label = "Primary Outcomes — Uncorrected statistics" if is_primary else \
                 "Exploratory Outcomes — Uncorrected statistics"

    ylabel_clean = ylabel
    if ylabel.startswith(comp_name):
        ylabel_clean = ylabel[len(comp_name):].strip()

    plt.suptitle(
         f"NBACK — {comp_name} {ylabel_clean}: Control vs Creatine\n"
         f"Dark grey dotted=control · Light grey solid=creatine\n"
         f"Bars=group mean · Error bars=SEM · Dots=individual participants\n"
         f"{type_label}",
         fontsize=12, fontweight='bold', y=0.99)
    plt.tight_layout(rect=[0.05, 0.08, 0.98, 0.88])
    fig.savefig(str(group_folder/fname), dpi=300, bbox_inches='tight', pad_inches=0.15)
    plt.close(fig)
    print(f"  Saved: {fname}")

# Primary amplitude
make_bar_chart('N200','mean_amp_uv','N200 mean amplitude (µV)',
               'group_nback_N200_amplitude_bars.png',
               is_primary=True, invert_y=True)
make_bar_chart('P300','mean_amp_uv','P300 mean amplitude (µV)',
               'group_nback_P300_amplitude_bars.png',
               is_primary=True, invert_y=False)

# Primary latency
make_bar_chart('N200','peak_lat_ms','N200 peak latency (ms)',
               'group_nback_N200_latency_bars.png',
               is_primary=True, invert_y=False)
make_bar_chart('P300','peak_lat_ms','P300 peak latency (ms)',
               'group_nback_P300_latency_bars.png',
               is_primary=True, invert_y=False)
make_bar_chart('P3b','peak_lat_ms','P3b peak latency (ms)',
               'group_nback_P3b_latency_bars.png',
               is_primary=True, invert_y=False)

# Exploratory
make_bar_chart('N1','mean_amp_uv','N1 mean amplitude (µV)',
               'group_nback_N1_amplitude_bars.png',
               is_primary=False, invert_y=True)
make_bar_chart('P2','mean_amp_uv','P2 mean amplitude (µV)',
               'group_nback_P2_amplitude_bars.png',
               is_primary=False, invert_y=False)
make_bar_chart('FSW','mean_amp_uv','Frontal slow wave amplitude (µV)',
               'group_nback_FSW_amplitude_bars.png',
               is_primary=False, invert_y=True,
               cond_subset=['target/hit'])

# ── Step 7: Target effect plot ────────────────────────────────────────────────
print(f"\nSTEP 7: Target effect plot")

effect_comps = ['N200','P300']
fig, axes    = plt.subplots(1, len(effect_comps), figsize=(7.5, 7.0))

for ax, comp in zip(axes, effect_comps):
    measure_col = 'mean_amp_uv'
    ctrl_eff, creat_eff = [], []
    for gi, grp in enumerate(groups):
        sub = combined[(combined['component']==comp) &
                       (combined['group']==grp)]
        for pid in sub['participant_id'].unique():
            p    = sub[sub['participant_id']==pid]
            hit  = p[p['condition']=='target/hit'][measure_col].values
            nont = p[p['condition']=='nontarget/correct'][measure_col].values
            if len(hit)>0 and len(nont)>0:
                eff = hit[0]-nont[0]
                if grp=='control':    ctrl_eff.append(eff)
                elif grp=='creatine': creat_eff.append(eff)

    ctrl_eff = np.array(ctrl_eff)
    creat_eff = np.array(creat_eff)
    t, tp, u, up, d = run_tests(ctrl_eff, creat_eff)

    for gi, (grp, eff_list) in enumerate([('control', ctrl_eff), ('creatine', creat_eff)]):
        mean_eff = np.mean(eff_list) if len(eff_list) > 0 else np.nan
        sem_eff  = np.std(eff_list, ddof=1)/np.sqrt(len(eff_list)) if len(eff_list) > 1 else 0
        sty      = bar_styles[grp]
        if not np.isnan(mean_eff):
            ax.bar(gi, mean_eff, width=0.35,
                   facecolor=sty['facecolor'], edgecolor=sty['edgecolor'],
                   linewidth=sty['linewidth'], linestyle=sty['linestyle'],
                   label=f"{group_labels[grp]} (N={len(eff_list)})")
            ax.errorbar(gi, mean_eff, yerr=sem_eff, fmt='none',
                        color='black', capsize=5, linewidth=1.5)
        jitter = np.random.default_rng(42).uniform(-0.07, 0.07, len(eff_list))
        ax.scatter(gi+jitter, eff_list, color='#333333', s=40, zorder=5, alpha=0.85)

    ax.axhline(0, color='#cccccc', lw=0.8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([group_labels[g] for g in groups], fontsize=14)
    ax.set_xlim([-0.5, 1.5])
    ax.set_title(f'{comp}\nTarget effect (Target hit − Non-target)', fontsize=13, pad=75)
    ax.set_ylabel('Amplitude difference (µV)', fontsize=16)
    ax.legend(fontsize=11, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Build stats text block
    t_stars = f" ({stars(tp)})" if not np.isnan(tp) else ""
    u_stars = f" ({stars(up)})" if not np.isnan(up) else ""
    tp_str = f"{tp:.3f}" if not np.isnan(tp) else "n/a"
    up_str = f"{up:.3f}" if not np.isnan(up) else "n/a"
    d_str = f"{d:.2f}" if not np.isnan(d) else "n/a"
    
    stats_text = f"t-test: p = {tp_str}{t_stars}\nU-test: p = {up_str}{u_stars}\nd = {d_str}"
    ax.text(0.5, 1.02, stats_text, transform=ax.transAxes,
            ha='center', va='bottom', fontsize=11.5, color='#333333',
            fontweight='normal', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))

    # Expand y-limits for the subplot by standard 10% on visually top side (no large 55% expansion)
    ylim = ax.get_ylim()
    v_bottom, v_top = ylim[0], ylim[1]
    span = v_top - v_bottom
    new_top = v_top + 0.10 * span
    new_bottom = v_bottom - 0.05 * span
    ax.set_ylim(new_bottom, new_top)

plt.suptitle(
    f"NBACK — Target effect (Target hit − Non-target correct)\n"
    f"Dark grey dotted=control · Light grey solid=creatine\n"
    f"Bars=group mean · Error bars=SEM · Dots=individual participants\n"
    f"Primary Outcomes — Uncorrected statistics",
    fontsize=12, fontweight='bold', y=0.99)
plt.tight_layout(rect=[0.05, 0.08, 0.98, 0.88])
fig.savefig(str(group_folder/'group_nback_target_effect.png'),
            dpi=300, bbox_inches='tight', pad_inches=0.15)
plt.close(fig)
print(f"  Saved: group_nback_target_effect.png")

# ── Step 8: P3b vs P300 latency scatter ───────────────────────────────────────
print(f"\nSTEP 8: P3b vs P300 latency scatter")

fig, axes = plt.subplots(1, len(conditions), figsize=(4.5*len(conditions), 4.5))

for ax, cond in zip(axes, conditions):
    for grp in groups:
        sub_p300 = combined[(combined['component']=='P300') &
                            (combined['condition']==cond) &
                            (combined['group']==grp)]
        sub_p3b  = combined[(combined['component']=='P3b') &
                            (combined['condition']==cond) &
                            (combined['group']==grp)]
        for pid in sub_p300['participant_id'].unique():
            l300 = sub_p300[sub_p300['participant_id']==pid]['peak_lat_ms'].values
            lp3b = sub_p3b[ sub_p3b[ 'participant_id']==pid]['peak_lat_ms'].values
            if len(l300)>0 and len(lp3b)>0:
                ax.scatter(l300[0], lp3b[0], color=group_colours[grp],
                           s=80, zorder=5, alpha=0.85,
                           label=group_labels[grp])
                ax.annotate(pid, (l300[0], lp3b[0]),
                            textcoords='offset points', xytext=(4,4),
                            fontsize=9, color=group_colours[grp])

    lims = [200, 700]
    ax.plot(lims, lims, '--', color='#cccccc', lw=1.2, zorder=1,
            label='Identity (P300=P3b)')
    ax.set_xlim(lims); ax.set_ylim(lims)
    ax.set_xlabel('P300 peak latency (ms)', fontsize=13)
    ax.set_ylabel('P3b peak latency (ms)', fontsize=14)
    ax.set_title(cond_labels.get(cond, cond), fontsize=13)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    handles, lbls = ax.get_legend_handles_labels()
    by_label = dict(zip(lbls, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=10)

plt.suptitle(
    f"NBACK — P3b latency vs P300 latency (300–600 ms, Pz)\n"
    f"Points on diagonal: P3b = P300 peak · "
    f"Points off diagonal: peaks diverge\n"
    f"Red=control · Blue=creatine",
    fontsize=12, y=1.04)
plt.tight_layout()
fig.savefig(str(group_folder/'group_nback_P3b_vs_P300_latency.png'),
            dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: group_nback_P3b_vs_P300_latency.png")

# ── Step 9: Group comparison topomaps — 5×9 grid ─────────────────────────────
# Layout: 5 columns (time points) × 9 rows
#   Rows 1–3 : Non-target  — Control / Creatine / Difference
#   Rows 4–6 : Target hit  — Control / Creatine / Difference
#   Rows 7–9 : Target miss — Control / Creatine / Difference
#
# Colour scales:
#   Amplitude rows (1,2,4,5,7,8): shared ±vmax across all amplitude rows
#   Difference rows (3,6,9):      shared ±vdiff symmetric around zero
#   Positive difference = creatine larger; negative = control larger
# ─────────────────────────────────────────────────────────────────────────────
print(f"\nSTEP 9: Group comparison topomaps (5×9 grid)")

topomap_times   = [0.120, 0.220, 0.350, 0.450, 0.550]
topomap_labels  = ['120 ms','220 ms','350 ms','450 ms','550 ms']
topo_conditions = [
    ('nontarget/correct', 'Non-target',  '#1D9E75'),
    ('target/hit',        'Target hit',  '#185FA5'),
    ('target/miss',       'Target miss', '#A32D2D'),
]
n_cols = len(topomap_times)       # 5
n_rows = len(topo_conditions) * 3  # 9

def get_topo_data(evoked, t, half_win=0.025):
    i0 = np.searchsorted(evoked.times, t - half_win)
    i1 = np.searchsorted(evoked.times, t + half_win)
    return evoked.data[:, i0:i1].mean(axis=1) * 1e6

# ── Shared colour scales ──────────────────────────────────────────────────────
amp_vals  = []
diff_vals = []

for cond, _, _ in topo_conditions:
    for t in topomap_times:
        ctrl_ev  = grand_avgs.get(('control',  cond))
        creat_ev = grand_avgs.get(('creatine', cond))
        if ctrl_ev is None or creat_ev is None:
            continue
        ctrl_data  = get_topo_data(ctrl_ev,  t)
        creat_data = get_topo_data(creat_ev, t)
        amp_vals.extend(np.abs(ctrl_data).tolist())
        amp_vals.extend(np.abs(creat_data).tolist())
        diff_vals.extend(np.abs(creat_data - ctrl_data).tolist())

vmax_amp  = np.percentile(amp_vals,  95) if amp_vals  else 5.0
vmax_diff = np.percentile(diff_vals, 95) if diff_vals else 2.0

print(f"  Amplitude scale : ±{vmax_amp:.2f} µV")
print(f"  Difference scale: ±{vmax_diff:.2f} µV")

# ── Build figure ──────────────────────────────────────────────────────────────
row_h   = 1.8
col_w   = 1.98
label_w = 1.62
cbar_h  = 0.315

fig_w = label_w + col_w * n_cols + 0.5
fig_h = row_h * n_rows + cbar_h * 2 + 1.2

fig = plt.figure(figsize=(fig_w, fig_h))

title_frac = 0.06
cbar_frac  = (cbar_h * 2) / fig_h

gs_main = plt.GridSpec(
    n_rows, n_cols,
    figure=fig,
    top=1.0 - title_frac - 0.01,
    bottom=cbar_frac + 0.06,
    left=label_w / fig_w,
    right=0.97,
    hspace=0.08,
    wspace=0.05,
)

last_amp_im  = None
last_diff_im = None

for cond_idx, (cond, cond_lbl, cond_col) in enumerate(topo_conditions):
    base_row = cond_idx * 3

    ctrl_ev  = grand_avgs.get(('control',  cond))
    creat_ev = grand_avgs.get(('creatine', cond))

    for col_idx, t in enumerate(topomap_times):

        # Row 0+base: Control
        ax = fig.add_subplot(gs_main[base_row, col_idx])
        if ctrl_ev is not None:
            data = get_topo_data(ctrl_ev, t)
            im, _ = mne.viz.plot_topomap(
                data, ctrl_ev.info, axes=ax, show=False,
                cmap='RdBu_r', contours=4,
                vlim=(-vmax_amp, vmax_amp)
            )
            last_amp_im = im
        if col_idx == 0:
            ax.set_ylabel(f"{cond_lbl}\nControl",
                          fontsize=11, color=cond_col,
                          fontweight='bold', labelpad=4)
        if cond_idx == 0:
            ax.set_title(topomap_labels[col_idx], fontsize=11, pad=3)

        # Row 1+base: Creatine
        ax = fig.add_subplot(gs_main[base_row + 1, col_idx])
        if creat_ev is not None:
            data = get_topo_data(creat_ev, t)
            im, _ = mne.viz.plot_topomap(
                data, creat_ev.info, axes=ax, show=False,
                cmap='RdBu_r', contours=4,
                vlim=(-vmax_amp, vmax_amp)
            )
            last_amp_im = im
        if col_idx == 0:
            ax.set_ylabel(f"{cond_lbl}\nCreatine",
                          fontsize=11, color=cond_col,
                          fontweight='bold', labelpad=4)

        # Row 2+base: Difference (creatine − control)
        ax = fig.add_subplot(gs_main[base_row + 2, col_idx])
        if ctrl_ev is not None and creat_ev is not None:
            ctrl_data  = get_topo_data(ctrl_ev,  t)
            creat_data = get_topo_data(creat_ev, t)
            diff_data  = creat_data - ctrl_data
            im, _ = mne.viz.plot_topomap(
                diff_data, ctrl_ev.info, axes=ax, show=False,
                cmap='PuOr', contours=4,
                vlim=(-vmax_diff, vmax_diff)
            )
            last_diff_im = im
        if col_idx == 0:
            ax.set_ylabel(f"{cond_lbl}\nDifference\n(Creat−Ctrl)",
                          fontsize=10, color='#444444',
                          fontstyle='italic', labelpad=4)

# Horizontal dividers between condition blocks
for divider_row in [3, 6]:
    y_top    = gs_main.get_subplot_params(fig).top
    y_bottom = gs_main.get_subplot_params(fig).bottom
    y_pos    = y_top - (divider_row / n_rows) * (y_top - y_bottom)
    fig.add_artist(
        plt.Line2D(
            [label_w/fig_w, 0.97],
            [y_pos, y_pos],
            transform=fig.transFigure,
            color='#bbbbbb', linewidth=0.8, linestyle='--'
        )
    )

# Colourbars
cb_bottom = 0.01
cb_height = 0.025

if last_amp_im is not None:
    cax_amp = fig.add_axes([0.10, cb_bottom + 0.04, 0.35, cb_height])
    cb_amp  = fig.colorbar(last_amp_im, cax=cax_amp, orientation='horizontal')
    cb_amp.set_label('Amplitude (µV)', fontsize=11)
    cb_amp.ax.tick_params(labelsize=10)

if last_diff_im is not None:
    cax_diff = fig.add_axes([0.55, cb_bottom + 0.04, 0.35, cb_height])
    cb_diff  = fig.colorbar(last_diff_im, cax=cax_diff, orientation='horizontal')
    cb_diff.set_label('Difference amplitude (µV)\nOrange = Creatine larger   '
                      'Purple = Control larger', fontsize=11)
    cb_diff.ax.tick_params(labelsize=10)

# Title
ctrl_n  = len(combined[combined['group']=='control' ]['participant_id'].unique())
creat_n = len(combined[combined['group']=='creatine']['participant_id'].unique())

fig.text(
    0.5, 0.995,
    f"NBACK — Group grand average topomaps: Control (N={ctrl_n}) vs "
    f"Creatine (N={creat_n})\n"
    f"Rows 1–2: amplitude (shared scale ±{vmax_amp:.1f} µV)  |  "
    f"Row 3: difference Creatine−Control (±{vmax_diff:.1f} µV)\n"
    f"Time points: 120, 220, 350, 450, 550 ms post-stimulus (±25 ms avg)",
    ha='center', va='top', fontsize=12, fontweight='bold'
)

topo_outpath = str(group_folder / 'group_nback_topomaps_group_comparison.png')
fig.savefig(topo_outpath, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"  Saved: group_nback_topomaps_group_comparison.png")
print(f"  Figure size : {fig_w:.1f} × {fig_h:.1f} inches  |  "
      f"Grid: {n_cols}×{n_rows}")

# ── Step 10: Save CSVs ────────────────────────────────────────────────────────
print(f"\nSTEP 10: Saving CSVs")
df_stats.to_csv(str(group_folder/'group_nback_summary.csv'), index=False)
combined.to_csv(str(group_folder/'group_nback_individual_components.csv'),
                index=False)
print(f"  Saved: group_nback_summary.csv")
print(f"  Saved: group_nback_individual_components.csv")

print("\n" + "="*60)
print(f"COMPLETE — Group ERP NBACK")
print(f"Finished : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*60)
print(f"\nParticipants: {sorted(combined['participant_id'].unique())}")
print(f"\nNext: python3 scripts/08_statistics_nback.py")
