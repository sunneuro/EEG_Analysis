import argparse
import mne
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats
from pathlib import Path
from datetime import datetime

# Configuration
primary_components_nback = {
    'N200': {'tmin':0.200,'tmax':0.350,'electrode':'FC1+FC2',
             'polarity':-1,'colour':'#185FA5'},
    'P300': {'tmin':0.300,'tmax':0.600,'electrode':'Pz',
             'polarity':1,'colour':'#A32D2D'},
    'P3b':  {'tmin':0.300,'tmax':0.600,'electrode':'Pz',
             'polarity':1,'colour':'#A32D2D'},
}
exploratory_components_nback = {
    'N1':  {'tmin':0.080,'tmax':0.160,'electrode':'FC1+FC2',
            'polarity':-1,'colour':'#6A0DAD'},
    'P2':  {'tmin':0.150,'tmax':0.250,'electrode':'Pz',
            'polarity':1,'colour':'#E88C2A'},
    'FSW': {'tmin':0.200,'tmax':0.500,'electrode':'Fz',
            'polarity':-1,'colour':'#2E8B57'},
}

primary_components_stroop = {
    'N200': {'tmin':0.200,'tmax':0.350,'electrode':'FC1+FC2',
             'polarity':-1,'colour':'#185FA5'},
    'P300': {'tmin':0.300,'tmax':0.600,'electrode':'Pz',
             'polarity':1,'colour':'#A32D2D'},
    'P3b':  {'tmin':0.300,'tmax':0.600,'electrode':'Pz',
             'polarity':1,'colour':'#A32D2D'},
}
exploratory_components_stroop = {
    'N1':  {'tmin':0.080,'tmax':0.160,'electrode':'FC1+FC2',
            'polarity':-1,'colour':'#6A0DAD'},
    'CSW': {'tmin':0.400,'tmax':0.700,'electrode':'FC1+FC2',
            'polarity':-1,'colour':'#1D7A6A'},
}

cond_colours_nback = {
    'nontarget/correct': '#1D9E75',
    'target/hit':        '#185FA5',
    'target/miss':       '#A32D2D',
}
cond_labels_nback = {
    'nontarget/correct': 'Non-target',
    'target/hit':        'Target hit',
    'target/miss':       'Target miss',
}

cond_colours_stroop = {
    'congruent/correct':   '#1D9E75',
    'incongruent/correct': '#A32D2D',
    'no_response':         '#888780',
}
cond_labels_stroop = {
    'congruent/correct':   'Congruent',
    'incongruent/correct': 'Incongruent',
    'no_response':         'No response',
}



IND_LW = 1.2
IND_ALPHA = 0.45
GA_LW = 2.8
GA_ALPHA = 1.0

def get_ch(evoked, elec):
    if '+' in elec:
        chs = elec.split('+')
        idx = [evoked.ch_names.index(c) for c in chs if c in evoked.ch_names]
        return np.mean([evoked.data[i] for i in idx], axis=0)*1e6 if idx else None
    return evoked.data[evoked.ch_names.index(elec)]*1e6 if elec in evoked.ch_names else None

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
        u, up = stats.mannwhitneyu(a, b, alternative='two-sided') if len(a)>=1 and len(b)>=1 else (np.nan, np.nan)
    except ValueError:
        u, up = np.nan, np.nan
    d = cohens_d(a, b)
    return t, tp, u, up, d

def stars(p):
    if np.isnan(p): return 'n/a'
    if p < 0.001: return '***'
    if p < 0.01: return '**'
    if p < 0.05: return '*'
    if p < 0.10: return '†'
    return 'ns'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['nback', 'stroop'], required=True)
    args = parser.parse_args()
    task = args.task

    print("=" * 60)
    print(f"GROUP ERP ANALYSIS — {task.upper()}")
    print(f"Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    erp_folder   = Path('output/erp')
    group_folder = Path('output/group')
    group_folder.mkdir(parents=True, exist_ok=True)

    participants_file = Path('data/participants.csv')
    participants      = pd.read_csv(participants_file)
    group_map         = dict(zip(participants['participant_id'], participants['group']))

    unique_groups = sorted([g for g in participants['group'].dropna().unique()])
    if len(unique_groups) != 2:
        print(f"ERROR: Expected exactly 2 groups in participants.csv, found {len(unique_groups)}: {unique_groups}")
        import sys; sys.exit(1)
        
    g1, g2 = unique_groups
    groups = [g1, g2]
    group_colours = {g1: '#A32D2D', g2: '#185FA5'}
    group_linestyles = {g1: ':', g2: '-'}
    group_labels = {g1: g1.capitalize(), g2: g2.capitalize()}
    
    bar_styles = {
        g1: {'facecolor':'#DDDDDD','edgecolor':'#333333','linewidth':2.5,'linestyle':':'},
        g2: {'facecolor':'#AAAAAA','edgecolor':'#333333','linewidth':2.5,'linestyle':'-'},
    }

    if task == 'nback':
        primary_components = primary_components_nback
        exploratory_components = exploratory_components_nback
        cond_labels = cond_labels_nback
        cond_colours = cond_colours_nback
        conditions = ['nontarget/correct','target/hit','target/miss']
        fc1fc2_windows = {'N1': exploratory_components['N1'], 'N200': primary_components['N200']}
        pz_windows = {'P2': exploratory_components['P2'], 'P300 / P3b': primary_components['P300']}
        elec_cfgs = [
            {'elec':'FC1+FC2','label':'FC1+FC2 (virtual FCz)','fname':'FC1_FC2',
             'windows':fc1fc2_windows, 'subtitle':'Purple=N1 (80–160ms) · Blue=N200 (200–350ms)'},
            {'elec':'Pz','label':'Pz','fname':'Pz',
             'windows':pz_windows, 'subtitle':'Orange=P2 (150–250ms) · Red=P300/P3b (300–600ms)'},
            {'elec':'Fz','label':'Fz (display only)','fname':'Fz',
             'windows':{}, 'subtitle':'Frontal slow wave — no component extraction'},
        ]
        subtitle_color = "Green=non-target · Blue=target hit · Red=target miss"
        topo_conditions = [
            ('nontarget/correct', 'Non-target',  '#1D9E75'),
            ('target/hit',        'Target hit',  '#185FA5'),
            ('target/miss',       'Target miss', '#A32D2D'),
        ]
    else:
        primary_components = primary_components_stroop
        exploratory_components = exploratory_components_stroop
        cond_labels = cond_labels_stroop
        cond_colours = cond_colours_stroop
        conditions = ['congruent/correct','incongruent/correct','no_response']
        fc1fc2_windows = {'N1': exploratory_components['N1'], 'N200': primary_components['N200'], 'CSW': exploratory_components['CSW']}
        pz_windows = {'P300 / P3b': primary_components['P300']}
        elec_cfgs = [
            {'elec':'FC1+FC2','label':'FC1+FC2 (virtual FCz)','fname':'FC1_FC2',
             'windows': fc1fc2_windows, 'subtitle':'Purple=N1 · Blue=N200 · Teal=CSW'},
            {'elec':'Pz','label':'Pz','fname':'Pz',
             'windows': pz_windows, 'subtitle':'Red=P300/P3b (300–600ms)'},
            {'elec':'Fz','label':'Fz (display only)','fname':'Fz',
             'windows': {}, 'subtitle':'Frontal midline — no component extraction'},
        ]
        subtitle_color = "Green=congruent · Red=incongruent · Grey=no response"
        topo_conditions = [
            ('congruent/correct',   'Congruent',   '#1D9E75'),
            ('incongruent/correct', 'Incongruent', '#A32D2D'),
            ('no_response',         'No response', '#888780'),
        ]

    all_components = {**primary_components, **exploratory_components}

    print(f"\nSTEP 1: Loading individual ERP component files")
    csv_files = sorted(erp_folder.glob(f'P*_{task}_erp_components.csv'))
    if not csv_files:
        print(f"ERROR: No component CSVs found. Run 06_erp.py --task {task} first.")
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

    print(f"\nSTEP 2: Group statistics")
    stats_rows = []
    for comp in all_components:
        for cond in conditions:
            for grp in groups:
                sub = combined[(combined['component']==comp) & (combined['condition']==cond) & (combined['group']==grp)]
                if sub.empty: continue
                n = len(sub)
                stats_rows.append({
                    'task':task,'component':comp,'condition':cond,'group':grp,'n':n,
                    'mean_amp_uv': round(sub['mean_amp_uv'].mean(),4) if sub['mean_amp_uv'].notna().any() else np.nan,
                    'sd_amp_uv': round(sub['mean_amp_uv'].std(),4) if sub['mean_amp_uv'].notna().any() else np.nan,
                    'sem_amp_uv': round(sub['mean_amp_uv'].std()/np.sqrt(n),4) if sub['mean_amp_uv'].notna().any() else np.nan,
                    'mean_peak_amp': round(sub['peak_amp_uv'].mean(),4) if sub['peak_amp_uv'].notna().any() else np.nan,
                    'mean_lat_ms': round(sub['peak_lat_ms'].mean(),1) if sub['peak_lat_ms'].notna().any() else np.nan,
                    'sd_lat_ms': round(sub['peak_lat_ms'].std(),1) if sub['peak_lat_ms'].notna().any() else np.nan,
                })

    df_stats = pd.DataFrame(stats_rows)

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

    print(f"\nSTEP 4: Waveform plots — side-by-side")
    for ecfg in elec_cfgs:
        elec = ecfg['elec']
        fig, axes = plt.subplots(1, 2, figsize=(14.4, 4.5), sharey=False)

        for ax, grp in zip(axes, groups):
            for cond in conditions:
                light_col = lighten(cond_colours[cond], 0.50)
                for pid, ev in ind_evokeds.get((grp, cond), []):
                    d = get_ch(ev, elec)
                    if d is None: continue
                    ax.plot(ev.times*1000, d, color=light_col, lw=IND_LW, alpha=IND_ALPHA, zorder=1)

            for cond in conditions:
                if (grp, cond) not in grand_avgs: continue
                ev = grand_avgs[(grp, cond)]
                d  = get_ch(ev, elec)
                if d is None: continue
                n_p = len(ind_evokeds.get((grp, cond), []))
                ax.plot(ev.times*1000, d, color=cond_colours[cond], lw=GA_LW, alpha=GA_ALPHA, zorder=5, label=f"{cond_labels[cond]} (N={n_p})")

            shade(ax, ecfg['windows'])
            style(ax)
            ax.set_xlabel('Time (ms)', fontsize=15)
            ax.set_ylabel('Amplitude (µV)', fontsize=16)
            ax.legend(fontsize=13, loc='upper right')
            grp_n = len(combined[combined['group']==grp]['participant_id'].unique())
            ax.set_title(f"{group_labels[grp]} group (N={grp_n})", fontsize=16, fontweight='bold', color=group_colours[grp])
            ticks(ax)

        ylims = [ax.get_ylim() for ax in axes]
        new_bottom = max(lim[0] for lim in ylims)
        new_top = min(lim[1] for lim in ylims)
        for ax in axes:
            ax.set_ylim(new_bottom, new_top)

        plt.suptitle(f"{task.upper()} — Grand average ERPs at {ecfg['label']}\n{subtitle_color}\nDotted={g1} · Solid={g2} · Bold=grand average · Faint=individual participants · Negative up", fontsize=13, y=1.02)
        plt.tight_layout()
        fig.savefig(str(group_folder/f"group_{task}_erp_{ecfg['fname']}.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: group_{task}_erp_{ecfg['fname']}.png")

    print(f"\nSTEP 5: Waveform plots — by condition")
    for ecfg in elec_cfgs:
        elec = ecfg['elec']
        fig, axes = plt.subplots(len(conditions), 1, figsize=(12.6, len(conditions)*3.42), sharex=True)

        for ax, cond in zip(axes, conditions):
            for grp in groups:
                light_col = lighten(cond_colours[cond], 0.50)
                for pid, ev in ind_evokeds.get((grp, cond), []):
                    d = get_ch(ev, elec)
                    if d is None: continue
                    ax.plot(ev.times*1000, d, color=light_col, lw=IND_LW, ls=group_linestyles[grp], alpha=IND_ALPHA, zorder=1)

                if (grp, cond) not in grand_avgs: continue
                ev = grand_avgs[(grp, cond)]
                d  = get_ch(ev, elec)
                if d is None: continue
                grp_n = len(ind_evokeds.get((grp, cond), []))
                ax.plot(ev.times*1000, d, color=cond_colours[cond], lw=GA_LW, ls=group_linestyles[grp], alpha=GA_ALPHA, zorder=5, label=f"{group_labels[grp]} grand avg (N={grp_n})")

            shade(ax, ecfg['windows'])
            style(ax)
            ax.set_ylabel('Amplitude (µV)', fontsize=16)
            ax.set_title(cond_labels[cond], fontsize=17, loc='left', color=cond_colours[cond])
            ax.legend(fontsize=13, loc='upper right')
            ticks(ax)

        ylims = [ax.get_ylim() for ax in axes]
        new_bottom = max(lim[0] for lim in ylims)
        new_top = min(lim[1] for lim in ylims)
        for ax in axes:
            ax.set_ylim(new_bottom, new_top)

        axes[-1].set_xlabel('Time (ms)', fontsize=15)
        plt.suptitle(f"{task.upper()} — {group_labels[g1]} vs {group_labels[g2]} at {ecfg['label']}\n{subtitle_color}\nDotted={g1} · Solid={g2} · Bold=grand average · Faint=individual participants · Negative up", fontsize=13, y=1.01)
        plt.tight_layout()
        fig.savefig(str(group_folder/f"group_{task}_erp_{ecfg['fname']}_by_condition.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  Saved: group_{task}_erp_{ecfg['fname']}_by_condition.png")

    print(f"\nSTEP 6: Bar charts")
    def make_bar_chart(comp_name, measure_col, ylabel, fname, is_primary=True, invert_y=False, cond_subset=None):
        conds_to_plot = cond_subset if cond_subset else conditions
        w = max(6.5, 3.5 * len(conds_to_plot))
        h = 7.0
        fig, axes = plt.subplots(1, len(conds_to_plot), figsize=(w, h), sharey=True)
        if len(conds_to_plot) == 1: axes = [axes]

        for ax, cond in zip(axes, conds_to_plot):
            g1_vals  = combined[(combined['component']==comp_name) & (combined['condition']==cond) & (combined['group']==g1)][measure_col].dropna().values
            g2_vals = combined[(combined['component']==comp_name) & (combined['condition']==cond) & (combined['group']==g2)][measure_col].dropna().values
            
            t_val, tp, u, up, d = run_tests(g1_vals, g2_vals)
            
            for gi, (grp, vals) in enumerate([(g1, g1_vals), (g2, g2_vals)]):
                m   = np.mean(vals) if len(vals) > 0 else np.nan
                sem = np.std(vals, ddof=1)/np.sqrt(len(vals)) if len(vals) > 1 else 0
                sty = bar_styles[grp]
                if not np.isnan(m):
                    ax.bar(gi, m, width=0.35, facecolor=sty['facecolor'], edgecolor=sty['edgecolor'], linewidth=sty['linewidth'], linestyle=sty['linestyle'], label=f"{group_labels[grp]} (N={len(vals)})")
                    ax.errorbar(gi, m, yerr=sem, fmt='none', color='black', capsize=5, linewidth=1.5)
                jitter = np.random.default_rng(42).uniform(-0.07, 0.07, len(vals))
                ax.scatter(gi+jitter, vals, color='#333333', s=40, zorder=5, alpha=0.85)

            ax.axhline(0, color='#cccccc', lw=0.8)
            ax.set_xticks([0, 1])
            ax.set_xticklabels([group_labels[g] for g in groups], fontsize=14)
            ax.set_xlim([-0.5, 1.5])
            ax.set_title(cond_labels.get(cond, cond), fontsize=13, pad=60)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            t_stars = f" ({stars(tp)})" if not np.isnan(tp) else ""
            u_stars = f" ({stars(up)})" if not np.isnan(up) else ""
            tp_str = f"{tp:.3f}" if not np.isnan(tp) else "n/a"
            up_str = f"{up:.3f}" if not np.isnan(up) else "n/a"
            d_str = f"{d:.2f}" if not np.isnan(d) else "n/a"
            
            stats_text = f"t-test: p = {tp_str}{t_stars}\nU-test: p = {up_str}{u_stars}\nd = {d_str}"
            ax.text(0.5, 1.02, stats_text, transform=ax.transAxes, ha='center', va='bottom', fontsize=11.5, color='#333333', fontweight='normal', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))

        if invert_y:
            axes[0].invert_yaxis()
        axes[0].set_ylabel(ylabel, fontsize=16)
        axes[0].tick_params(axis='y', which='major', labelsize=13)

        ylim = axes[0].get_ylim()
        v_bottom, v_top = ylim[0], ylim[1]
        span = v_top - v_bottom
        new_top = v_top + 0.10 * span
        new_bottom = v_bottom - 0.05 * span
        axes[0].set_ylim(new_bottom, new_top)

        type_label = "Primary Outcomes — Uncorrected statistics" if is_primary else "Exploratory Outcomes — Uncorrected statistics"
        ylabel_clean = ylabel
        if ylabel.startswith(comp_name):
            ylabel_clean = ylabel[len(comp_name):].strip()

        plt.suptitle(f"{task.upper()} — {comp_name} {ylabel_clean}: {group_labels[g1]} vs {group_labels[g2]}\nDark grey dotted={g1} · Light grey solid={g2}\nBars=group mean · Error bars=SEM · Dots=individual participants\n{type_label}", fontsize=12, fontweight='bold', y=0.99)
        plt.tight_layout(rect=[0.05, 0.08, 0.98, 0.88])
        fig.savefig(str(group_folder/fname), dpi=300, bbox_inches='tight', pad_inches=0.15)
        plt.close(fig)
        print(f"  Saved: {fname}")

    make_bar_chart('N200','mean_amp_uv','N200 mean amplitude (µV)', f'group_{task}_N200_amplitude_bars.png', is_primary=True, invert_y=True)
    make_bar_chart('P300','mean_amp_uv','P300 mean amplitude (µV)', f'group_{task}_P300_amplitude_bars.png', is_primary=True, invert_y=False)
    make_bar_chart('N200','peak_lat_ms','N200 peak latency (ms)', f'group_{task}_N200_latency_bars.png', is_primary=True, invert_y=False)
    make_bar_chart('P300','peak_lat_ms','P300 peak latency (ms)', f'group_{task}_P300_latency_bars.png', is_primary=True, invert_y=False)
    make_bar_chart('P3b','peak_lat_ms','P3b peak latency (ms)', f'group_{task}_P3b_latency_bars.png', is_primary=True, invert_y=False)
    make_bar_chart('N1','mean_amp_uv','N1 mean amplitude (µV)', f'group_{task}_N1_amplitude_bars.png', is_primary=False, invert_y=True)
    
    if task == 'nback':
        make_bar_chart('P2','mean_amp_uv','P2 mean amplitude (µV)', f'group_nback_P2_amplitude_bars.png', is_primary=False, invert_y=False)
        make_bar_chart('FSW','mean_amp_uv','Frontal slow wave amplitude (µV)', f'group_nback_FSW_amplitude_bars.png', is_primary=False, invert_y=True, cond_subset=['target/hit'])
    else:
        make_bar_chart('CSW','mean_amp_uv','CSW mean amplitude (µV)', f'group_stroop_CSW_amplitude_bars.png', is_primary=False, invert_y=True, cond_subset=['incongruent/correct'])

    print(f"\nSTEP 7: Effect plot")
    effect_comps = ['N200','P300']
    fig, axes = plt.subplots(1, len(effect_comps), figsize=(7.5, 7.0))

    for ax, comp in zip(axes, effect_comps):
        measure_col = 'mean_amp_uv'
        g1_eff, g2_eff = [], []
        for gi, grp in enumerate(groups):
            sub = combined[(combined['component']==comp) & (combined['group']==grp)]
            for pid in sub['participant_id'].unique():
                p = sub[sub['participant_id']==pid]
                if task == 'nback':
                    cond_a = p[p['condition']=='target/hit'][measure_col].values
                    cond_b = p[p['condition']=='nontarget/correct'][measure_col].values
                else:
                    cond_a = p[p['condition']=='incongruent/correct'][measure_col].values
                    cond_b = p[p['condition']=='congruent/correct'][measure_col].values
                    
                if len(cond_a)>0 and len(cond_b)>0:
                    eff = cond_a[0]-cond_b[0]
                    if grp==g1: g1_eff.append(eff)
                    elif grp==g2: g2_eff.append(eff)

        g1_eff = np.array(g1_eff)
        g2_eff = np.array(g2_eff)
        t_val, tp, u, up, d = run_tests(g1_eff, g2_eff)

        for gi, (grp, eff_list) in enumerate([(g1, g1_eff), (g2, g2_eff)]):
            mean_eff = np.mean(eff_list) if len(eff_list) > 0 else np.nan
            sem_eff  = np.std(eff_list, ddof=1)/np.sqrt(len(eff_list)) if len(eff_list) > 1 else 0
            sty      = bar_styles[grp]
            if not np.isnan(mean_eff):
                ax.bar(gi, mean_eff, width=0.35, facecolor=sty['facecolor'], edgecolor=sty['edgecolor'], linewidth=sty['linewidth'], linestyle=sty['linestyle'], label=f"{group_labels[grp]} (N={len(eff_list)})")
                ax.errorbar(gi, mean_eff, yerr=sem_eff, fmt='none', color='black', capsize=5, linewidth=1.5)
            jitter = np.random.default_rng(42).uniform(-0.07, 0.07, len(eff_list))
            ax.scatter(gi+jitter, eff_list, color='#333333', s=40, zorder=5, alpha=0.85)

        ax.axhline(0, color='#cccccc', lw=0.8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([group_labels[g] for g in groups], fontsize=14)
        ax.set_xlim([-0.5, 1.5])
        
        if task == 'nback':
            ax.set_title(f'{comp}\nTarget effect (Target hit − Non-target)', fontsize=13, pad=75)
        else:
            ax.set_title(f'{comp}\nStroop conflict effect (Incongruent − Congruent)', fontsize=13, pad=75)
            
        ax.set_ylabel('Amplitude difference (µV)', fontsize=16)
        ax.legend(fontsize=11, loc='upper right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        t_stars = f" ({stars(tp)})" if not np.isnan(tp) else ""
        u_stars = f" ({stars(up)})" if not np.isnan(up) else ""
        tp_str = f"{tp:.3f}" if not np.isnan(tp) else "n/a"
        up_str = f"{up:.3f}" if not np.isnan(up) else "n/a"
        d_str = f"{d:.2f}" if not np.isnan(d) else "n/a"
        
        stats_text = f"t-test: p = {tp_str}{t_stars}\nU-test: p = {up_str}{u_stars}\nd = {d_str}"
        ax.text(0.5, 1.02, stats_text, transform=ax.transAxes, ha='center', va='bottom', fontsize=11.5, color='#333333', fontweight='normal', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))

        ylim = ax.get_ylim()
        v_bottom, v_top = ylim[0], ylim[1]
        span = v_top - v_bottom
        new_top = v_top + 0.10 * span
        new_bottom = v_bottom - 0.05 * span
        ax.set_ylim(new_bottom, new_top)

    if task == 'nback':
        plt.suptitle(f"NBACK — Target effect (Target hit − Non-target correct)\nDark grey dotted={g1} · Light grey solid={g2}\nBars=group mean · Error bars=SEM · Dots=individual participants\nPrimary Outcomes — Uncorrected statistics", fontsize=12, fontweight='bold', y=0.99)
        fname = 'group_nback_target_effect.png'
    else:
        plt.suptitle(f"STROOP — Conflict effect (Incongruent − Congruent)\nDark grey dotted = {g1} · Light grey solid = {g2}\nBars=group mean · Error bars=SEM · Dots=individual participants\nPrimary Outcomes — Uncorrected statistics", fontsize=12, fontweight='bold', y=0.99)
        fname = 'group_stroop_conflict_effect.png'
        
    plt.tight_layout(rect=[0.05, 0.08, 0.98, 0.88])
    fig.savefig(str(group_folder/fname), dpi=300, bbox_inches='tight', pad_inches=0.15)
    plt.close(fig)
    print(f"  Saved: {fname}")

    print(f"\nSTEP 8: P3b vs P300 latency scatter")
    fig, axes = plt.subplots(1, len(conditions), figsize=(4.5*len(conditions), 4.5))

    for ax, cond in zip(axes, conditions):
        for grp in groups:
            sub_p300 = combined[(combined['component']=='P300') & (combined['condition']==cond) & (combined['group']==grp)]
            sub_p3b  = combined[(combined['component']=='P3b') & (combined['condition']==cond) & (combined['group']==grp)]
            for pid in sub_p300['participant_id'].unique():
                l300 = sub_p300[sub_p300['participant_id']==pid]['peak_lat_ms'].values
                lp3b = sub_p3b[sub_p3b['participant_id']==pid]['peak_lat_ms'].values
                if len(l300)>0 and len(lp3b)>0:
                    ax.scatter(l300[0], lp3b[0], color=group_colours[grp], s=80, zorder=5, alpha=0.85, label=group_labels[grp])
                    ax.annotate(pid, (l300[0], lp3b[0]), textcoords='offset points', xytext=(4,4), fontsize=9, color=group_colours[grp])

        lims = [200, 700]
        ax.plot(lims, lims, '--', color='#cccccc', lw=1.2, zorder=1, label='Identity (P300=P3b)')
        ax.set_xlim(lims); ax.set_ylim(lims)
        ax.set_xlabel('P300 peak latency (ms)', fontsize=13)
        ax.set_ylabel('P3b peak latency (ms)', fontsize=14)
        ax.set_title(cond_labels.get(cond, cond), fontsize=13)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        handles, lbls = ax.get_legend_handles_labels()
        by_label = dict(zip(lbls, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=10)

    diag_note = "Points on diagonal: P3b = P300 peak" if task == 'nback' else "Points on diagonal: P3b peak = P300 peak (same latency)"
    plt.suptitle(f"{task.upper()} — P3b latency vs P300 latency (300–600 ms, Pz)\n{diag_note} · Points off diagonal: peaks diverge\nRed={g1} · Blue={g2}", fontsize=12, y=1.04)
    plt.tight_layout()
    fig.savefig(str(group_folder/f'group_{task}_P3b_vs_P300_latency.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: group_{task}_P3b_vs_P300_latency.png")

    print(f"\nSTEP 9: Group comparison topomaps (5×9 grid)")
    topomap_times   = [0.120, 0.220, 0.350, 0.450, 0.550]
    topomap_labels  = ['120 ms','220 ms','350 ms','450 ms','550 ms']
    n_cols = len(topomap_times)
    n_rows = len(topo_conditions) * 3

    def get_topo_data(evoked, t, half_win=0.025):
        i0 = np.searchsorted(evoked.times, t - half_win)
        i1 = np.searchsorted(evoked.times, t + half_win)
        return evoked.data[:, i0:i1].mean(axis=1) * 1e6

    amp_vals  = []
    diff_vals = []

    for cond, _, _ in topo_conditions:
        for t in topomap_times:
            g1_ev  = grand_avgs.get((g1,  cond))
            g2_ev = grand_avgs.get((g2, cond))
            if g1_ev is None or g2_ev is None: continue
            g1_data  = get_topo_data(g1_ev,  t)
            g2_data = get_topo_data(g2_ev, t)
            amp_vals.extend(np.abs(g1_data).tolist())
            amp_vals.extend(np.abs(g2_data).tolist())
            diff_vals.extend(np.abs(g2_data - g1_data).tolist())

    vmax_amp  = np.percentile(amp_vals,  95) if amp_vals  else 5.0
    vmax_diff = np.percentile(diff_vals, 95) if diff_vals else 2.0

    print(f"  Amplitude scale : ±{vmax_amp:.2f} µV")
    print(f"  Difference scale: ±{vmax_diff:.2f} µV")

    row_h   = 1.8
    col_w   = 1.98
    label_w = 1.62
    cbar_h  = 0.315

    fig_w = label_w + col_w * n_cols + 0.5
    fig_h = row_h * n_rows + cbar_h * 2 + 1.2

    fig = plt.figure(figsize=(fig_w, fig_h))

    title_frac = 0.06
    cbar_frac  = (cbar_h * 2) / fig_h

    gs_main = plt.GridSpec(n_rows, n_cols, figure=fig, top=1.0 - title_frac - 0.01, bottom=cbar_frac + 0.06, left=label_w / fig_w, right=0.97, hspace=0.08, wspace=0.05)

    last_amp_im  = None
    last_diff_im = None

    for cond_idx, (cond, cond_lbl, cond_col) in enumerate(topo_conditions):
        base_row = cond_idx * 3
        g1_ev  = grand_avgs.get((g1,  cond))
        g2_ev = grand_avgs.get((g2, cond))

        for col_idx, t in enumerate(topomap_times):
            ax = fig.add_subplot(gs_main[base_row, col_idx])
            if g1_ev is not None:
                data = get_topo_data(g1_ev, t)
                im, _ = mne.viz.plot_topomap(data, g1_ev.info, axes=ax, show=False, cmap='RdBu_r', contours=4, vlim=(-vmax_amp, vmax_amp))
                last_amp_im = im
            if col_idx == 0:
                ax.set_ylabel(f"{cond_lbl}\n{group_labels[g1]}", fontsize=11, color=cond_col, fontweight='bold', labelpad=4)
            if cond_idx == 0:
                ax.set_title(topomap_labels[col_idx], fontsize=11, pad=3)

            ax = fig.add_subplot(gs_main[base_row + 1, col_idx])
            if g2_ev is not None:
                data = get_topo_data(g2_ev, t)
                im, _ = mne.viz.plot_topomap(data, g2_ev.info, axes=ax, show=False, cmap='RdBu_r', contours=4, vlim=(-vmax_amp, vmax_amp))
                last_amp_im = im
            if col_idx == 0:
                ax.set_ylabel(f"{cond_lbl}\n{group_labels[g2]}", fontsize=11, color=cond_col, fontweight='bold', labelpad=4)

            ax = fig.add_subplot(gs_main[base_row + 2, col_idx])
            if g1_ev is not None and g2_ev is not None:
                g1_data  = get_topo_data(g1_ev,  t)
                g2_data = get_topo_data(g2_ev, t)
                diff_data  = g2_data - g1_data
                im, _ = mne.viz.plot_topomap(diff_data, g1_ev.info, axes=ax, show=False, cmap='PuOr', contours=4, vlim=(-vmax_diff, vmax_diff))
                last_diff_im = im
            if col_idx == 0:
                ax.set_ylabel(f"{cond_lbl}\nDifference\n({group_labels[g2][:5]}−{group_labels[g1][:5]})", fontsize=10, color='#444444', fontstyle='italic', labelpad=4)

    for divider_row in [3, 6]:
        y_top    = gs_main.get_subplot_params(fig).top
        y_bottom = gs_main.get_subplot_params(fig).bottom
        y_pos    = y_top - (divider_row / n_rows) * (y_top - y_bottom)
        fig.add_artist(plt.Line2D([label_w/fig_w, 0.97], [y_pos, y_pos], transform=fig.transFigure, color='#bbbbbb', linewidth=0.8, linestyle='--'))

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
        cb_diff.set_label(f'Difference amplitude (µV)\nOrange = {group_labels[g2]} larger   |   Purple = {group_labels[g1]} larger', fontsize=11)
        cb_diff.ax.tick_params(labelsize=10)

    fig.text(0.5, 0.98, f"{task.upper()} — Group comparison topomaps\nShared amplitude scale (±{vmax_amp:.1f} µV) and difference scale (±{vmax_diff:.1f} µV)", fontsize=14, fontweight='bold', ha='center', va='top')
    fig.savefig(str(group_folder/f'group_{task}_topomaps_comparison_grid.png'), dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: group_{task}_topomaps_comparison_grid.png")

    print(f"\nSTEP 10: Saving summary CSVs")
    df_stats.to_csv(str(group_folder/f'group_{task}_summary.csv'), index=False)
    combined.to_csv(str(group_folder/f'group_{task}_individual_components.csv'), index=False)
    print(f"  Saved: group_{task}_summary.csv")
    print(f"  Saved: group_{task}_individual_components.csv")

    print("\n" + "="*60)
    print(f"COMPLETE — GROUP ERP ANALYSIS {task.upper()}  |  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    print(f"\nNext: python3 scripts/08_statistics.py --task {task}")

if __name__ == '__main__':
    main()
