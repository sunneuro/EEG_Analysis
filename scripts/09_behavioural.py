import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm
from pathlib import Path
from datetime import datetime

def run_nback():
    print("=" * 65)
    print(f"BEHAVIOURAL ANALYSIS — NBACK")
    print(f"Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)
    
    # ── Paths ─────────────────────────────────────────────────────────────────────
    behav_dir    = Path('data/behavioural/nback')
    group_folder = Path('output/group')
    stats_folder = Path('output/stats/nback_beh_output')
    stats_folder.mkdir(parents=True, exist_ok=True)
    
    # ── Study parameters ──────────────────────────────────────────────────────────
    participants_file = Path('data/participants.csv')
    participants      = pd.read_csv(participants_file) if participants_file.exists() else None
    
    if participants is not None:
        group_map = dict(zip(participants['participant_id'], participants['group']))
        unique_groups = sorted([g for g in participants['group'].dropna().unique()])
        g1, g2 = unique_groups if len(unique_groups) == 2 else ('group1', 'group2')
    else:
        group_map = {}
        g1, g2 = 'group1', 'group2'
        
    groups = [g1, g2]
    group_labels = {g1: g1.capitalize(), g2: g2.capitalize()}
    group_colours = {g1: '#185FA5', g2: '#A32D2D'}
    bar_styles    = {
        g1:  {'facecolor':'#DDDDDD','edgecolor':'#333333',
                     'linewidth':2.5,'linestyle':':'},
        g2: {'facecolor':'#AAAAAA','edgecolor':'#333333',
                     'linewidth':2.5,'linestyle':'-'},
    }
    load_labels = {'ZeroList':'0-back','OneList':'1-back','TwoList':'2-back'}
    
    # ── Helpers ───────────────────────────────────────────────────────────────────
    def find_col(df, *keywords):
        """Case-insensitive column finder."""
        for kw in keywords:
            matches = [c for c in df.columns if kw.lower() in c.lower()]
            if matches: return matches[0]
        return None
    
    def cohens_d(a, b):
        na, nb = len(a), len(b)
        if na < 2 or nb < 2: return np.nan
        s = np.sqrt(((na-1)*np.std(a,ddof=1)**2 +
                     (nb-1)*np.std(b,ddof=1)**2) / (na+nb-2))
        return np.nan if s==0 else (np.mean(a)-np.mean(b))/s
    
    def sig_stars(p):
        if np.isnan(p): return 'n/a'
        if p < 0.001: return '***'
        if p < 0.01:  return '**'
        if p < 0.05:  return '*'
        if p < 0.10:  return '†'
        return 'ns'
    
    def run_tests(a, b, label=''):
        na, nb = len(a), len(b)
        t = tp = u = up = np.nan
        if na>=2 and nb>=2: t, tp = stats.ttest_ind(a, b, equal_var=False)
        try:
            if na>=1 and nb>=1: u, up = stats.mannwhitneyu(a, b, alternative='two-sided')
        except ValueError: pass
        d = cohens_d(a, b)
        return {
            'comparison':    label,
            f'n_{g1}':     na,   f'n_{g2}':    nb,
            f'mean_{g1}':  round(float(np.mean(a)),3) if na>0 else np.nan,
            f'mean_{g2}': round(float(np.mean(b)),3) if nb>0 else np.nan,
            f'sd_{g1}':    round(float(np.std(a,ddof=1)),3) if na>1 else np.nan,
            f'sd_{g2}':   round(float(np.std(b,ddof=1)),3) if nb>1 else np.nan,
            f'sem_{g1}':   round(float(np.std(a,ddof=1)/np.sqrt(na)),3) if na>1 else np.nan,
            f'sem_{g2}':  round(float(np.std(b,ddof=1)/np.sqrt(nb)),3) if nb>1 else np.nan,
            't_stat':  round(float(t),3)  if not np.isnan(t)  else np.nan,
            't_p':     round(float(tp),4) if not np.isnan(tp) else np.nan,
            't_sig':   sig_stars(tp),
            'U_stat':  round(float(u),3)  if not np.isnan(u)  else np.nan,
            'U_p':     round(float(up),4) if not np.isnan(up) else np.nan,
            'U_sig':   sig_stars(up),
            'cohens_d':round(float(d),3)  if not np.isnan(d)  else np.nan,
            'd_interp':('small' if abs(d)<0.5 else 'medium' if abs(d)<0.8 else 'large')
                       if not np.isnan(d) else 'n/a',
        }
    
    def annotate_bar(ax, x0, x1, y_top, t_p, u_p):
        t_s = sig_stars(t_p); u_s = sig_stars(u_p)
        h   = abs(y_top)*0.10 if abs(y_top)>0 else 0.3
        ax.plot([x0,x0,x1,x1],[y_top,y_top+h,y_top+h,y_top],
                color='black', lw=1.2, clip_on=False)
        label = t_s if (t_s==u_s or u_s=='n/a') else f"{t_s}\n(U:{u_s})"
        ax.text((x0+x1)/2, y_top+h*0.2, label,
                ha='center', va='bottom', fontsize=10,
                fontweight='bold' if t_s not in ('ns','n/a') else 'normal')
    
    def draw_bars(ax, g1_v, g2_v, title):
        """Draw two bars with individual dots, SEM, and p-value annotation.
        Style matched to 09_EEG_variation_*.py."""
        g1_v  = np.array(g1_v)
        g2_v = np.array(g2_v)
        for gi, (grp, vals) in enumerate([(g1, g1_v), (g2, g2_v)]):
            m   = np.mean(vals) if len(vals) > 0 else np.nan
            sem = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0
            sty = bar_styles[grp]
            if not np.isnan(m):
                ax.bar(gi, m, width=0.35,
                       facecolor=sty['facecolor'], edgecolor=sty['edgecolor'],
                       linewidth=sty['linewidth'], linestyle=sty['linestyle'],
                       label=f"{group_labels[grp]} (N={len(vals)})")
                ax.errorbar(gi, m, yerr=sem, fmt='none',
                            color='black', capsize=5, linewidth=1.5)
            jitter = np.random.default_rng(42).uniform(-0.07, 0.07, len(vals))
            ax.scatter(gi + jitter, vals, color='#333333', s=40, zorder=5, alpha=0.85)
        ax.axhline(0, color='#cccccc', lw=0.8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([group_labels[g] for g in groups], fontsize=12)
        ax.set_xlim([-0.5, 1.5])
        ax.set_title(title, fontsize=11, fontweight='bold', pad=45)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        # Adjust y-limits: 10% visual margin at top, 5% at bottom (standard style)
        ylim = ax.get_ylim()
        span = ylim[1] - ylim[0]
        ax.set_ylim(ylim[0] - 0.05 * span, ylim[1] + 0.10 * span)
        # p-value text box (actual values, no bracket)
        r = run_tests(g1_v, g2_v)
        t_txt = f"t={r['t_stat']:.2f}, p={r['t_p']:.4f}" if not np.isnan(r['t_stat']) else "t: n/a"
        u_txt = f"U={r['U_stat']:.1f}, p={r['U_p']:.4f}" if not np.isnan(r['U_stat']) else "U: n/a"
        ax.text(0.5, 1.02, f"{t_txt}\n{u_txt}",
                transform=ax.transAxes, ha='center', va='bottom', fontsize=10,
                bbox=dict(facecolor='white', alpha=0.85, edgecolor='#cccccc',
                          boxstyle='round,pad=0.3'))
    
    def compute_dprime(hits, fas, n_targets, n_nontargets):
        if n_targets==0 or n_nontargets==0: return np.nan, np.nan
        hr = np.clip((hits+0.5)/(n_targets+1),    1e-6, 1-1e-6)
        fr = np.clip((fas +0.5)/(n_nontargets+1), 1e-6, 1-1e-6)
        return round(norm.ppf(hr)-norm.ppf(fr),3), \
               round(-0.5*(norm.ppf(hr)+norm.ppf(fr)),3)
    
    # ── Step 1: Load data ─────────────────────────────────────────────────────────
    print(f"\nSTEP 1: Loading behavioural data")
    
    dfs = []
    for pid, grp in group_map.items():
        fpath = behav_dir / f'{pid}_nback_clean.csv'
        if not fpath.exists():
            print(f"  WARNING: {fpath.name} not found"); continue
        df = pd.read_csv(fpath)
        df['participant_id'] = pid; df['group'] = grp
        dfs.append(df)
        print(f"  Loaded: {fpath.name}  ({len(df)} trials)")
    
    if not dfs:
        print("ERROR: No files found. Run 01_parse_eprime_nback.py first.")
        sys.exit(1)
    
    behav = pd.concat(dfs, ignore_index=True)
    
    # Auto-detect columns
    rt_col   = find_col(behav,'Stimulus.RT','StimulusRT','stimulus_rt','RT')
    acc_col  = find_col(behav,'Stimulus.ACC','StimulusACC','stimulus_acc','ACC')
    tt_col   = find_col(behav,'TrialType','trialtype','trial_type','Type')
    list_col = find_col(behav,'ListName','listname','list_name','List')
    miss_col = find_col(behav,'missed','miss')
    
    print(f"\n  Columns detected:")
    print(f"    RT         : {rt_col}")
    print(f"    Accuracy   : {acc_col}")
    print(f"    Trial type : {tt_col}")
    print(f"    List       : {list_col}")
    print(f"    Missed     : {miss_col}")
    
    # Normalise trial type values to title case
    if tt_col:
        behav[tt_col] = behav[tt_col].astype(str).str.strip().str.title()
        print(f"\n  Trial type values: {sorted(behav[tt_col].unique())}")
    
    # Detect actual trial type labels
    if tt_col:
        tt_vals = behav[tt_col].unique()
        target_val  = next((v for v in tt_vals if 'target' in v.lower()
                            and 'non' not in v.lower()), 'Target')
        nontarg_val = next((v for v in tt_vals if 'non' in v.lower()
                            or 'nontarget' in v.lower()), 'Non-Target')
    else:
        target_val = 'Target'; nontarg_val = 'Non-Target'
    
    print(f"\n  Target label    : '{target_val}'")
    print(f"  Non-target label: '{nontarg_val}'")
    trial_types = [target_val, nontarg_val]
    
    # Filter to experimental (non-practice) trials
    if list_col:
        prac = behav[list_col].astype(str).str.lower().str.contains('prac',na=False)
        behav_exp = behav[~prac].copy()
    else:
        behav_exp = behav.copy()
    
    behav_corr = behav_exp[behav_exp[acc_col]==1].copy() if acc_col else behav_exp.copy()
    print(f"\n  Experimental trials: {len(behav_exp)}")
    
    # Detect load levels present
    if list_col:
        load_levels_raw = [l for l in behav_exp[list_col].unique()
                           if 'prac' not in str(l).lower()]
        # Map to standard labels where possible
        load_levels_present = sorted(load_levels_raw,
                                     key=lambda x: load_labels.get(x, x))
        print(f"  Load levels: {load_levels_present}")
    else:
        load_levels_present = []
    
    # ── Step 2: Descriptive statistics ───────────────────────────────────────────
    print(f"\nSTEP 2: Descriptive statistics")
    
    desc_rows = []
    for pid in sorted(behav_exp['participant_id'].unique()):
        grp = group_map.get(pid,'?')
        sub = behav_exp[behav_exp['participant_id']==pid]
    
        for tt in trial_types + ['All']:
            sub_tt   = sub if tt=='All' else \
                       (sub[sub[tt_col]==tt] if tt_col else sub)
            sub_corr = sub_tt[sub_tt[acc_col]==1] if acc_col else sub_tt
    
            n_t      = len(sub_tt)
            n_c      = len(sub_corr)
            acc      = n_c/n_t if n_t>0 else np.nan
            miss_r   = (sub_tt[miss_col]==1).mean() \
                       if miss_col and miss_col in sub_tt.columns else np.nan
            mean_rt  = sub_corr[rt_col].mean()      if len(sub_corr)>0 else np.nan
            sd_rt    = sub_corr[rt_col].std(ddof=1) if len(sub_corr)>1 else np.nan
            med_rt   = sub_corr[rt_col].median()    if len(sub_corr)>0 else np.nan
            cv_rt    = (sd_rt/mean_rt*100) if (not np.isnan(mean_rt) and mean_rt>0
                                               and not np.isnan(sd_rt)) else np.nan
    
            # d-prime for target condition
            dp = crit = np.nan
            if tt == target_val and tt_col:
                n_tgt  = len(sub[sub[tt_col]==target_val])
                n_nt   = len(sub[sub[tt_col]==nontarg_val])
                hits   = len(sub[(sub[tt_col]==target_val) & (sub[acc_col]==1)])
                # false alarms = non-targets responded to (acc=0, had a response)
                fas    = len(sub[(sub[tt_col]==nontarg_val) &
                                 (sub[acc_col]==0) &
                                 (sub[rt_col].notna())])
                dp, crit = compute_dprime(hits, fas, n_tgt, n_nt)
    
            desc_rows.append({
                'participant_id': pid, 'group': grp,
                'trial_type': tt, 'load_level': 'All',
                'n_trials':    n_t,  'n_correct':  n_c,
                'accuracy':    round(acc,4)     if not np.isnan(acc)     else np.nan,
                'miss_rate':   round(miss_r,4)  if not np.isnan(miss_r)  else np.nan,
                'mean_RT_ms':  round(mean_rt,2) if not np.isnan(mean_rt) else np.nan,
                'sd_RT_ms':    round(sd_rt,2)   if not np.isnan(sd_rt)   else np.nan,
                'median_RT_ms':round(med_rt,2)  if not np.isnan(med_rt)  else np.nan,
                'cv_RT_pct':   round(cv_rt,2)   if not np.isnan(cv_rt)   else np.nan,
                'd_prime':     dp, 'criterion':  crit,
            })
    
        # By load level
        for load in load_levels_present:
            sub_l  = sub[sub[list_col]==load] if list_col else sub
            sub_lc = sub_l[sub_l[acc_col]==1] if acc_col else sub_l
            n_t    = len(sub_l); n_c = len(sub_lc)
            acc    = n_c/n_t if n_t>0 else np.nan
            mrt    = sub_lc[rt_col].mean() if len(sub_lc)>0 else np.nan
            desc_rows.append({
                'participant_id': pid, 'group': grp,
                'trial_type': 'All', 'load_level': load,
                'n_trials': n_t, 'n_correct': n_c,
                'accuracy': round(acc,4) if not np.isnan(acc) else np.nan,
                'miss_rate': np.nan,
                'mean_RT_ms': round(mrt,2) if mrt and not np.isnan(mrt) else np.nan,
                'sd_RT_ms': np.nan, 'median_RT_ms': np.nan,
                'cv_RT_pct': np.nan, 'd_prime': np.nan, 'criterion': np.nan,
            })
    
    df_desc = pd.DataFrame(desc_rows)
    df_desc.to_csv(str(stats_folder/'nback_behavioural_descriptives.csv'), index=False)
    print(f"  Saved: nback_behavioural_descriptives.csv")
    
    # Print summary
    print(f"\n{'PID':5s} {'Group':10s} {'Type':14s} {'Load':10s} "
          f"{'N':>5} {'Acc':>7} {'MeanRT':>8} {'d prime':>8}")
    print("-"*70)
    for _, r in df_desc[df_desc['load_level']=='All'].iterrows():
        dp_s = f"{r['d_prime']:.2f}"  if r['d_prime'] and not np.isnan(r['d_prime'])   else '—'
        rt_s = f"{r['mean_RT_ms']:.1f}" if r['mean_RT_ms'] and not np.isnan(r['mean_RT_ms']) else '—'
        ac_s = f"{r['accuracy']:.3f}" if r['accuracy'] and not np.isnan(r['accuracy']) else '—'
        print(f"  {r['participant_id']:3s}  {r['group']:10s} "
              f"{r['trial_type']:14s} {r['load_level']:10s} "
              f"{int(r['n_trials']) if not np.isnan(r['n_trials']) else 0:>5} "
              f"{ac_s:>7} {rt_s:>8} {dp_s:>8}")
    
    # ── Step 3: Group comparison statistics ──────────────────────────────────────
    print(f"\nSTEP 3: Group comparison statistics")
    
    sub_all_df = df_desc[df_desc['load_level']=='All']
    stat_rows  = []
    
    for tt in trial_types + ['All']:
        sub = sub_all_df[sub_all_df['trial_type']==tt]
        for meas, col in [('Mean RT (ms)','mean_RT_ms'),('Accuracy','accuracy'),
                           ('Miss rate','miss_rate'),("d-prime",'d_prime')]:
            g1_v  = sub[sub['group']==g1 ][col].dropna().values
            g2_v = sub[sub['group']==g2][col].dropna().values
            if len(g1_v)==0 and len(g2_v)==0: continue
            r = run_tests(g1_v, g2_v, f"Nback {meas} — {tt}")
            r['trial_type']=tt; r['load_level']='All'; r['measure']=meas
            stat_rows.append(r)
            print(f"\n  {r['comparison']}")
            print(f"    Ctrl  : M={r[f'mean_{g1}']:.3f}  SD={r[f'sd_{g1}']:.3f}  N={r[f'n_{g1}']}")
            print(f"    Creat : M={r[f'mean_{g2}']:.3f}  SD={r[f'sd_{g2}']:.3f}  N={r[f'n_{g2}']}")
            print(f"    t={r['t_stat']:.3f} p={r['t_p']:.4f} {r['t_sig']}  "
                  f"U p={r['U_p']:.4f} {r['U_sig']}  d={r['cohens_d']:.3f} ({r['d_interp']})")
    
    if load_levels_present:
        for load in load_levels_present:
            sub = df_desc[(df_desc['load_level']==load) & (df_desc['trial_type']=='All')]
            for meas, col in [('Mean RT (ms)','mean_RT_ms'),('Accuracy','accuracy')]:
                g1_v  = sub[sub['group']==g1 ][col].dropna().values
                g2_v = sub[sub['group']==g2][col].dropna().values
                if len(g1_v)==0 and len(g2_v)==0: continue
                r = run_tests(g1_v,g2_v,
                              f"Nback {meas} — {load_labels.get(load,load)}")
                r['trial_type']='All'; r['load_level']=load; r['measure']=meas
                stat_rows.append(r)
    
    pd.DataFrame(stat_rows).to_csv(
        str(stats_folder/'nback_behavioural_statistics.csv'), index=False)
    print(f"\n  Saved: nback_behavioural_statistics.csv")
    
    # ── Step 4: Bar charts (consolidated) ─────────────────────────────────────────
    print(f"\nSTEP 4: Bar charts")
    
    # ── Figure 1: Performance (RT + Accuracy) — 2 rows × N trial types ────────
    n_tt = len(trial_types)
    fig_perf, axes_perf = plt.subplots(2, n_tt,
                                        figsize=(3.3 * n_tt, 9.9), sharey='row')
    if n_tt == 1:
        axes_perf = axes_perf.reshape(2, 1)
    for ci, tt in enumerate(trial_types):
        sub = sub_all_df[sub_all_df['trial_type'] == tt]
        # Row 0: RT
        draw_bars(axes_perf[0, ci],
                  sub[sub['group'] == g1]['mean_RT_ms'].dropna().values,
                  sub[sub['group'] == g2]['mean_RT_ms'].dropna().values, tt)
        # Row 1: Accuracy
        draw_bars(axes_perf[1, ci],
                  sub[sub['group'] == g1]['accuracy'].dropna().values,
                  sub[sub['group'] == g2]['accuracy'].dropna().values, tt)
    axes_perf[0, 0].set_ylabel('Mean RT (ms)', fontsize=14)
    axes_perf[1, 0].set_ylabel('Accuracy (proportion)', fontsize=14)
    plt.suptitle("N-BACK BEHAVIOURAL PERFORMANCE \u2014 {group_labels[g1]} vs {group_labels[g2]}",
                 fontsize=13, fontweight='bold', y=0.98)
    fig_perf.text(0.5, 0.94,
        "Dotted/Light Grey = {group_labels[g1]} \u00b7 Solid/Darker Grey = {group_labels[g2]}\\n"
        "Bars = group mean \u00b7 Error bars = SEM \u00b7 Dots = individual participants\n"
        "Row 1 = Reaction Time (correct trials only) \u00b7 Row 2 = Accuracy",
        fontsize=9, style='italic', ha='center', va='top', color='#333333')
    plt.tight_layout(rect=[0.01, 0.08, 0.99, 0.89], h_pad=2.5)
    fig_perf.savefig(str(stats_folder / 'nback_behav_performance_bars.png'),
                     dpi=300, bbox_inches='tight', pad_inches=0.15)
    plt.close(fig_perf)
    print(f"  Saved: nback_behav_performance_bars.png")
    
    # ── Figure 2: Target sensitivity (Hit rate + Miss rate + d-prime) — 1×3 ───
    sub_tgt = sub_all_df[sub_all_df['trial_type'] == target_val]
    fig_sens, axes_sens = plt.subplots(1, 3, figsize=(11.0, 5.5), sharey=False)
    for ax_s, (mcol, ylabel_s, ttl) in zip(axes_sens, [
        ('accuracy',  'Hit rate (proportion)',  'Hit Rate'),
        ('miss_rate', 'Miss rate (proportion)', 'Miss Rate'),
        ('d_prime',   "d-prime",                "d\u2032"),
    ]):
        draw_bars(ax_s,
                  sub_tgt[sub_tgt['group'] == g1][mcol].dropna().values,
                  sub_tgt[sub_tgt['group'] == g2][mcol].dropna().values, ttl)
        ax_s.set_ylabel(ylabel_s, fontsize=14)
    plt.suptitle("N-BACK TARGET SENSITIVITY \u2014 {group_labels[g1]} vs {group_labels[g2]}",
                 fontsize=13, fontweight='bold', y=0.98)
    fig_sens.text(0.5, 0.94,
        "Dotted/Light Grey = {group_labels[g1]} \u00b7 Solid/Darker Grey = {group_labels[g2]}\\n"
        "Bars = group mean \u00b7 Error bars = SEM \u00b7 Dots = individual participants\n"
        "d\u2032 = Z(hit rate) \u2212 Z(false alarm rate) \u2014 Higher = better WM performance",
        fontsize=9, style='italic', ha='center', va='top', color='#333333')
    plt.tight_layout(rect=[0.01, 0.08, 0.99, 0.89], h_pad=2.5)
    fig_sens.savefig(str(stats_folder / 'nback_behav_sensitivity_bars.png'),
                     dpi=300, bbox_inches='tight', pad_inches=0.15)
    plt.close(fig_sens)
    print(f"  Saved: nback_behav_sensitivity_bars.png")
    
    # ── Figure 3: N-back load (RT + Accuracy) — 2 rows × n_loads ─────────────
    if load_levels_present:
        n_loads = len(load_levels_present)
        fig_load, axes_load = plt.subplots(2, n_loads,
                                            figsize=(3.3 * n_loads, 9.9), sharey='row')
        if n_loads == 1:
            axes_load = axes_load.reshape(2, 1)
        for li, load in enumerate(load_levels_present):
            sub_ld = df_desc[(df_desc['load_level'] == load) &
                             (df_desc['trial_type'] == 'All')]
            lbl = load_labels.get(load, load)
            draw_bars(axes_load[0, li],
                      sub_ld[sub_ld['group'] == g1]['mean_RT_ms'].dropna().values,
                      sub_ld[sub_ld['group'] == g2]['mean_RT_ms'].dropna().values, lbl)
            draw_bars(axes_load[1, li],
                      sub_ld[sub_ld['group'] == g1]['accuracy'].dropna().values,
                      sub_ld[sub_ld['group'] == g2]['accuracy'].dropna().values, lbl)
        axes_load[0, 0].set_ylabel('Mean RT (ms)', fontsize=14)
        axes_load[1, 0].set_ylabel('Accuracy (proportion)', fontsize=14)
        plt.suptitle("N-BACK LOAD EFFECTS \u2014 {group_labels[g1]} vs {group_labels[g2]}",
                     fontsize=13, fontweight='bold', y=0.98)
        fig_load.text(0.5, 0.94,
            "Dotted/Light Grey = {group_labels[g1]} \u00b7 Solid/Darker Grey = {group_labels[g2]}\\n"
            "Bars = group mean \u00b7 Error bars = SEM \u00b7 Dots = individual participants\n"
            "Row 1 = Reaction Time \u00b7 Row 2 = Accuracy",
            fontsize=9, style='italic', ha='center', va='top', color='#333333')
        plt.tight_layout(rect=[0.01, 0.08, 0.99, 0.89], h_pad=2.5)
        fig_load.savefig(str(stats_folder / 'nback_behav_load_bars.png'),
                         dpi=300, bbox_inches='tight', pad_inches=0.15)
        plt.close(fig_load)
        print(f"  Saved: nback_behav_load_bars.png")
    
    # ── Step 5: Behavioural variability comparison ────────────────────────────────
    print(f"\nSTEP 5: Behavioural variability (RT SD, RT CV)")
    
    var_rows = []
    fig_v, axes_v = plt.subplots(2, len(trial_types),
                                  figsize=(3.3 * len(trial_types), 9.9), sharey='row')
    if len(trial_types) == 1:
        axes_v = axes_v.reshape(2, 1)
    
    for ci, tt in enumerate(trial_types):
        sub = sub_all_df[sub_all_df['trial_type'] == tt]
        for mi, (meas_label, col) in enumerate([('RT SD (ms)', 'sd_RT_ms'),
                                                 ('RT CV (%)',  'cv_RT_pct')]):
            g1_v  = sub[sub['group'] == g1 ][col].dropna().values
            g2_v = sub[sub['group'] == g2][col].dropna().values
            # ── Variability metrics (identical formula to 09_EEG_variation_nback) ──
            sd_ctrl  = np.std(g1_v,  ddof=1) if len(g1_v)  > 1 else np.nan
            sd_creat = np.std(g2_v, ddof=1) if len(g2_v) > 1 else np.nan
            var_ratio = (sd_ctrl ** 2) / (sd_creat ** 2) \
                        if (not np.isnan(sd_ctrl) and not np.isnan(sd_creat)
                            and sd_creat > 0) else np.nan
            try:
                lev_s, lev_p = stats.levene(g1_v, g2_v, center='median') \
                               if len(g1_v) > 1 and len(g2_v) > 1 \
                               else (np.nan, np.nan)
            except Exception:
                lev_s, lev_p = np.nan, np.nan
            var_rows.append({
                'trial_type':   tt,   'measure':       meas_label,
                f'sd_{g1}':   round(float(sd_ctrl),  4) if not np.isnan(sd_ctrl)  else np.nan,
                f'sd_{g2}':  round(float(sd_creat), 4) if not np.isnan(sd_creat) else np.nan,
                'variance_ratio': round(float(var_ratio), 4) if not np.isnan(var_ratio) else np.nan,
                'levene_stat':  round(float(lev_s), 4) if not np.isnan(lev_s) else np.nan,
                'levene_p':     round(float(lev_p), 4) if not np.isnan(lev_p) else np.nan,
                'levene_sig':   sig_stars(lev_p),
            })
            print(f"  {meas_label} — {tt}")
            print(f"    Ctrl SD={sd_ctrl:.3f}  Creat SD={sd_creat:.3f}  "
                  f"VarRatio={var_ratio:.2f}  Levene p={lev_p:.4f}  {sig_stars(lev_p)}")
    
            ax = axes_v[mi, ci]
            for gi, (grp, vals) in enumerate([(g1, g1_v),
                                               (g2, g2_v)]):
                m   = np.mean(vals) if len(vals) > 0 else np.nan
                sem = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0
                sty = bar_styles[grp]
                if not np.isnan(m):
                    ax.bar(gi, m, width=0.35,
                           facecolor=sty['facecolor'], edgecolor=sty['edgecolor'],
                           linewidth=sty['linewidth'], linestyle=sty['linestyle'],
                           label=f"{group_labels[grp]} (N={len(vals)})")
                    ax.errorbar(gi, m, yerr=sem, fmt='none',
                                color='black', capsize=5, linewidth=1.5)
                jitter = np.random.default_rng(42).uniform(-0.07, 0.07, len(vals))
                ax.scatter(gi + jitter, vals, color='#333333', s=40, zorder=5, alpha=0.85)
            ax.axhline(0, color='#cccccc', lw=0.8)
            ax.set_xticks([0, 1])
            ax.set_xticklabels([group_labels[g] for g in groups], fontsize=10)
            ax.set_xlim([-0.5, 1.5])
            ax.set_title(f"{tt} — {meas_label}", fontsize=11, fontweight='bold', pad=50)
            ax.set_ylabel(meas_label, fontsize=14)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            # Adjust y-limits: 10% visual margin at top, 5% at bottom (standard style)
            ylim = ax.get_ylim()
            span = ylim[1] - ylim[0]
            ax.set_ylim(ylim[0] - 0.05 * span, ylim[1] + 0.10 * span)
            # Stats text block
            levene_stars = f" ({sig_stars(lev_p)})" if not np.isnan(lev_p) else ""
            p_str = f"{lev_p:.3f}" if not np.isnan(lev_p) else "n/a"
            vr_str = f"{var_ratio:.2f}" if not np.isnan(var_ratio) else "n/a"
            
            stats_text = (
                f"SD: Ctrl={sd_ctrl:.3f}, Creat={sd_creat:.3f}\n"
                f"Var Ratio (Ctrl/Creat) = {vr_str}\n"
                f"Levene p = {p_str} {sig_stars(lev_p)}"
            )
            ax.text(0.5, 1.02, stats_text, transform=ax.transAxes,
                    ha='center', va='bottom', fontsize=11, color='#333333',
                    fontweight='normal',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none',
                              boxstyle='round,pad=0.2'))
    
    plt.suptitle(
        "N-BACK BEHAVIOURAL RT VARIABILITY — {group_labels[g1]} vs {group_labels[g2]}",
        fontsize=13, fontweight='bold', y=0.98)
    fig_v.text(
        0.5, 0.94,
        "Dotted/Light Grey = {group_labels[g1]} · Solid/Darker Grey = {group_labels[g2]}\\n"
        "Bars = group mean · Error bars = SEM · Dots = individual participants\n"
        "SD = Standard Deviation · CV = Coefficient of Variation\n"
        "VR = Variance Ratio (Ctrl SD²/Creat SD²) · "
        "Levene's test (Brown-Forsythe) centered on median",
        fontsize=9, style='italic', ha='center', va='top', color='#333333'
    )
    plt.tight_layout(rect=[0.01, 0.08, 0.99, 0.89], h_pad=2.5)
    fig_v.savefig(str(stats_folder / 'nback_behav_variability_bars.png'),
                  dpi=300, bbox_inches='tight', pad_inches=0.15)
    plt.close(fig_v)
    print(f"  Saved: nback_behav_variability_bars.png")
    
    if var_rows:
        pd.DataFrame(var_rows).to_csv(
            str(stats_folder / 'nback_variability_behavioural.csv'), index=False)
        print(f"  Saved: nback_variability_behavioural.csv")
    
    # ── Step 6: Behavioural–ERP correlations ─────────────────────────────────────
    print(f"\nSTEP 6: Behavioural-ERP correlations (Spearman rho)")
    
    erp_file = group_folder / 'group_nback_individual_components.csv'
    if not erp_file.exists():
        print(f"  WARNING: {erp_file.name} not found.")
        print(f"           Run 07_group_erp_nback.py first then re-run this script.")
    else:
        erp_df = pd.read_csv(erp_file)
        # Normalise legacy component names (N2→N200, P3b→P300)
        erp_df['component'] = erp_df['component'].map(
            lambda x: {'N2':'N200','P3b':'P300','N1':'N1','P2':'P2','FSW':'FSW'}.get(x,x))
        print(f"  ERP data loaded: {len(erp_df)} rows")
        print(f"  Components: {sorted(erp_df['component'].unique())}")
        print(f"  Conditions: {sorted(erp_df['condition'].unique())}")
        print(f"  PIDs      : {sorted(erp_df['participant_id'].unique())}")
    
        # Build behavioural summary per participant
        behav_sum = {}
        for pid in sorted(behav_exp['participant_id'].unique()):
            sub    = behav_exp[behav_exp['participant_id']==pid]
            sub_tc = behav_corr[(behav_corr['participant_id']==pid) &
                                (behav_corr[tt_col]==target_val)] if tt_col else behav_corr
            sub_nt = sub[sub[tt_col]==nontarg_val] if tt_col else sub
            sub_tg = sub[sub[tt_col]==target_val]  if tt_col else sub
    
            hits    = len(sub_tg[sub_tg[acc_col]==1]) if acc_col else 0
            fas     = len(sub_nt[(sub_nt[acc_col]==0) &
                                 (sub_nt[rt_col].notna())]) if acc_col else 0
            dp, _   = compute_dprime(hits, fas, len(sub_tg), len(sub_nt))
            hit_rate= hits/len(sub_tg) if len(sub_tg)>0 else np.nan
            tgt_rt  = sub_tc[rt_col].mean() if len(sub_tc)>0 else np.nan
            ov_acc  = sub[acc_col].mean() if acc_col else np.nan
            miss_r  = 1-hit_rate if not np.isnan(hit_rate) else np.nan
    
            acc_2back = np.nan
            if list_col and 'TwoList' in sub[list_col].unique():
                sub_2 = sub[sub[list_col]=='TwoList']
                acc_2back = sub_2[acc_col].mean() if acc_col and len(sub_2)>0 else np.nan
    
            behav_sum[pid] = {
                'group':           group_map.get(pid,'?'),
                'd_prime':         dp,
                'hit_rate':        round(hit_rate,4) if not np.isnan(hit_rate) else np.nan,
                'miss_rate':       round(miss_r,4)   if not np.isnan(miss_r)   else np.nan,
                'target_RT':       round(tgt_rt,2)   if not np.isnan(tgt_rt)   else np.nan,
                'overall_accuracy':round(ov_acc,4)   if not np.isnan(ov_acc)   else np.nan,
                'accuracy_2back':  round(acc_2back,4) if not np.isnan(acc_2back) else np.nan,
            }
    
        # ── Pre-compute per-participant ERP absolute deviation from grand mean ──────
        # Formula identical to 09_EEG_variation_nback.py: SD = np.std(vals, ddof=1)
        # Grand mean = mean across all 8 participants for that component/condition/measure
        # Per-participant abs_dev = |val_i - grand_mean|  (their contribution to SD)
        erp_abs_dev = {}   # keyed (pid, comp, cond, measure) → float
        for (comp, cond, measure) in [
            ('P300', 'target/hit',  'mean_amp_uv'),
            ('P300', 'target/hit',  'peak_lat_ms'),
            ('N200', 'target/hit',  'mean_amp_uv'),
            ('N200', 'target/hit',  'peak_lat_ms'),
            ('N200', 'target/miss', 'mean_amp_uv'),
        ]:
            sub_erp = erp_df[(erp_df['component'] == comp) &
                             (erp_df['condition'] == cond)][['participant_id', measure]].dropna()
            grand_mean = sub_erp[measure].mean()  # mean across all participants
            # np.std(ddof=1) denominator basis: each deviation = |xi - mean|
            for _, row in sub_erp.iterrows():
                erp_abs_dev[(row['participant_id'], comp, cond, measure)] = \
                    abs(row[measure] - grand_mean)
        print(f"  ERP absolute deviation (|val - grand mean|) computed for "
              f"{len(erp_abs_dev)} participant-measure entries")
    
        # ── Per-participant behavioural variability for correlation ─────────────────
        # RT SD and RT CV come from the descriptives table (Step 2)
        bvar_sum = {}
        for pid in sorted(behav_exp['participant_id'].unique()):
            sub_tt = sub_all_df[(sub_all_df['participant_id'] == pid) &
                                (sub_all_df['trial_type'] == target_val) &
                                (sub_all_df['load_level'] == 'All')] \
                     if target_val in sub_all_df['trial_type'].values \
                     else pd.DataFrame()
            sub_all_pid = sub_all_df[(sub_all_df['participant_id'] == pid) &
                                      (sub_all_df['trial_type'] == 'All') &
                                      (sub_all_df['load_level'] == 'All')]
            rt_sd_tgt = sub_tt['sd_RT_ms'].values[0] \
                        if len(sub_tt) > 0 and not np.isnan(sub_tt['sd_RT_ms'].values[0]) \
                        else np.nan
            rt_cv_tgt = sub_tt['cv_RT_pct'].values[0] \
                        if len(sub_tt) > 0 and not np.isnan(sub_tt['cv_RT_pct'].values[0]) \
                        else np.nan
            rt_sd_all = sub_all_pid['sd_RT_ms'].values[0] \
                        if len(sub_all_pid) > 0 and not np.isnan(sub_all_pid['sd_RT_ms'].values[0]) \
                        else np.nan
            rt_cv_all = sub_all_pid['cv_RT_pct'].values[0] \
                        if len(sub_all_pid) > 0 and not np.isnan(sub_all_pid['cv_RT_pct'].values[0]) \
                        else np.nan
            bvar_sum[pid] = {
                'group':      group_map.get(pid, '?'),
                'rt_sd_tgt':  rt_sd_tgt,
                'rt_cv_tgt':  rt_cv_tgt,
                'rt_sd_all':  rt_sd_all,
                'rt_cv_all':  rt_cv_all,
            }
    
        corr_pairs = [
            # ── Existing pairs: behavioural performance vs ERP mean amplitude/latency ──
            ('d_prime',        "d-prime",
             'P300','target/hit','mean_amp_uv','P300 amplitude (µV)'),
            ('d_prime',        "d-prime",
             'N200','target/hit','mean_amp_uv','N200 amplitude (µV)'),
            ('d_prime',        "d-prime",
             'P300','target/hit','peak_lat_ms','P300 latency (ms)'),
            ('hit_rate',       'Hit rate',
             'P300','target/hit','mean_amp_uv','P300 amplitude (µV)'),
            ('hit_rate',       'Hit rate',
             'N200','target/hit','peak_lat_ms','N200 latency (ms)'),
            ('miss_rate',      'Miss rate',
             'N200','target/miss','mean_amp_uv','N200 amplitude (µV)'),
            ('target_RT',      'Target RT (ms)',
             'N200','target/hit','peak_lat_ms','N200 latency (ms)'),
            ('target_RT',      'Target RT (ms)',
             'P300','target/hit','peak_lat_ms','P300 latency (ms)'),
            ('accuracy_2back', '2-back accuracy',
             'P300','target/hit','mean_amp_uv','P300 amplitude (µV)'),
            ('accuracy_2back', '2-back accuracy',
             'N200','target/hit','mean_amp_uv','N200 amplitude (µV)'),
        ]
    
        corr_results = []
    
        def _compute_corr(bkey, blabel, comp, cond, emeas, elabel,
                          bsource, erp_source, var_type='mean'):
            """Compute one Spearman correlation. Returns (result_dict, arrays) or None."""
            b_vals, e_vals, pids_u, grps_u = [], [], [], []
            for pid, bdata in bsource.items():
                b_val = bdata.get(bkey, np.nan)
                if np.isnan(b_val): continue
                e_val = erp_source.get(pid, np.nan)
                if np.isnan(e_val): continue
                b_vals.append(b_val); e_vals.append(e_val)
                pids_u.append(pid); grps_u.append(bdata['group'])
            suffix = 'AbsDev' if var_type == 'abs_dev' else ''
            print(f"\n  {blabel} vs {elabel}{suffix}: N={len(b_vals)} pairs")
            if len(b_vals) < 4:
                print(f"    SKIP \u2014 fewer than 4 complete pairs")
                return None
            b_arr = np.array(b_vals); e_arr = np.array(e_vals)
            rho, p_val = stats.spearmanr(b_arr, e_arr)
            print(f"    \u03c1={rho:.3f}  p={p_val:.4f}  {sig_stars(p_val)}")
            res = {
                'behavioural_measure': blabel,
                'erp_component': comp, 'erp_condition': cond,
                'erp_measure': emeas, 'erp_label': elabel + suffix,
                'erp_variable_type': var_type,
                'n': len(b_vals), 'spearman_rho': round(rho, 3),
                'p_value': round(p_val, 4), 'significance': sig_stars(p_val),
            }
            return res, b_arr, e_arr, pids_u, grps_u, rho, p_val
    
        def _plot_scatter(ax, b_arr, e_arr, pids_u, grps_u, blabel, elabel,
                          rho, p_val, var_type='mean'):
            """Draw one scatter sub-panel (grid-friendly, smaller fonts)."""
            for bv, ev, pid, grp in zip(b_arr, e_arr, pids_u, grps_u):
                ax.scatter(bv, ev, color=group_colours[grp], s=50, zorder=5, alpha=0.9)
                ax.annotate(pid, (bv, ev), textcoords='offset points',
                            xytext=(4, 3), fontsize=6, color=group_colours[grp])
            if len(b_arr) >= 2:
                m_fit, b_fit_ = np.polyfit(b_arr, e_arr, 1)
                x_l = np.linspace(b_arr.min(), b_arr.max(), 100)
                ax.plot(x_l, m_fit * x_l + b_fit_,
                        color='#555555', lw=1.2, ls=':', alpha=0.7)
            ax.set_xlabel(blabel, fontsize=10)
            ylabel_full = f"{elabel} |dev. grand mean|" \
                          if var_type == 'abs_dev' else elabel
            ax.set_ylabel(ylabel_full, fontsize=10)
            ax.set_title(f"{blabel}\nvs {ylabel_full}", fontsize=9, fontweight='bold')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.text(0.05, 0.95,
                    f"\u03c1={rho:.3f}\np={p_val:.4f} {sig_stars(p_val)}\nN={len(b_arr)}",
                    transform=ax.transAxes, va='top', fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.85, edgecolor='#cccccc',
                              boxstyle='round,pad=0.2'))
            for grp in groups:
                ax.scatter([], [], color=group_colours[grp], s=30, label=group_labels[grp])
            ax.legend(fontsize=8, loc='lower right')
    
        # ── 6a: Performance vs ERP mean amplitude / latency ─────────────────────────
        corr_6a_data = []
        for (bkey, blabel, comp, cond, emeas, elabel) in corr_pairs:
            erp_lkp = {}
            for pid in behav_sum:
                erow = erp_df[(erp_df['participant_id'] == pid) &
                              (erp_df['component'] == comp) &
                              (erp_df['condition'] == cond)]
                if not erow.empty:
                    v = erow[emeas].values[0]
                    erp_lkp[pid] = v if not np.isnan(v) else np.nan
                else:
                    erp_lkp[pid] = np.nan
            out = _compute_corr(bkey, blabel, comp, cond, emeas, elabel,
                                behav_sum, erp_lkp, var_type='mean')
            if out is not None:
                res, b_arr, e_arr, pids_u, grps_u, rho, p_val = out
                corr_results.append(res)
                corr_6a_data.append((blabel, elabel, b_arr, e_arr, pids_u, grps_u,
                                     rho, p_val, 'mean'))
    
        # Grid figure for 6a — 2 columns × 5 rows
        if corr_6a_data:
            n_6a = len(corr_6a_data)
            nc = 2; nr = (n_6a + nc - 1) // nc
            fig_6a, axes_6a = plt.subplots(nr, nc, figsize=(4.95 * nc, 4.5 * nr))
            axes_6a = np.atleast_2d(axes_6a)
            for idx, (bl, el, ba, ea, pu, gu, rho, pv, vt) in enumerate(corr_6a_data):
                ri, ci = divmod(idx, nc)
                _plot_scatter(axes_6a[ri, ci], ba, ea, pu, gu, bl, el, rho, pv, vt)
            for idx in range(n_6a, nr * nc):
                ri, ci = divmod(idx, nc)
                axes_6a[ri, ci].set_visible(False)
            plt.suptitle(
                "N-BACK — Behavioural Performance vs ERP (Spearman)",
                fontsize=14, fontweight='bold', y=0.98)
            fig_6a.text(0.5, 0.94,
                "N=8 — interpret with caution · "
                "Dotted line = linear trend · PID labels on dots",
                fontsize=10, style='italic', ha='center', va='top', color='#333333')
            plt.tight_layout(rect=[0.05, 0.08, 0.98, 0.92])
            fig_6a.savefig(str(stats_folder / 'nback_corr_performance.png'),
                           dpi=300, bbox_inches='tight', pad_inches=0.15)
            plt.close(fig_6a)
            print(f"  Saved: nback_corr_performance.png")
    
        # ── 6b: Behavioural RT SD vs ERP absolute deviation ─────────────────────────
        # (SD only — CV% removed per request)
        print(f"\n  [6b] Behavioural RT SD vs ERP absolute deviation")
        var_corr_pairs = [
            ('rt_sd_tgt', 'Target RT SD (ms)',
             'P300', 'target/hit', 'mean_amp_uv', 'P300 amplitude'),
            ('rt_sd_tgt', 'Target RT SD (ms)',
             'N200', 'target/hit', 'mean_amp_uv', 'N200 amplitude'),
            ('rt_sd_tgt', 'Target RT SD (ms)',
             'P300', 'target/hit', 'peak_lat_ms', 'P300 latency'),
            ('rt_sd_tgt', 'Target RT SD (ms)',
             'N200', 'target/hit', 'peak_lat_ms', 'N200 latency'),
            ('rt_sd_all', 'Overall RT SD (ms)',
             'P300', 'target/hit', 'mean_amp_uv', 'P300 amplitude'),
        ]
        corr_6b_data = []
        for (bkey, blabel, comp, cond, emeas, elabel) in var_corr_pairs:
            erp_lkp = {}
            for pid in bvar_sum:
                key = (pid, comp, cond, emeas)
                erp_lkp[pid] = erp_abs_dev.get(key, np.nan)
            out = _compute_corr(bkey, blabel, comp, cond, emeas, elabel,
                                bvar_sum, erp_lkp, var_type='abs_dev')
            if out is not None:
                res, b_arr, e_arr, pids_u, grps_u, rho, p_val = out
                corr_results.append(res)
                corr_6b_data.append((blabel, elabel, b_arr, e_arr, pids_u, grps_u,
                                     rho, p_val, 'abs_dev'))
    
        # Grid figure for 6b — 3 columns
        if corr_6b_data:
            n_6b = len(corr_6b_data)
            nc = 3; nr = (n_6b + nc - 1) // nc
            fig_6b, axes_6b = plt.subplots(nr, nc, figsize=(4.95 * nc, 4.5 * nr))
            axes_6b = np.atleast_2d(axes_6b)
            for idx, (bl, el, ba, ea, pu, gu, rho, pv, vt) in enumerate(corr_6b_data):
                ri, ci = divmod(idx, nc)
                _plot_scatter(axes_6b[ri, ci], ba, ea, pu, gu, bl, el, rho, pv, vt)
            for idx in range(n_6b, nr * nc):
                ri, ci = divmod(idx, nc)
                axes_6b[ri, ci].set_visible(False)
            plt.suptitle(
                "N-BACK \u2014 Behavioural Variability vs ERP Absolute Deviation (Spearman)",
                fontsize=13, fontweight='bold', y=0.98)
            fig_6b.text(0.5, 0.93,
                "ERP variable = |ERP_i \u2212 grand mean| (np.std ddof=1, same as 09_EEG_variation)\n"
                "N=8 \u2014 interpret with caution \u00b7 Dotted line = linear trend",
                fontsize=9, style='italic', ha='center', va='top', color='#333333')
            plt.tight_layout(rect=[0.05, 0.08, 0.98, 0.90])
            fig_6b.savefig(str(stats_folder / 'nback_corr_variability.png'),
                           dpi=300, bbox_inches='tight', pad_inches=0.15)
            plt.close(fig_6b)
            print(f"  Saved: nback_corr_variability.png")
    
        if corr_results:
            pd.DataFrame(corr_results).to_csv(
                str(stats_folder / 'nback_behavioural_erp_correlations.csv'), index=False)
            print(f"\n  Saved: nback_behavioural_erp_correlations.csv")
        else:
            print(f"\n  No correlations computed \u2014 check ERP file exists and PIDs match")
    
    # ── Step 7: ERP variability — separate figure per component ───────────────────
    # (Removed - analyzed in 09_EEG_variation_nback.py)
    
    # ── Summary ───────────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"COMPLETE — Behavioural analysis NBACK")
    print(f"Finished : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*65}")
    print(f"\nOutputs in output/stats/nback_beh_output/:")
    print(f"  nback_behavioural_descriptives.csv")
    print(f"  nback_behavioural_statistics.csv")
    print(f"  nback_variability_behavioural.csv")
    print(f"  nback_behavioural_erp_correlations.csv")
    print(f"  nback_behav_performance_bars.png   (RT + Accuracy × trial type)")
    print(f"  nback_behav_sensitivity_bars.png   (Hit rate + Miss rate + d-prime)")
    print(f"  nback_behav_load_bars.png           (RT + Accuracy × n-back load)")
    print(f"  nback_behav_variability_bars.png   (RT SD + CV)")
    print(f"  nback_corr_performance.png         (6a scatter grid)")
    print(f"  nback_corr_variability.png         (6b scatter grid — |ERP_i-grand mean|)")
    

def run_stroop():
    print("=" * 65)
    print(f"BEHAVIOURAL ANALYSIS — STROOP")
    print(f"Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)
    
    # ── Paths ─────────────────────────────────────────────────────────────────────
    behav_dir    = Path('data/behavioural/stroop')
    group_folder = Path('output/group')
    stats_folder = Path('output/stats/stroop_beh_output')
    stats_folder.mkdir(parents=True, exist_ok=True)
    
    # ── Study parameters ──────────────────────────────────────────────────────────
    participants_file = Path('data/participants.csv')
    participants      = pd.read_csv(participants_file) if participants_file.exists() else None
    
    if participants is not None:
        group_map = dict(zip(participants['participant_id'], participants['group']))
        unique_groups = sorted([g for g in participants['group'].dropna().unique()])
        g1, g2 = unique_groups if len(unique_groups) == 2 else ('group1', 'group2')
    else:
        group_map = {}
        g1, g2 = 'group1', 'group2'
        
    groups = [g1, g2]
    group_labels = {g1: g1.capitalize(), g2: g2.capitalize()}
    group_colours = {g1: '#185FA5', g2: '#A32D2D'}
    bar_styles    = {
        g1:  {'facecolor':'#DDDDDD','edgecolor':'#333333',
                     'linewidth':2.5,'linestyle':':'},
        g2: {'facecolor':'#AAAAAA','edgecolor':'#333333',
                     'linewidth':2.5,'linestyle':'-'},
    }
    
    # ── Statistical helpers ───────────────────────────────────────────────────────
    def cohens_d(a, b):
        na, nb = len(a), len(b)
        if na < 2 or nb < 2: return np.nan
        s = np.sqrt(((na-1)*np.std(a,ddof=1)**2 +
                     (nb-1)*np.std(b,ddof=1)**2) / (na+nb-2))
        return np.nan if s == 0 else (np.mean(a)-np.mean(b)) / s
    
    def sig_stars(p):
        if np.isnan(p): return 'n/a'
        if p < 0.001: return '***'
        if p < 0.01:  return '**'
        if p < 0.05:  return '*'
        if p < 0.10:  return '†'
        return 'ns'
    
    def run_tests(a, b, label=''):
        na, nb = len(a), len(b)
        t = tp = u = up = np.nan
        if na >= 2 and nb >= 2:
            t, tp = stats.ttest_ind(a, b, equal_var=False)
        try:
            if na >= 1 and nb >= 1:
                u, up = stats.mannwhitneyu(a, b, alternative='two-sided')
        except ValueError:
            pass
        d = cohens_d(a, b)
        return {
            'comparison':    label,
            f'n_{g1}':     na,   f'n_{g2}':    nb,
            f'mean_{g1}':  round(float(np.mean(a)),3) if na>0 else np.nan,
            f'mean_{g2}': round(float(np.mean(b)),3) if nb>0 else np.nan,
            f'sd_{g1}':    round(float(np.std(a,ddof=1)),3) if na>1 else np.nan,
            f'sd_{g2}':   round(float(np.std(b,ddof=1)),3) if nb>1 else np.nan,
            f'sem_{g1}':   round(float(np.std(a,ddof=1)/np.sqrt(na)),3) if na>1 else np.nan,
            f'sem_{g2}':  round(float(np.std(b,ddof=1)/np.sqrt(nb)),3) if nb>1 else np.nan,
            't_stat':        round(float(t),3)  if not np.isnan(t)  else np.nan,
            't_p':           round(float(tp),4) if not np.isnan(tp) else np.nan,
            't_sig':         sig_stars(tp),
            'U_stat':        round(float(u),3)  if not np.isnan(u)  else np.nan,
            'U_p':           round(float(up),4) if not np.isnan(up) else np.nan,
            'U_sig':         sig_stars(up),
            'cohens_d':      round(float(d),3)  if not np.isnan(d)  else np.nan,
            'd_interp':      ('small' if abs(d)<0.5 else
                              'medium' if abs(d)<0.8 else 'large')
                             if not np.isnan(d) else 'n/a',
        }
    
    def annotate_bar(ax, x0, x1, y_top, t_p, u_p):
        t_s = sig_stars(t_p)
        u_s = sig_stars(u_p)
        h   = abs(y_top) * 0.10 if abs(y_top) > 0 else 0.5
        ax.plot([x0,x0,x1,x1],[y_top,y_top+h,y_top+h,y_top],
                color='black', lw=1.2, clip_on=False)
        label = t_s if (t_s==u_s or u_s=='n/a') else f"{t_s}\n(U:{u_s})"
        ax.text((x0+x1)/2, y_top+h*0.2, label,
                ha='center', va='bottom', fontsize=12,
                fontweight='bold' if t_s not in ('ns','n/a') else 'normal')
    
    def draw_bars(ax, g1_vals, g2_vals, cond_title):
        """Draw two bars (g1 left, g2 right) with dots and SEM.
        Style matched to 09_EEG_variation_*.py."""
        for gi, (grp, vals) in enumerate([(g1, g1_vals),
                                           (g2, g2_vals)]):
            vals = np.array(vals)
            m    = np.mean(vals) if len(vals) > 0 else np.nan
            sem  = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0
            sty  = bar_styles[grp]
            if not np.isnan(m):
                ax.bar(gi, m, width=0.35,
                       facecolor=sty['facecolor'], edgecolor=sty['edgecolor'],
                       linewidth=sty['linewidth'], linestyle=sty['linestyle'],
                       label=f"{group_labels[grp]} (N={len(vals)})")
                ax.errorbar(gi, m, yerr=sem, fmt='none',
                            color='black', capsize=5, linewidth=1.5)
            jitter = np.random.default_rng(42).uniform(-0.07, 0.07, len(vals))
            ax.scatter(gi + jitter, vals, color='#333333', s=40, zorder=5, alpha=0.85)
        ax.axhline(0, color='#cccccc', lw=0.8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([group_labels[g] for g in groups], fontsize=12)
        ax.set_xlim([-0.5, 1.5])
        ax.set_title(cond_title, fontsize=13, fontweight='bold', pad=45)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='y', which='major', labelsize=11)
        # Adjust y-limits: 10% visual margin at top, 5% at bottom (standard style)
        ylim = ax.get_ylim()
        span = ylim[1] - ylim[0]
        ax.set_ylim(ylim[0] - 0.05 * span, ylim[1] + 0.10 * span)
        # p-value text box (actual values, no bracket)
        r = run_tests(np.array(g1_vals), np.array(g2_vals))
        t_txt = f"t={r['t_stat']:.2f}, p={r['t_p']:.4f}" if not np.isnan(r['t_stat']) else "t: n/a"
        u_txt = f"U={r['U_stat']:.1f}, p={r['U_p']:.4f}" if not np.isnan(r['U_stat']) else "U: n/a"
        ax.text(0.5, 1.02, f"{t_txt}\n{u_txt}",
                transform=ax.transAxes, ha='center', va='bottom', fontsize=10,
                bbox=dict(facecolor='white', alpha=0.85, edgecolor='#cccccc',
                          boxstyle='round,pad=0.3'))
        return r
    
    # ── Step 1: Load data ─────────────────────────────────────────────────────────
    print(f"\nSTEP 1: Loading behavioural data")
    
    dfs = []
    for pid, grp in group_map.items():
        fpath = behav_dir / f'{pid}_stroop_clean.csv'
        if not fpath.exists():
            print(f"  WARNING: {fpath.name} not found — skipping {pid}")
            continue
        df = pd.read_csv(fpath)
        df['participant_id'] = pid
        df['group']          = grp
        dfs.append(df)
        print(f"  Loaded: {fpath.name}  ({len(df)} trials)")
    
    if not dfs:
        print("ERROR: No behavioural files found. Run 02_parse_eprime_stroop.py first.")
        sys.exit(1)
    
    behav = pd.concat(dfs, ignore_index=True)
    
    # Auto-detect column names (case-insensitive)
    def find_col(df, *keywords):
        for kw in keywords:
            matches = [c for c in df.columns if kw.lower() in c.lower()]
            if matches: return matches[0]
        return None
    
    rt_col   = find_col(behav, 'Stimulus3.RT', 'stimulus3rt', 'RT')
    acc_col  = find_col(behav, 'Stimulus3.ACC', 'stimulus3acc', 'ACC')
    cong_col = find_col(behav, 'Congruency', 'congruent')
    miss_col = find_col(behav, 'missed', 'miss')
    
    print(f"\n  Columns detected:")
    print(f"    RT         : {rt_col}")
    print(f"    Accuracy   : {acc_col}")
    print(f"    Congruency : {cong_col}")
    print(f"    Missed     : {miss_col}")
    
    # Normalise congruency values to title case
    if cong_col:
        behav[cong_col] = behav[cong_col].astype(str).str.strip().str.title()
        print(f"\n  Congruency values: {sorted(behav[cong_col].unique())}")
    
    conditions   = ['Congruent','Incongruent']
    behav_corr   = behav[behav[acc_col]==1].copy() if acc_col else behav.copy()
    
    # ── Step 2: Descriptive statistics ───────────────────────────────────────────
    print(f"\nSTEP 2: Descriptive statistics")
    
    desc_rows = []
    for pid in sorted(behav['participant_id'].unique()):
        grp = group_map.get(pid,'?')
        sub = behav[behav['participant_id']==pid]
    
        for cond in conditions + ['Overall']:
            if cond == 'Overall':
                sub_c = sub
            else:
                sub_c = sub[sub[cong_col]==cond] if cong_col else sub
    
            sub_corr  = sub_c[sub_c[acc_col]==1] if acc_col else sub_c
            n_t       = len(sub_c)
            n_correct = len(sub_corr)
            accuracy  = n_correct/n_t if n_t>0 else np.nan
            miss_rate = (sub_c[miss_col]==1).mean() \
                        if miss_col and miss_col in sub_c.columns else np.nan
            mean_rt   = sub_corr[rt_col].mean()   if len(sub_corr)>0 else np.nan
            sd_rt     = sub_corr[rt_col].std(ddof=1) if len(sub_corr)>1 else np.nan
            median_rt = sub_corr[rt_col].median() if len(sub_corr)>0 else np.nan
            cv_rt     = (sd_rt/mean_rt*100) if (not np.isnan(mean_rt) and mean_rt>0
                                                and not np.isnan(sd_rt)) else np.nan
    
            desc_rows.append({
                'participant_id': pid, 'group': grp, 'condition': cond,
                'n_trials':    n_t,
                'n_correct':   n_correct,
                'accuracy':    round(accuracy,4)  if not np.isnan(accuracy)  else np.nan,
                'miss_rate':   round(miss_rate,4) if not np.isnan(miss_rate) else np.nan,
                'mean_RT_ms':  round(mean_rt,2)   if not np.isnan(mean_rt)   else np.nan,
                'sd_RT_ms':    round(sd_rt,2)     if not np.isnan(sd_rt)     else np.nan,
                'median_RT_ms':round(median_rt,2) if not np.isnan(median_rt) else np.nan,
                'cv_RT_pct':   round(cv_rt,2)     if not np.isnan(cv_rt)     else np.nan,
            })
    
        # Stroop effect row
        rt_i  = behav_corr[(behav_corr['participant_id']==pid) &
                            (behav_corr[cong_col]=='Incongruent')][rt_col].mean() \
                if cong_col else np.nan
        rt_c  = behav_corr[(behav_corr['participant_id']==pid) &
                            (behav_corr[cong_col]=='Congruent')][rt_col].mean() \
                if cong_col else np.nan
        acc_i = sub[sub[cong_col]=='Incongruent'][acc_col].mean() if cong_col else np.nan
        acc_c = sub[sub[cong_col]=='Congruent'][acc_col].mean()   if cong_col else np.nan
    
        desc_rows.append({
            'participant_id': pid, 'group': grp, 'condition': 'Stroop_RT_effect',
            'n_trials': np.nan, 'n_correct': np.nan,
            'accuracy':    round(acc_c-acc_i,4) if not (np.isnan(acc_c) or np.isnan(acc_i)) else np.nan,
            'miss_rate':   np.nan,
            'mean_RT_ms':  round(rt_i-rt_c,2)  if not (np.isnan(rt_i)  or np.isnan(rt_c))  else np.nan,
            'sd_RT_ms': np.nan, 'median_RT_ms': np.nan, 'cv_RT_pct': np.nan,
        })
    
    df_desc = pd.DataFrame(desc_rows)
    df_desc.to_csv(str(stats_folder/'stroop_behavioural_descriptives.csv'), index=False)
    print(f"  Saved: stroop_behavioural_descriptives.csv")
    
    # Print summary
    print(f"\n{'PID':5s} {'Group':10s} {'Cond':20s} {'N':>5} "
          f"{'Acc':>7} {'MeanRT':>8} {'SD_RT':>8} {'CV%':>7}")
    print("-"*65)
    for _, r in df_desc.iterrows():
        n_str  = str(int(r['n_trials'])) if not np.isnan(r['n_trials']) else '—'
        ac_str = f"{r['accuracy']:.3f}"  if not np.isnan(r['accuracy']) else '—'
        rt_str = f"{r['mean_RT_ms']:.1f}" if not np.isnan(r['mean_RT_ms']) else '—'
        sd_str = f"{r['sd_RT_ms']:.1f}"  if not np.isnan(r['sd_RT_ms']) else '—'
        cv_str = f"{r['cv_RT_pct']:.1f}" if not np.isnan(r['cv_RT_pct']) else '—'
        print(f"  {r['participant_id']:3s}  {r['group']:10s} "
              f"{r['condition']:20s} {n_str:>5} "
              f"{ac_str:>7} {rt_str:>8} {sd_str:>8} {cv_str:>7}")
    
    # ── Step 3: Group comparison statistics ──────────────────────────────────────
    print(f"\nSTEP 3: Group comparison statistics")
    
    stat_rows = []
    for cond in conditions + ['Stroop_RT_effect']:
        sub = df_desc[df_desc['condition']==cond]
        for measure, col in [('Mean RT (ms)','mean_RT_ms'),
                              ('Accuracy','accuracy'),
                              ('Miss rate','miss_rate')]:
            g1_v  = sub[sub['group']==g1 ][col].dropna().values
            g2_v = sub[sub['group']==g2][col].dropna().values
            if len(g1_v)==0 and len(g2_v)==0: continue
            r = run_tests(g1_v, g2_v, f"Stroop {measure} — {cond}")
            r['condition'] = cond; r['measure'] = measure
            stat_rows.append(r)
            print(f"\n  {r['comparison']}")
            print(f"    Ctrl  : M={r[f'mean_{g1}']:.3f}  SD={r[f'sd_{g1}']:.3f}  N={r[f'n_{g1}']}")
            print(f"    Creat : M={r[f'mean_{g2}']:.3f}  SD={r[f'sd_{g2}']:.3f}  N={r[f'n_{g2}']}")
            print(f"    t={r['t_stat']:.3f} p={r['t_p']:.4f} {r['t_sig']}  "
                  f"U p={r['U_p']:.4f} {r['U_sig']}  d={r['cohens_d']:.3f} ({r['d_interp']})")
    
    pd.DataFrame(stat_rows).to_csv(
        str(stats_folder/'stroop_behavioural_statistics.csv'), index=False)
    print(f"\n  Saved: stroop_behavioural_statistics.csv")
    
    # ── Step 4: Bar charts (consolidated) ─────────────────────────────────────────
    print(f"\nSTEP 4: Bar charts")
    
    # ── Figure 1: Performance + Stroop Effect — 4 rows × N conditions ──────────
    # Rows: RT, Accuracy, Miss rate, Stroop effect (only 2 panels in row 4)
    n_cond = len(conditions)
    fig_perf, axes_perf = plt.subplots(4, n_cond,
                                        figsize=(3.3 * n_cond, 19.8),
                                        sharey='row')
    if n_cond == 1:
        axes_perf = axes_perf.reshape(4, 1)
    for ci, cond in enumerate(conditions):
        sub = df_desc[df_desc['condition'] == cond]
        # Row 0: RT
        draw_bars(axes_perf[0, ci],
                  sub[sub['group'] == g1]['mean_RT_ms'].dropna().values,
                  sub[sub['group'] == g2]['mean_RT_ms'].dropna().values, cond)
        # Row 1: Accuracy
        draw_bars(axes_perf[1, ci],
                  sub[sub['group'] == g1]['accuracy'].dropna().values,
                  sub[sub['group'] == g2]['accuracy'].dropna().values, cond)
        # Row 2: Miss rate
        draw_bars(axes_perf[2, ci],
                  sub[sub['group'] == g1]['miss_rate'].dropna().values,
                  sub[sub['group'] == g2]['miss_rate'].dropna().values, cond)
    # Row 3: Stroop effect (only 2 panels: RT effect + Accuracy effect)
    stroop_eff_items = [
        ('mean_RT_ms',  'RT effect'),
        ('accuracy',    'Accuracy effect'),
    ]
    sub_eff = df_desc[df_desc['condition'] == 'Stroop_RT_effect']
    for ei, (col, title_sub) in enumerate(stroop_eff_items):
        draw_bars(axes_perf[3, ei],
                  sub_eff[sub_eff['group'] == g1][col].dropna().values,
                  sub_eff[sub_eff['group'] == g2][col].dropna().values, title_sub)
    # Hide unused panels in row 3 (if n_cond > 2)
    for ei in range(len(stroop_eff_items), n_cond):
        axes_perf[3, ei].set_visible(False)
    axes_perf[0, 0].set_ylabel('Mean RT (ms)', fontsize=14)
    axes_perf[1, 0].set_ylabel('Accuracy (proportion)', fontsize=14)
    axes_perf[2, 0].set_ylabel('Miss rate (proportion)', fontsize=14)
    axes_perf[3, 0].set_ylabel('Stroop effect', fontsize=14)
    plt.suptitle("STROOP BEHAVIOURAL PERFORMANCE — {group_labels[g1]} vs {group_labels[g2]}",
                 fontsize=14, fontweight='bold', y=0.98)
    fig_perf.text(0.5, 0.94,
        "Dotted/Light Grey = {group_labels[g1]} · Solid/Darker Grey = {group_labels[g2]}\\n"
        "Bars = group mean · Error bars = SEM · Dots = individual participants\n"
        "Rows 1–3 = RT, Accuracy, Miss Rate by condition · Row 4 = Stroop effect (Incongruent − Congruent)",
        fontsize=11, style='italic', ha='center', va='top', color='#333333')
    plt.tight_layout(rect=[0.05, 0.08, 0.98, 0.90], h_pad=2.5)
    fig_perf.savefig(str(stats_folder / 'stroop_behav_performance_bars.png'),
                     dpi=300, bbox_inches='tight', pad_inches=0.15)
    plt.close(fig_perf)
    print(f"  Saved: stroop_behav_performance_bars.png")
    
    # ── Step 5: Behavioural variability comparison ──────────────────────────────
    print(f"\nSTEP 5: Behavioural variability (RT SD, RT CV)")
    
    var_rows = []
    fig_v, axes_v = plt.subplots(2, len(conditions),
                                  figsize=(3.3 * len(conditions), 9.9), sharey='row')
    if len(conditions) == 1:
        axes_v = axes_v.reshape(2, 1)
    
    for ci, cond in enumerate(conditions):
        sub = df_desc[df_desc['condition'] == cond]
        for mi, (meas_label, col) in enumerate([('RT SD (ms)', 'sd_RT_ms'),
                                                 ('RT CV (%)',  'cv_RT_pct')]):
            g1_v  = sub[sub['group'] == g1 ][col].dropna().values
            g2_v = sub[sub['group'] == g2][col].dropna().values
            # ── Variability metrics (identical formula to 09_EEG_variation_stroop) ──
            sd_ctrl  = np.std(g1_v,  ddof=1) if len(g1_v)  > 1 else np.nan
            sd_creat = np.std(g2_v, ddof=1) if len(g2_v) > 1 else np.nan
            var_ratio = (sd_ctrl ** 2) / (sd_creat ** 2) \
                        if (not np.isnan(sd_ctrl) and not np.isnan(sd_creat)
                            and sd_creat > 0) else np.nan
            try:
                lev_s, lev_p = stats.levene(g1_v, g2_v, center='median') \
                               if len(g1_v) > 1 and len(g2_v) > 1 \
                               else (np.nan, np.nan)
            except Exception:
                lev_s, lev_p = np.nan, np.nan
            var_rows.append({
                'condition':    cond, 'measure':       meas_label,
                f'sd_{g1}':   round(float(sd_ctrl),  4) if not np.isnan(sd_ctrl)  else np.nan,
                f'sd_{g2}':  round(float(sd_creat), 4) if not np.isnan(sd_creat) else np.nan,
                'variance_ratio': round(float(var_ratio), 4) if not np.isnan(var_ratio) else np.nan,
                'levene_stat':  round(float(lev_s), 4) if not np.isnan(lev_s) else np.nan,
                'levene_p':     round(float(lev_p), 4) if not np.isnan(lev_p) else np.nan,
                'levene_sig':   sig_stars(lev_p),
            })
            print(f"  {meas_label} — {cond}")
            print(f"    Ctrl SD={sd_ctrl:.3f}  Creat SD={sd_creat:.3f}  "
                  f"VarRatio={var_ratio:.2f}  Levene p={lev_p:.4f}  {sig_stars(lev_p)}")
    
            ax = axes_v[mi, ci]
            for gi, (grp, vals) in enumerate([(g1,  g1_v),
                                               (g2, g2_v)]):
                m   = np.mean(vals) if len(vals) > 0 else np.nan
                sem = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0
                sty = bar_styles[grp]
                if not np.isnan(m):
                    ax.bar(gi, m, width=0.35,
                           facecolor=sty['facecolor'], edgecolor=sty['edgecolor'],
                           linewidth=sty['linewidth'], linestyle=sty['linestyle'],
                           label=f"{group_labels[grp]} (N={len(vals)})")
                    ax.errorbar(gi, m, yerr=sem, fmt='none',
                                color='black', capsize=5, linewidth=1.5)
                jitter = np.random.default_rng(42).uniform(-0.07, 0.07, len(vals))
                ax.scatter(gi + jitter, vals, color='#333333', s=40, zorder=5, alpha=0.85)
            ax.axhline(0, color='#cccccc', lw=0.8)
            ax.set_xticks([0, 1])
            ax.set_xticklabels([group_labels[g] for g in groups], fontsize=10)
            ax.set_xlim([-0.5, 1.5])
            ax.set_title(f"{cond} — {meas_label}", fontsize=11, fontweight='bold', pad=50)
            ax.set_ylabel(meas_label, fontsize=14)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            # Adjust y-limits: 10% visual margin at top, 5% at bottom (standard style)
            ylim = ax.get_ylim()
            span = ylim[1] - ylim[0]
            ax.set_ylim(ylim[0] - 0.05 * span, ylim[1] + 0.10 * span)
            
            # Stats text block
            levene_stars = f" ({sig_stars(lev_p)})" if not np.isnan(lev_p) else ""
            p_str = f"{lev_p:.3f}" if not np.isnan(lev_p) else "n/a"
            vr_str = f"{var_ratio:.2f}" if not np.isnan(var_ratio) else "n/a"
            
            stats_text = (
                f"SD: Ctrl={sd_ctrl:.3f}, Creat={sd_creat:.3f}\n"
                f"Var Ratio (Ctrl/Creat) = {vr_str}\n"
                f"Levene p = {p_str} {sig_stars(lev_p)}"
            )
            ax.text(0.5, 1.02, stats_text, transform=ax.transAxes,
                    ha='center', va='bottom', fontsize=11, color='#333333',
                    fontweight='normal',
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none',
                              boxstyle='round,pad=0.2'))
    
    plt.suptitle(
        "STROOP BEHAVIOURAL RT VARIABILITY — {group_labels[g1]} vs {group_labels[g2]}",
        fontsize=13, fontweight='bold', y=0.98)
    fig_v.text(
        0.5, 0.94,
        "Dotted/Light Grey = {group_labels[g1]} · Solid/Darker Grey = {group_labels[g2]}\\n"
        "Bars = group mean · Error bars = SEM · Dots = individual participants\n"
        "SD = Standard Deviation · CV = Coefficient of Variation\n"
        "VR = Variance Ratio (Ctrl SD²/Creat SD²) · "
        "Levene's test (Brown-Forsythe) centered on median",
        fontsize=11, style='italic', ha='center', va='top', color='#333333'
    )
    plt.tight_layout(rect=[0.01, 0.08, 0.99, 0.89], h_pad=2.5)
    fig_v.savefig(str(stats_folder / 'stroop_behav_variability_bars.png'),
                  dpi=300, bbox_inches='tight', pad_inches=0.15)
    plt.close(fig_v)
    print(f"  Saved: stroop_behav_variability_bars.png")
    
    if var_rows:
        pd.DataFrame(var_rows).to_csv(
            str(stats_folder / 'stroop_variability_behavioural.csv'), index=False)
        print(f"  Saved: stroop_variability_behavioural.csv")
    
    # ── Step 6: Behavioural–ERP correlations ───────────────────────────────────
    print(f"\nSTEP 6: Behavioural-ERP correlations (Spearman rho)")
    
    erp_file = group_folder / 'group_stroop_individual_components.csv'
    if not erp_file.exists():
        print(f"  WARNING: {erp_file.name} not found.")
        print(f"           Run 07_group_erp_stroop.py first then re-run this script.")
    else:
        erp_df = pd.read_csv(erp_file)
        print(f"  ERP data loaded: {len(erp_df)} rows")
        print(f"  Components found in file: {sorted(erp_df['component'].unique())}")
        print(f"  Conditions found in file: {sorted(erp_df['condition'].unique())}")
        print(f"  PIDs found in file      : {sorted(erp_df['participant_id'].unique())}")
    
        # ── Normalise legacy component names to current names ─────────────────────
        # Old script used 'N2' and 'P3b' — new scripts use 'N200' and 'P300'
        # Map old names to new so correlations work regardless of which
        # version of 07_group_erp_stroop.py generated the CSV
        name_map = {
            'N2':  'N200',
            'P3b': 'P300',
            'N1':  'N1',
            'CSW': 'CSW',
        }
        erp_df['component'] = erp_df['component'].map(
            lambda x: name_map.get(x, x)
        )
        print(f"  Components after normalisation: {sorted(erp_df['component'].unique())}")
        print(f"  Participants in ERP file: "
              f"{sorted(erp_df['participant_id'].unique())}")
    
        # Build per-participant behavioural summary
        behav_sum = {}
        for pid in sorted(behav['participant_id'].unique()):
            sub     = behav[behav['participant_id']==pid]
            sub_c   = behav_corr[behav_corr['participant_id']==pid]
            rt_i    = sub_c[sub_c[cong_col]=='Incongruent'][rt_col].mean() \
                      if cong_col else np.nan
            rt_c2   = sub_c[sub_c[cong_col]=='Congruent'][rt_col].mean() \
                      if cong_col else np.nan
            ac_i    = sub[sub[cong_col]=='Incongruent'][acc_col].mean() \
                      if cong_col else np.nan
            ac_c2   = sub[sub[cong_col]=='Congruent'][acc_col].mean() \
                      if cong_col else np.nan
            ac_o    = sub[acc_col].mean() if acc_col else np.nan
            stroop_eff = rt_i-rt_c2 if not (np.isnan(rt_i) or np.isnan(rt_c2)) else np.nan
            behav_sum[pid] = {
                'group':              group_map.get(pid,'?'),
                'stroop_RT_effect':   stroop_eff,
                'RT_incongruent':     rt_i,
                'RT_congruent':       rt_c2,
                'accuracy_overall':   ac_o,
                'accuracy_congruent': ac_c2,
                'accuracy_incongruent': ac_i,
            }
    
        print(f"\n  Behavioural summary built for: {list(behav_sum.keys())}")
    
        # ── Pre-compute per-participant ERP absolute deviation from grand mean ──────
        # Formula identical to 09_EEG_variation_stroop.py: SD = np.std(vals, ddof=1)
        # Grand mean = mean across all 8 participants for that component/condition/measure
        # Per-participant abs_dev = |val_i - grand_mean|  (their contribution to SD)
        erp_abs_dev = {}   # keyed (pid, comp, cond, measure) → float
        for (comp, cond_erp, measure) in [
            ('N200', 'incongruent/correct', 'mean_amp_uv'),
            ('N200', 'incongruent/correct', 'peak_lat_ms'),
            ('N200', 'congruent/correct',   'mean_amp_uv'),
            ('P300', 'incongruent/correct', 'peak_lat_ms'),
            ('P300', 'incongruent/correct', 'mean_amp_uv'),
            ('P300', 'congruent/correct',   'mean_amp_uv'),
            ('P300', 'congruent/correct',   'peak_lat_ms'),
        ]:
            sub_erp = erp_df[(erp_df['component'] == comp) &
                             (erp_df['condition'] == cond_erp)][
                                 ['participant_id', measure]].dropna()
            if len(sub_erp) == 0:
                continue
            grand_mean = sub_erp[measure].mean()
            for _, row in sub_erp.iterrows():
                erp_abs_dev[(row['participant_id'], comp, cond_erp, measure)] = \
                    abs(row[measure] - grand_mean)
        print(f"  ERP absolute deviation (|val - grand mean|) computed for "
              f"{len(erp_abs_dev)} participant-measure entries")
    
        # ── Per-participant behavioural RT variability for correlation ──────────────
        # Pull RT SD and RT CV from the descriptives table (computed in Step 2)
        bvar_sum = {}
        for pid in sorted(behav['participant_id'].unique()):
            def _get_desc(cnd, col):
                row = df_desc[(df_desc['participant_id'] == pid) &
                              (df_desc['condition'] == cnd)]
                if len(row) == 0:
                    return np.nan
                v = row[col].values[0]
                return v if not np.isnan(v) else np.nan
            bvar_sum[pid] = {
                'group':       group_map.get(pid, '?'),
                'rt_sd_inc':   _get_desc('Incongruent', 'sd_RT_ms'),
                'rt_cv_inc':   _get_desc('Incongruent', 'cv_RT_pct'),
                'rt_sd_cong':  _get_desc('Congruent',   'sd_RT_ms'),
                'rt_cv_cong':  _get_desc('Congruent',   'cv_RT_pct'),
                'rt_sd_all':   _get_desc('Overall',     'sd_RT_ms'),
                'rt_cv_all':   _get_desc('Overall',     'cv_RT_pct'),
            }
    
        # ── 6a: Standard pairs (behavioural performance vs ERP mean amplitude/latency)
        corr_pairs = [
            ('stroop_RT_effect', 'Stroop RT effect (ms)',
             'N200', 'incongruent/correct', 'mean_amp_uv', 'N200 amplitude (µV)'),
            ('stroop_RT_effect', 'Stroop RT effect (ms)',
             'N200', 'incongruent/correct', 'peak_lat_ms', 'N200 latency (ms)'),
            ('stroop_RT_effect', 'Stroop RT effect (ms)',
             'P300', 'incongruent/correct', 'peak_lat_ms', 'P300 latency (ms)'),
            ('RT_congruent',     'Congruent RT (ms)',
             'P300', 'congruent/correct',   'mean_amp_uv', 'P300 amplitude (µV)'),
            ('accuracy_congruent', 'Congruent accuracy',
             'P300', 'congruent/correct',   'mean_amp_uv', 'P300 amplitude (µV)'),
            ('accuracy_overall', 'Overall accuracy',
             'N200', 'incongruent/correct', 'mean_amp_uv', 'N200 amplitude (µV)'),
            ('accuracy_overall', 'Overall accuracy',
             'P300', 'incongruent/correct', 'peak_lat_ms', 'P300 latency (ms)'),
        ]
    
        corr_results = []
    
        def _compute_corr(bkey, blabel, comp, cond, emeas, elabel,
                          bsource, erp_source, var_type='mean'):
            """Compute one Spearman correlation. Returns (result_dict, arrays) or None."""
            b_vals, e_vals, pids_used, grps_used = [], [], [], []
            for pid, bdata in bsource.items():
                b_val = bdata.get(bkey, np.nan)
                if np.isnan(b_val): continue
                e_val = erp_source.get(pid, np.nan)
                if np.isnan(e_val): continue
                b_vals.append(b_val); e_vals.append(e_val)
                pids_used.append(pid); grps_used.append(bdata['group'])
            suffix = 'AbsDev' if var_type == 'abs_dev' else ''
            print(f"\n  {blabel} vs {elabel}{suffix}: N={len(b_vals)} pairs")
            if len(b_vals) < 4:
                print(f"    SKIP — fewer than 4 complete pairs")
                return None
            b_arr = np.array(b_vals); e_arr = np.array(e_vals)
            rho, p_val = stats.spearmanr(b_arr, e_arr)
            print(f"    ρ={rho:.3f}  p={p_val:.4f}  {sig_stars(p_val)}")
            res = {
                'behavioural_measure': blabel,
                'erp_component': comp, 'erp_condition': cond,
                'erp_measure': emeas, 'erp_label': elabel + suffix,
                'erp_variable_type': var_type,
                'n': len(b_vals), 'spearman_rho': round(rho, 3),
                'p_value': round(p_val, 4), 'significance': sig_stars(p_val),
            }
            return res, b_arr, e_arr, pids_used, grps_used, rho, p_val
    
        def _plot_scatter(ax, b_arr, e_arr, pids_u, grps_u, blabel, elabel,
                          rho, p_val, var_type='mean'):
            """Draw one scatter sub-panel (grid-friendly, smaller fonts)."""
            for bv, ev, pid, grp in zip(b_arr, e_arr, pids_u, grps_u):
                ax.scatter(bv, ev, color=group_colours[grp], s=50, zorder=5, alpha=0.9)
                ax.annotate(pid, (bv, ev), textcoords='offset points',
                            xytext=(4, 3), fontsize=8, color=group_colours[grp])
            if len(b_arr) >= 2:
                m_fit, b_fit_ = np.polyfit(b_arr, e_arr, 1)
                x_l = np.linspace(b_arr.min(), b_arr.max(), 100)
                ax.plot(x_l, m_fit * x_l + b_fit_,
                        color='#555555', lw=1.2, ls=':', alpha=0.7)
            ax.set_xlabel(blabel, fontsize=12)
            ylabel_full = f"{elabel} |dev. grand mean|" \
                          if var_type == 'abs_dev' else elabel
            ax.set_ylabel(ylabel_full, fontsize=12)
            ax.set_title(f"{blabel}\nvs {ylabel_full}", fontsize=11, fontweight='bold')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.text(0.05, 0.95,
                    f"ρ={rho:.3f}\np={p_val:.4f} {sig_stars(p_val)}\nN={len(b_arr)}",
                    transform=ax.transAxes, va='top', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.85, edgecolor='#cccccc',
                              boxstyle='round,pad=0.2'))
            for grp in groups:
                ax.scatter([], [], color=group_colours[grp], s=30, label=group_labels[grp])
            ax.legend(fontsize=10, loc='lower right')
    
        # ── 6a: Performance vs ERP mean amplitude / latency ───────────────────
        print(f"\n  [6a] Performance vs ERP mean amplitude / latency")
        corr_6a_data = []
        for (bkey, blabel, comp, cond, emeas, elabel) in corr_pairs:
            erp_lkp = {}
            for pid in behav_sum:
                erow = erp_df[(erp_df['participant_id'] == pid) &
                              (erp_df['component'] == comp) &
                              (erp_df['condition'] == cond)]
                if not erow.empty:
                    v = erow[emeas].values[0]
                    erp_lkp[pid] = v if not np.isnan(v) else np.nan
                else:
                    erp_lkp[pid] = np.nan
                if erp_lkp[pid] != erp_lkp[pid]:  # isnan check via inequality
                    print(f"    No ERP data for {pid} {comp} {cond}")
            out = _compute_corr(bkey, blabel, comp, cond, emeas, elabel,
                                behav_sum, erp_lkp, var_type='mean')
            if out is not None:
                res, b_arr, e_arr, pids_u, grps_u, rho, p_val = out
                corr_results.append(res)
                corr_6a_data.append((blabel, elabel, b_arr, e_arr, pids_u, grps_u,
                                     rho, p_val, 'mean'))
    
        # Grid figure for 6a — 3×3 layout (Congruent RT vs P300 amp on row 3)
        if corr_6a_data:
            # Reorder: move 'Congruent RT (ms)' entry to the end so it lands on row 3
            main_data = [d for d in corr_6a_data if d[0] != 'Congruent RT (ms)']
            cong_data = [d for d in corr_6a_data if d[0] == 'Congruent RT (ms)']
            corr_6a_ordered = main_data + cong_data
            n_6a = len(corr_6a_ordered)
            nc = 3; nr = (n_6a + nc - 1) // nc
            fig_6a, axes_6a = plt.subplots(nr, nc, figsize=(4.95 * nc, 4.5 * nr))
            axes_6a = np.atleast_2d(axes_6a)
            for idx, (bl, el, ba, ea, pu, gu, rho, pv, vt) in enumerate(corr_6a_ordered):
                ri, ci = divmod(idx, nc)
                _plot_scatter(axes_6a[ri, ci], ba, ea, pu, gu, bl, el, rho, pv, vt)
            for idx in range(n_6a, nr * nc):
                ri, ci = divmod(idx, nc)
                axes_6a[ri, ci].set_visible(False)
            plt.suptitle(
                "STROOP — Behavioural Performance vs ERP (Spearman)",
                fontsize=14, fontweight='bold', y=0.98)
            fig_6a.text(0.5, 0.94,
                "N=8 — interpret with caution · "
                "Dotted line = linear trend · PID labels on dots",
                fontsize=11, style='italic', ha='center', va='top', color='#333333')
            plt.tight_layout(rect=[0.05, 0.08, 0.98, 0.92])
            fig_6a.savefig(str(stats_folder / 'stroop_corr_performance.png'),
                           dpi=300, bbox_inches='tight', pad_inches=0.15)
            plt.close(fig_6a)
            print(f"  Saved: stroop_corr_performance.png")
    
        # ── 6b: Behavioural RT variability vs ERP absolute deviation ───────────
        # (SD only — CV% removed per request)
        print(f"\n  [6b] Behavioural RT SD vs ERP absolute deviation")
        var_corr_pairs = [
            ('rt_sd_inc',  'Incongruent RT SD (ms)',
             'N200', 'incongruent/correct', 'mean_amp_uv', 'N200 amplitude'),
            ('rt_sd_inc',  'Incongruent RT SD (ms)',
             'P300', 'incongruent/correct', 'mean_amp_uv', 'P300 amplitude'),
            ('rt_sd_inc',  'Incongruent RT SD (ms)',
             'N200', 'incongruent/correct', 'peak_lat_ms', 'N200 latency'),
            ('rt_sd_cong', 'Congruent RT SD (ms)',
             'P300', 'congruent/correct',   'mean_amp_uv', 'P300 amplitude'),
            ('rt_sd_all',  'Overall RT SD (ms)',
             'N200', 'incongruent/correct', 'mean_amp_uv', 'N200 amplitude'),
        ]
        corr_6b_data = []
        for (bkey, blabel, comp, cond, emeas, elabel) in var_corr_pairs:
            erp_lkp = {}
            for pid in bvar_sum:
                key = (pid, comp, cond, emeas)
                erp_lkp[pid] = erp_abs_dev.get(key, np.nan)
            out = _compute_corr(bkey, blabel, comp, cond, emeas, elabel,
                                bvar_sum, erp_lkp, var_type='abs_dev')
            if out is not None:
                res, b_arr, e_arr, pids_u, grps_u, rho, p_val = out
                corr_results.append(res)
                corr_6b_data.append((blabel, elabel, b_arr, e_arr, pids_u, grps_u,
                                     rho, p_val, 'abs_dev'))
    
        # Grid figure for 6b — 3 columns
        if corr_6b_data:
            n_6b = len(corr_6b_data)
            nc = 3; nr = (n_6b + nc - 1) // nc
            fig_6b, axes_6b = plt.subplots(nr, nc, figsize=(4.95 * nc, 4.5 * nr))
            axes_6b = np.atleast_2d(axes_6b)
            for idx, (bl, el, ba, ea, pu, gu, rho, pv, vt) in enumerate(corr_6b_data):
                ri, ci = divmod(idx, nc)
                _plot_scatter(axes_6b[ri, ci], ba, ea, pu, gu, bl, el, rho, pv, vt)
            for idx in range(n_6b, nr * nc):
                ri, ci = divmod(idx, nc)
                axes_6b[ri, ci].set_visible(False)
            plt.suptitle(
                "STROOP — Behavioural RT SD vs ERP Absolute Deviation (Spearman)",
                fontsize=14, fontweight='bold', y=0.98)
            fig_6b.text(0.5, 0.94,
                "ERP variable = |ERP_i − grand mean| (np.std ddof=1, same as 09_EEG_variation)\n"
                "N=8 — interpret with caution · Dotted line = linear trend",
                fontsize=11, style='italic', ha='center', va='top', color='#333333')
            plt.tight_layout(rect=[0.05, 0.08, 0.98, 0.92])
            fig_6b.savefig(str(stats_folder / 'stroop_corr_variability.png'),
                           dpi=300, bbox_inches='tight', pad_inches=0.15)
            plt.close(fig_6b)
            print(f"  Saved: stroop_corr_variability.png")
    
        if corr_results:
            pd.DataFrame(corr_results).to_csv(
                str(stats_folder / 'stroop_behavioural_erp_correlations.csv'), index=False)
            print(f"\n  Saved: stroop_behavioural_erp_correlations.csv")
        else:
            print(f"\n  No correlations computed — check ERP file exists and has matching PIDs")
    
    # ── Step 7: ERP variability — separate figure per component ───────────────────
    # (Removed - analyzed in 09_EEG_variation_stroop.py)
    
    # ── Summary ───────────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"COMPLETE — Behavioural analysis STROOP")
    print(f"Finished : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*65}")
    print(f"\nOutputs in output/stats/stroop_beh_output/:")
    print(f"  stroop_behavioural_descriptives.csv")
    print(f"  stroop_behavioural_statistics.csv")
    print(f"  stroop_variability_behavioural.csv")
    print(f"  stroop_behavioural_erp_correlations.csv")
    print(f"  stroop_behav_performance_bars.png   (RT + Accuracy + Miss rate + Stroop effect)")
    print(f"  stroop_behav_variability_bars.png   (RT SD + CV)")
    print(f"  stroop_corr_performance.png         (6a scatter grid — 3×3)")
    print(f"  stroop_corr_variability.png         (6b scatter grid — SD only, 3-col)")
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['nback', 'stroop'], required=True)
    args = parser.parse_args()
    if args.task == 'nback':
        run_nback()
    elif args.task == 'stroop':
        run_stroop()

if __name__ == "__main__":
    main()
