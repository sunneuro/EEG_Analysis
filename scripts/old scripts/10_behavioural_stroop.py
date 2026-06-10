# scripts/10_behavioural_stroop.py
# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 OF EEG PIPELINE: Behavioural analysis — STROOP
#
# Sections:
#   1. Load and inspect behavioural data (auto-detects column names)
#   2. Descriptive statistics per participant per condition
#   3. Group comparison statistics (Control vs Creatine)
#   4. Bar charts — RT, accuracy, Stroop effect, miss rate
#   5. Behavioural variability comparison (RT SD, RT CV — Levene/Brown-Forsythe)
#   6. Behavioural–ERP correlations (Spearman rho, N=8)
#      incl. ERP absolute variability pairs (|ERP_i − grand mean|, same SD
#      formula as 09_EEG_variation_stroop.py)
#
# Usage:
#   python scripts/10_behavioural_stroop.py
# ─────────────────────────────────────────────────────────────────────────────

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from pathlib import Path
from datetime import datetime

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
group_map = {
    'P01':'creatine','P02':'creatine','P03':'creatine','P04':'creatine',
    'P05':'control', 'P06':'control', 'P07':'control', 'P08':'control',
}
groups        = ['control','creatine']
group_colours = {'control':'#A32D2D','creatine':'#185FA5'}
group_labels  = {'control':'Control','creatine':'Creatine'}
bar_styles    = {
    'control':  {'facecolor':'#DDDDDD','edgecolor':'#333333',
                 'linewidth':2.5,'linestyle':':'},
    'creatine': {'facecolor':'#AAAAAA','edgecolor':'#333333',
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
        'n_control':     na,   'n_creatine':    nb,
        'mean_control':  round(float(np.mean(a)),3) if na>0 else np.nan,
        'mean_creatine': round(float(np.mean(b)),3) if nb>0 else np.nan,
        'sd_control':    round(float(np.std(a,ddof=1)),3) if na>1 else np.nan,
        'sd_creatine':   round(float(np.std(b,ddof=1)),3) if nb>1 else np.nan,
        'sem_control':   round(float(np.std(a,ddof=1)/np.sqrt(na)),3) if na>1 else np.nan,
        'sem_creatine':  round(float(np.std(b,ddof=1)/np.sqrt(nb)),3) if nb>1 else np.nan,
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

def draw_bars(ax, ctrl_vals, creat_vals, cond_title):
    """Draw two bars (control left, creatine right) with dots and SEM.
    Style matched to 09_EEG_variation_*.py."""
    for gi, (grp, vals) in enumerate([('control', ctrl_vals),
                                       ('creatine', creat_vals)]):
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
    r = run_tests(np.array(ctrl_vals), np.array(creat_vals))
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
        ctrl_v  = sub[sub['group']=='control' ][col].dropna().values
        creat_v = sub[sub['group']=='creatine'][col].dropna().values
        if len(ctrl_v)==0 and len(creat_v)==0: continue
        r = run_tests(ctrl_v, creat_v, f"Stroop {measure} — {cond}")
        r['condition'] = cond; r['measure'] = measure
        stat_rows.append(r)
        print(f"\n  {r['comparison']}")
        print(f"    Ctrl  : M={r['mean_control']:.3f}  SD={r['sd_control']:.3f}  N={r['n_control']}")
        print(f"    Creat : M={r['mean_creatine']:.3f}  SD={r['sd_creatine']:.3f}  N={r['n_creatine']}")
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
              sub[sub['group'] == 'control']['mean_RT_ms'].dropna().values,
              sub[sub['group'] == 'creatine']['mean_RT_ms'].dropna().values, cond)
    # Row 1: Accuracy
    draw_bars(axes_perf[1, ci],
              sub[sub['group'] == 'control']['accuracy'].dropna().values,
              sub[sub['group'] == 'creatine']['accuracy'].dropna().values, cond)
    # Row 2: Miss rate
    draw_bars(axes_perf[2, ci],
              sub[sub['group'] == 'control']['miss_rate'].dropna().values,
              sub[sub['group'] == 'creatine']['miss_rate'].dropna().values, cond)
# Row 3: Stroop effect (only 2 panels: RT effect + Accuracy effect)
stroop_eff_items = [
    ('mean_RT_ms',  'RT effect'),
    ('accuracy',    'Accuracy effect'),
]
sub_eff = df_desc[df_desc['condition'] == 'Stroop_RT_effect']
for ei, (col, title_sub) in enumerate(stroop_eff_items):
    draw_bars(axes_perf[3, ei],
              sub_eff[sub_eff['group'] == 'control'][col].dropna().values,
              sub_eff[sub_eff['group'] == 'creatine'][col].dropna().values, title_sub)
# Hide unused panels in row 3 (if n_cond > 2)
for ei in range(len(stroop_eff_items), n_cond):
    axes_perf[3, ei].set_visible(False)
axes_perf[0, 0].set_ylabel('Mean RT (ms)', fontsize=14)
axes_perf[1, 0].set_ylabel('Accuracy (proportion)', fontsize=14)
axes_perf[2, 0].set_ylabel('Miss rate (proportion)', fontsize=14)
axes_perf[3, 0].set_ylabel('Stroop effect', fontsize=14)
plt.suptitle("STROOP BEHAVIOURAL PERFORMANCE — Control vs Creatine",
             fontsize=14, fontweight='bold', y=0.98)
fig_perf.text(0.5, 0.94,
    "Dotted/Light Grey = Control · Solid/Darker Grey = Creatine\n"
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
        ctrl_v  = sub[sub['group'] == 'control' ][col].dropna().values
        creat_v = sub[sub['group'] == 'creatine'][col].dropna().values
        # ── Variability metrics (identical formula to 09_EEG_variation_stroop) ──
        sd_ctrl  = np.std(ctrl_v,  ddof=1) if len(ctrl_v)  > 1 else np.nan
        sd_creat = np.std(creat_v, ddof=1) if len(creat_v) > 1 else np.nan
        var_ratio = (sd_ctrl ** 2) / (sd_creat ** 2) \
                    if (not np.isnan(sd_ctrl) and not np.isnan(sd_creat)
                        and sd_creat > 0) else np.nan
        try:
            lev_s, lev_p = stats.levene(ctrl_v, creat_v, center='median') \
                           if len(ctrl_v) > 1 and len(creat_v) > 1 \
                           else (np.nan, np.nan)
        except Exception:
            lev_s, lev_p = np.nan, np.nan
        var_rows.append({
            'condition':    cond, 'measure':       meas_label,
            'sd_control':   round(float(sd_ctrl),  4) if not np.isnan(sd_ctrl)  else np.nan,
            'sd_creatine':  round(float(sd_creat), 4) if not np.isnan(sd_creat) else np.nan,
            'variance_ratio': round(float(var_ratio), 4) if not np.isnan(var_ratio) else np.nan,
            'levene_stat':  round(float(lev_s), 4) if not np.isnan(lev_s) else np.nan,
            'levene_p':     round(float(lev_p), 4) if not np.isnan(lev_p) else np.nan,
            'levene_sig':   sig_stars(lev_p),
        })
        print(f"  {meas_label} — {cond}")
        print(f"    Ctrl SD={sd_ctrl:.3f}  Creat SD={sd_creat:.3f}  "
              f"VarRatio={var_ratio:.2f}  Levene p={lev_p:.4f}  {sig_stars(lev_p)}")

        ax = axes_v[mi, ci]
        for gi, (grp, vals) in enumerate([('control',  ctrl_v),
                                           ('creatine', creat_v)]):
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
    "STROOP BEHAVIOURAL RT VARIABILITY — Control vs Creatine",
    fontsize=13, fontweight='bold', y=0.98)
fig_v.text(
    0.5, 0.94,
    "Dotted/Light Grey = Control · Solid/Darker Grey = Creatine\n"
    "Bars = group mean · Error bars = SEM · Dots = individual participants\n"
    "SD = Standard Deviation · CV = Coefficient of Variation · "
    "VR = Variance Ratio (Ctrl SD²/Creat SD²) · "
    "Levene's test (Brown-Forsythe) centered on median",
    fontsize=11, style='italic', ha='center', va='top', color='#333333'
)
plt.tight_layout(rect=[0.05, 0.08, 0.98, 0.89], h_pad=2.5)
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
