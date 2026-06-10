# scripts/09_EEG_variation_stroop.py
# ─────────────────────────────────────────────────────────────────────────────
# BETWEEN-SUBJECT VARIABILITY ANALYSIS — STROOP TASK
#
# This script loads the Stroop group ERP component data and compares
# between-subject variability (variance/standard deviation) between
# the Control and Creatine groups.
#
# Outputs:
#   CSV files in output/stats/:
#     stroop_variability_descriptives.csv   - Descriptive statistics
#     stroop_variability_statistics.csv     - Inferential statistics (Levene's test)
#
#   Figures in output/group/:
#     group_stroop_variability_congruent.png
#     group_stroop_variability_incongruent.png
#     group_stroop_variability_no_response.png
#
# Usage:
#   python3 scripts/09_EEG_variation_stroop.py
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests
from pathlib import Path
from datetime import datetime
import sys

# ── Paths ─────────────────────────────────────────────────────────────────────
group_folder = Path('output/group')
stats_folder = Path('output/stats')
group_folder.mkdir(parents=True, exist_ok=True)
stats_folder.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print(f"STROOP EEG VARIABILITY ANALYSIS")
print(f"Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

# ── Load Data ─────────────────────────────────────────────────────────────────
csv_path = group_folder / 'group_stroop_individual_components.csv'
if not csv_path.exists():
    print(f"ERROR: {csv_path} not found. Run 07_group_erp_stroop.py first.")
    sys.exit(1)

df = pd.read_csv(csv_path)
print(f"Loaded database: {csv_path.name}")
print(f"Participants: {sorted(df['participant_id'].unique())} (N={len(df['participant_id'].unique())})")

# ── Analysis Specifications ───────────────────────────────────────────────────
# Note: "N300 Peak Amplitude" in user request is mapped to P300 Peak Amplitude
# because N300 is not a component extracted in the pipeline.
metrics = [
    {
        'component': 'N200',
        'measure': 'peak_amp_uv',
        'name': 'N200 Peak Amplitude',
        'ylabel': 'Amplitude (µV)',
        'invert_y': True
    },
    {
        'component': 'N200',
        'measure': 'mean_amp_uv',
        'name': 'N200 Mean Amplitude',
        'ylabel': 'Amplitude (µV)',
        'invert_y': True
    },
    {
        'component': 'P300',
        'measure': 'peak_lat_ms',
        'name': 'P300 Peak Latency',
        'ylabel': 'Latency (ms)',
        'invert_y': False
    },
    {
        'component': 'P300',
        'measure': 'peak_amp_uv',
        'name': 'P300 Peak Amplitude',
        'ylabel': 'Amplitude (µV)',
        'invert_y': False
    }
]

conditions = ['congruent/correct', 'incongruent/correct', 'no_response']
cond_labels = {
    'congruent/correct': 'Congruent',
    'incongruent/correct': 'Incongruent',
    'no_response': 'No response'
}

groups = ['control', 'creatine']
group_labels = {'control': 'Control', 'creatine': 'Creatine'}

bar_styles = {
    'control':  {'facecolor': '#DDDDDD', 'edgecolor': '#333333',
                 'linewidth': 2.5, 'linestyle': ':'},
    'creatine': {'facecolor': '#AAAAAA', 'edgecolor': '#333333',
                 'linewidth': 2.5, 'linestyle': '-'},
}

def sig(p):
    if np.isnan(p): return 'n/a'
    if p < 0.001:   return '***'
    if p < 0.01:    return '**'
    if p < 0.05:    return '*'
    if p < 0.10:    return '†'
    return 'ns'

# ── Step 1: Calculate Statistics ──────────────────────────────────────────────
results = []
descriptive_rows = []

for cond in conditions:
    for metric in metrics:
        comp = metric['component']
        mcol = metric['measure']
        mname = metric['name']
        
        # Filter data
        ctrl_vals = df[(df['component'] == comp) & (df['condition'] == cond) & (df['group'] == 'control')][mcol].dropna().values
        creat_vals = df[(df['component'] == comp) & (df['condition'] == cond) & (df['group'] == 'creatine')][mcol].dropna().values
        
        # Descriptive statistics
        for grp, vals in [('control', ctrl_vals), ('creatine', creat_vals)]:
            n = len(vals)
            if n > 0:
                mean_val = np.mean(vals)
                sd_val = np.std(vals, ddof=1) if n > 1 else 0.0
                var_val = np.var(vals, ddof=1) if n > 1 else 0.0
                sem_val = sd_val / np.sqrt(n) if n > 1 else 0.0
                min_val = np.min(vals)
                max_val = np.max(vals)
            else:
                mean_val, sd_val, var_val, sem_val, min_val, max_val = [np.nan] * 6
                
            descriptive_rows.append({
                'condition': cond_labels[cond],
                'component': comp,
                'measure': mname,
                'group': group_labels[grp],
                'n': n,
                'mean': round(mean_val, 4) if not np.isnan(mean_val) else np.nan,
                'sd': round(sd_val, 4) if not np.isnan(sd_val) else np.nan,
                'variance': round(var_val, 4) if not np.isnan(var_val) else np.nan,
                'sem': round(sem_val, 4) if not np.isnan(sem_val) else np.nan,
                'min': round(min_val, 4) if not np.isnan(min_val) else np.nan,
                'max': round(max_val, 4) if not np.isnan(max_val) else np.nan
            })
            
        # Inferential statistics (Levene's test centered on median / Brown-Forsythe)
        if len(ctrl_vals) >= 2 and len(creat_vals) >= 2:
            levene_stat, levene_p = stats.levene(ctrl_vals, creat_vals, center='median')
            sd_ctrl = np.std(ctrl_vals, ddof=1)
            sd_creat = np.std(creat_vals, ddof=1)
            var_ratio = (sd_ctrl ** 2) / (sd_creat ** 2) if sd_creat > 0 else np.nan
        else:
            levene_stat, levene_p = np.nan, np.nan
            sd_ctrl = np.std(ctrl_vals, ddof=1) if len(ctrl_vals) > 1 else np.nan
            sd_creat = np.std(creat_vals, ddof=1) if len(creat_vals) > 1 else np.nan
            var_ratio = np.nan
            
        results.append({
            'condition': cond_labels[cond],
            'component': comp,
            'measure': mname,
            'sd_control': round(sd_ctrl, 4) if not np.isnan(sd_ctrl) else np.nan,
            'sd_creatine': round(sd_creat, 4) if not np.isnan(sd_creat) else np.nan,
            'variance_ratio': round(var_ratio, 4) if not np.isnan(var_ratio) else np.nan,
            'levene_stat': round(levene_stat, 4) if not np.isnan(levene_stat) else np.nan,
            'levene_p': levene_p
        })

# Apply FDR correction on Levene's p-values (across all 12 tests)
p_values = [r['levene_p'] for r in results]
valid_p_idx = [idx for idx, p in enumerate(p_values) if not np.isnan(p)]
if len(valid_p_idx) > 0:
    _, adj_p_values, _, _ = multipletests([p_values[idx] for idx in valid_p_idx], method='fdr_bh')
    adj_p_dict = dict(zip(valid_p_idx, adj_p_values))
    for idx, r in enumerate(results):
        if idx in adj_p_dict:
            r['levene_p_fdr'] = round(float(adj_p_dict[idx]), 4)
        else:
            r['levene_p_fdr'] = np.nan
else:
    for r in results:
        r['levene_p_fdr'] = np.nan

# Format uncorrected levene_p to 4 decimals for table/CSV
for r in results:
    r['levene_p'] = round(r['levene_p'], 4) if not np.isnan(r['levene_p']) else np.nan
    r['multiple_comparisons_correction'] = 'FDR (Benjamini-Hochberg) applied across all 12 variability tests'

# Save DataFrames
df_descriptives = pd.DataFrame(descriptive_rows)
df_statistics = pd.DataFrame(results)

descriptives_path = stats_folder / 'stroop_EEG_variability_descriptives.csv'
statistics_path = stats_folder / 'stroop_EEG_variability_statistics.csv'

# Save with Excel-compatible delimiter hint sep=, at the top
with open(descriptives_path, 'w', encoding='utf-8-sig') as f:
    f.write("sep=,\n" + df_descriptives.to_csv(index=False))
with open(statistics_path, 'w', encoding='utf-8-sig') as f:
    f.write("sep=,\n" + df_statistics.to_csv(index=False))

# ── Step 2: Print Terminal Summary Table ──────────────────────────────────────
print(f"\n{'='*95}")
print(f"LEVENE'S TEST RESULTS (BROWN-FORSYTHE) — BETWEEN-SUBJECT VARIABILITY")
print(f"{'='*95}")
print(f"\nSignificance: *** p<.001  ** p<.01  * p<.05  † p<.10  ns")
print(f"\n{'Condition':15s} {'Metric':25s} {'SD Ctrl':>8} {'SD Creat':>8} {'Var Ratio':>9} {'Lev Stat':>8} {'Lev p':>7} {'Lev p_FDR':>9}")
print("-" * 105)

def fmt_val(v, decimals=3):
    return f"{v:.{decimals}f}" if not (isinstance(v, float) and np.isnan(v)) else 'n/a'

for r in results:
    p_str = fmt_val(r['levene_p'], 4)
    p_fdr_str = fmt_val(r['levene_p_fdr'], 4)
    sig_str = f" ({sig(r['levene_p'])})" if not np.isnan(r['levene_p']) else ""
    print(f"  {r['condition']:15s} {r['measure']:25s} "
          f"{fmt_val(r['sd_control']):>8} {fmt_val(r['sd_creatine']):>8} "
          f"{fmt_val(r['variance_ratio'], 2):>9} {fmt_val(r['levene_stat']):>8} "
          f"{p_str:>7}{sig_str:5s} {p_fdr_str:>9}")

print(f"\nSaved descriptives: {descriptives_path.name}")
print(f"Saved statistics  : {statistics_path.name}")
print(f"{'='*95}")

# ── Step 3: Generate Figures ──────────────────────────────────────────────────
print(f"\nSTEP 3: Generating variability figures...")

for cond_code in conditions:
    cond_lbl = cond_labels[cond_code]
    fig, axes = plt.subplots(1, 4, figsize=(12.6, 7.0), sharey=False)
    
    for ax, metric in zip(axes, metrics):
        comp = metric['component']
        mcol = metric['measure']
        mname = metric['name']
        ylabel = metric['ylabel']
        invert_y = metric['invert_y']
        
        # Get data
        ctrl_vals = df[(df['component'] == comp) & (df['condition'] == cond_code) & (df['group'] == 'control')][mcol].dropna().values
        creat_vals = df[(df['component'] == comp) & (df['condition'] == cond_code) & (df['group'] == 'creatine')][mcol].dropna().values
        
        sd_ctrl = np.std(ctrl_vals, ddof=1) if len(ctrl_vals) > 1 else 0.0
        sd_creat = np.std(creat_vals, ddof=1) if len(creat_vals) > 1 else 0.0
        var_ratio = (sd_ctrl ** 2) / (sd_creat ** 2) if sd_creat > 0 else np.nan
        
        # Look up pre-computed statistics (including FDR-corrected p-value)
        match = [r for r in results if r['condition'] == cond_lbl and r['measure'] == mname]
        if match:
            res_dict = match[0]
            p_levene = res_dict['levene_p']
            p_fdr = res_dict['levene_p_fdr']
        else:
            p_levene = np.nan
            p_fdr = np.nan
            
        # Plot bars
        for gi, (grp, vals) in enumerate([('control', ctrl_vals), ('creatine', creat_vals)]):
            m = np.mean(vals) if len(vals) > 0 else np.nan
            sem = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
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
        ax.set_xticklabels([group_labels[g] for g in ['control', 'creatine']], fontsize=14)
        ax.set_xlim([-0.5, 1.5])
        ax.set_title(mname, fontsize=13, fontweight='bold', pad=60)
        ax.set_ylabel(ylabel, fontsize=15)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # If y-axis is inverted (for N200)
        if invert_y:
            ax.invert_yaxis()
            
        # Adjust y-limits with standard 10% visual margin at the top (no large 55% expansion)
        ylim = ax.get_ylim()
        v_bottom, v_top = ylim[0], ylim[1]
        span = v_top - v_bottom
        new_top = v_top + 0.10 * span
        new_bottom = v_bottom - 0.05 * span
        ax.set_ylim(new_bottom, new_top)
        
        # Build stats text block
        levene_stars = f" ({sig(p_levene)})" if not np.isnan(p_levene) else ""
        fdr_stars = f" ({sig(p_fdr)})" if not np.isnan(p_fdr) else ""
        tp_str = f"{p_levene:.3f}" if not np.isnan(p_levene) else "n/a"
        fdr_str = f"{p_fdr:.3f}" if not np.isnan(p_fdr) else "n/a"
        vr_str = f"{var_ratio:.2f}" if not np.isnan(var_ratio) else "n/a"
        
        stats_text = (
            f"SD: Ctrl={sd_ctrl:.3f}, Creat={sd_creat:.3f}\n"
            f"Var Ratio (Ctrl/Creat) = {vr_str}\n"
            f"Levene p = {tp_str}{levene_stars}\n"
            f"p_FDR = {fdr_str}{fdr_stars}"
        )
        ax.text(0.5, 1.02, stats_text, transform=ax.transAxes,
                ha='center', va='bottom', fontsize=11.5, color='#333333',
                fontweight='normal', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))

    # Save figure
    fname = f"group_stroop_variability_{cond_code.split('/')[0]}.png"
    plt.suptitle(
        f"STROOP {cond_lbl.upper()} EEG VARIABILITY — Control vs Creatine",
        fontsize=15, fontweight='bold', y=0.99
    )
    fig.text(
        0.5, 0.95,
        f"Dotted/Light Grey = control · Solid/Darker Grey = creatine\n"
        f"Bars=group mean · Error bars=SEM · Dots=individual participants\n"
        f"Levene's test (Brown-Forsythe) centered on median",
        fontsize=9.5, style='italic', ha='center', va='top', color='#333333'
    )
    plt.tight_layout(rect=[0.05, 0.08, 0.98, 0.88])
    fig.savefig(str(group_folder / fname), dpi=300, bbox_inches='tight', pad_inches=0.15)
    plt.close(fig)
    print(f"  Saved figure: {fname}")

print(f"\nFINISHED : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)
