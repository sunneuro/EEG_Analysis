import sys
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
from pathlib import Path
from datetime import datetime

# ── Helpers ───────────────────────────────────────────────────────────────────
def cohens_d(a, b):
    na, nb = len(a), len(b)
    if na < 2 or nb < 2: return np.nan
    s = np.sqrt(((na-1)*np.std(a,ddof=1)**2+(nb-1)*np.std(b,ddof=1)**2)
                /(na+nb-2))
    return np.nan if s==0 else (np.mean(a)-np.mean(b))/s

def run_tests(a, b, label):
    na, nb = len(a), len(b)
    d = cohens_d(a, b)
    if na >= 2 and nb >= 2:
        t, tp = stats.ttest_ind(a, b, equal_var=False)
    else:
        t, tp = np.nan, np.nan
    try:
        u, up = stats.mannwhitneyu(a, b, alternative='two-sided') \
                if na>=1 and nb>=1 else (np.nan, np.nan)
    except ValueError:
        u, up = np.nan, np.nan

    def sig(p):
        if np.isnan(p): return 'n/a'
        if p < 0.001:   return '***'
        if p < 0.01:    return '**'
        if p < 0.05:    return '*'
        if p < 0.10:    return '†'
        return 'ns'

    return {
        'comparison':    label,
        f'n_{g1}':     na,
        f'n_{g2}':    nb,
        f'mean_{g1}':  round(float(np.mean(a)),4) if na>0 else np.nan,
        f'mean_{g2}': round(float(np.mean(b)),4) if nb>0 else np.nan,
        f'sd_{g1}':    round(float(np.std(a,ddof=1)),4) if na>1 else np.nan,
        f'sd_{g2}':   round(float(np.std(b,ddof=1)),4) if nb>1 else np.nan,
        f'sem_{g1}':   round(float(np.std(a,ddof=1)/np.sqrt(na)),4) if na>1 else np.nan,
        f'sem_{g2}':  round(float(np.std(b,ddof=1)/np.sqrt(nb)),4) if nb>1 else np.nan,
        't_stat':        round(float(t),3) if not np.isnan(t) else np.nan,
        't_p':           round(float(tp),4) if not np.isnan(tp) else np.nan,
        't_sig':         sig(tp),
        'U_stat':        round(float(u),3) if not np.isnan(u) else np.nan,
        'U_p':           round(float(up),4) if not np.isnan(up) else np.nan,
        'U_sig':         sig(up),
        'cohens_d':      round(float(d),3) if not np.isnan(d) else np.nan,
        'd_interp':      ('small' if abs(d)<0.5 else
                          'medium' if abs(d)<0.8 else
                          'large') if not np.isnan(d) else 'n/a',
    }

def print_primary(r, indent='  '):
    print(f"\n{indent}{r['comparison']}")
    print(f"{indent}  {group_labels[g1]:8s} : M={r[f'mean_{g1}']:.3f}  "
          f"SD={r[f'sd_{g1}']:.3f}  N={r[f'n_{g1}']}")
    print(f"{indent}  {group_labels[g2]:8s} : M={r[f'mean_{g2}']:.3f}  "
          f"SD={r[f'sd_{g2}']:.3f}  N={r[f'n_{g2}']}")
    print(f"{indent}  t-test   : t={r['t_stat']:.3f}  "
          f"p={r['t_p']:.4f}  {r['t_sig']}")
    print(f"{indent}  Mann-W   : U={r['U_stat']:.1f}  "
          f"p={r['U_p']:.4f}  {r['U_sig']}")
    print(f"{indent}  Cohen's d: {r['cohens_d']:.3f} ({r['d_interp']})")

def apply_fdr_corrections(results_list):
    groups_dict = {}
    for r in results_list:
        comp = r.get('component', '')
        meas = r.get('measure', '')
        if comp == 'behaviour':
            key = ('behaviour', meas)
        elif 'effect_' in str(meas):
            key = ('effect', meas)
        else:
            key = (comp, meas)
            
        if key not in groups_dict:
            groups_dict[key] = []
        groups_dict[key].append(r)
        
    def sig(p):
        if np.isnan(p): return 'n/a'
        if p < 0.001:   return '***'
        if p < 0.01:    return '**'
        if p < 0.05:    return '*'
        if p < 0.10:    return '†'
        return 'ns'

    # Apply FDR correction within each group
    for key, items in groups_dict.items():
        # Correct t-test p-values
        t_ps = [r['t_p'] for r in items]
        valid_t_idx = [idx for idx, p in enumerate(t_ps) if not np.isnan(p)]
        if len(valid_t_idx) > 0:
            _, adj_t_ps, _, _ = multipletests([t_ps[idx] for idx in valid_t_idx], method='fdr_bh')
            for idx, p_adj in zip(valid_t_idx, adj_t_ps):
                items[idx]['t_p_fdr'] = round(float(p_adj), 4)
                items[idx]['t_sig_fdr'] = sig(p_adj)
        else:
            for r in items:
                r['t_p_fdr'] = np.nan
                r['t_sig_fdr'] = 'n/a'
                
        for idx in range(len(items)):
            if 't_p_fdr' not in items[idx]:
                items[idx]['t_p_fdr'] = np.nan
                items[idx]['t_sig_fdr'] = 'ns' if not np.isnan(items[idx]['t_p']) else 'n/a'

        # Correct Mann-Whitney U test p-values
        u_ps = [r['U_p'] for r in items]
        valid_u_idx = [idx for idx, p in enumerate(u_ps) if not np.isnan(p)]
        if len(valid_u_idx) > 0:
            _, adj_u_ps, _, _ = multipletests([u_ps[idx] for idx in valid_u_idx], method='fdr_bh')
            for idx, p_adj in zip(valid_u_idx, adj_u_ps):
                items[idx]['U_p_fdr'] = round(float(p_adj), 4)
                items[idx]['U_sig_fdr'] = sig(p_adj)
        else:
            for r in items:
                r['U_p_fdr'] = np.nan
                r['U_sig_fdr'] = 'n/a'
                
        for idx in range(len(items)):
            if 'U_p_fdr' not in items[idx]:
                items[idx]['U_p_fdr'] = np.nan
                items[idx]['U_sig_fdr'] = 'ns' if not np.isnan(items[idx]['U_p']) else 'n/a'
                
    return results_list

def get_vals(df, comp, cond, grp, measure):
    return df[(df['component']==comp) &
              (df['condition']==cond) &
              (df['group']==grp)][measure].dropna().values

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['nback', 'stroop'], required=True)
    args = parser.parse_args()
    task = args.task

    print("=" * 65)
    print(f"STATISTICAL ANALYSIS — {task.upper()}")
    print(f"Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    # ── Paths ─────────────────────────────────────────────────────────────────────
    group_folder = Path('output/group')
    stats_folder = Path('output/stats')
    behav_folder = Path(f'data/behavioural/{task}')
    stats_folder.mkdir(parents=True, exist_ok=True)

    # ── Study parameters ──────────────────────────────────────────────────────────
    participants_file = Path('data/participants.csv')
    participants      = pd.read_csv(participants_file) if participants_file.exists() else None
    
    if participants is not None:
        group_map = dict(zip(participants['participant_id'], participants['group']))
        unique_groups = sorted([g for g in participants['group'].dropna().unique()])
        g1, g2 = unique_groups if len(unique_groups) == 2 else (g1, g2)
    else:
        group_map = {}
        g1, g2 = g1, g2
        
    groups = [g1, g2]

    if task == 'nback':
        conditions = ['nontarget/correct','target/hit','target/miss']
        cond_labels = {
            'nontarget/correct': 'Non-target',
            'target/hit':        'Target hit',
            'target/miss':       'Target miss',
        }
        OUTLIER_PIDS = {
            'target/hit':  ['P08','P05','P06'],
            'target/miss': ['P08','P07','P03'],
            'all':         ['P04'],
        }
        primary_components = ['N200','P300','P3b']
        exploratory_components = ['N1','P2','FSW']
        effect_cond_a, effect_cond_b = 'target/hit', 'nontarget/correct'
        effect_name = 'target_effect'
        effect_label = 'target effect (hit-nontarget)'
        fsw_cond = 'target/hit'
        fsw_label = "FSW mean amplitude — Target hit (frontal slow wave)"
        
        rt_col     = 'Stimulus.RT'
        acc_col    = 'Stimulus.ACC'
        tt_col     = 'TrialType'
        behav_conds = [('Target','Target'),('Non-target','Non-target')]
        behav_eff_lbl = 'Target RT effect (Target − Non-target)'
        behav_eff_cond = 'target-nontarget'
        
        outlier_exclusions = {
            'nontarget/correct': ['P04'],
            'target/hit':        ['P08','P05','P06','P04'],
            'target/miss':       ['P08','P07','P03','P04'],
        }
    else:
        conditions = ['congruent/correct','incongruent/correct','no_response']
        cond_labels = {
            'congruent/correct':   'Congruent',
            'incongruent/correct': 'Incongruent',
            'no_response':         'No response',
        }
        OUTLIER_PIDS = {
            'congruent/correct':   ['P03','P05','P08'],
            'incongruent/correct': ['P03'],
            'no_response':         ['P03'],
            'all':                 ['P03'],
        }
        primary_components = ['N200','P300','P3b']
        exploratory_components = ['N1','CSW']
        effect_cond_a, effect_cond_b = 'incongruent/correct', 'congruent/correct'
        effect_name = 'conflict_effect'
        effect_label = 'conflict effect (I-C)'
        csw_cond = 'incongruent/correct'
        csw_label = "CSW mean amplitude — Incongruent (conflict slow wave)"
        
        rt_col     = 'Stimulus3.RT'
        acc_col    = 'Stimulus3.ACC'
        tt_col     = 'Congruency'
        behav_conds = [('Congruent','Congruent'),('Incongruent','Incongruent')]
        behav_eff_lbl = 'Stroop RT effect (I-C)'
        behav_eff_cond = 'incongruent-congruent'
        
        outlier_exclusions = {
            'congruent/correct':   ['P03','P05','P08'],
            'incongruent/correct': ['P03'],
            'no_response':         ['P03'],
        }

    # ── Step 1: Load data ─────────────────────────────────────────────────────────
    print(f"\nSTEP 1: Loading data")

    ind_file = group_folder / f'group_{task}_individual_components.csv'
    if not ind_file.exists():
        print(f"ERROR: {ind_file} not found. Run 07_group_erp.py first.")
        sys.exit(1)

    df = pd.read_csv(ind_file)
    print(f"Loaded  : {ind_file.name}")
    print(f"PIDs    : {sorted(df['participant_id'].unique())}")
    print(f"Comps   : {sorted(df['component'].unique())}")

    # ── Step 2: PRIMARY — amplitude statistics ────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"STEP 2: PRIMARY — ERP AMPLITUDE ({group_labels[g1]} vs {group_labels[g2]})")
    print(f"{'='*65}")

    primary_results = []

    for comp in ['N200','P300']:
        for measure, mcol in [('mean amplitude','mean_amp_uv'),
                               ('peak amplitude','peak_amp_uv')]:
            print(f"\n── {comp} {measure} (µV) ──")
            for cond in conditions:
                ctrl   = get_vals(df, comp, cond, g1,  mcol)
                creat  = get_vals(df, comp, cond, g2, mcol)
                label  = f"{comp} {measure} — {cond_labels[cond]}"
                result = run_tests(ctrl, creat, label)
                result['component'] = comp
                result['condition'] = cond
                result['measure']   = mcol
                result['analysis']  = 'all_participants'
                outlier_pids = OUTLIER_PIDS.get(cond,[]) + OUTLIER_PIDS.get('all',[])
                result['outlier_participants_included'] = \
                    ','.join(sorted(set(p for p in outlier_pids
                                        if p in df['participant_id'].unique())))
                primary_results.append(result)
                print_primary(result)

    # ── Step 3: PRIMARY — latency statistics ──────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"STEP 3: PRIMARY — ERP LATENCY ({group_labels[g1]} vs {group_labels[g2]})")
    print(f"{'='*65}")

    for comp in ['N200','P300','P3b']:
        print(f"\n── {comp} peak latency (ms) ──")
        for cond in conditions:
            ctrl   = get_vals(df, comp, cond, g1,  'peak_lat_ms')
            creat  = get_vals(df, comp, cond, g2, 'peak_lat_ms')
            label  = f"{comp} peak latency — {cond_labels[cond]}"
            result = run_tests(ctrl, creat, label)
            result['component'] = comp
            result['condition'] = cond
            result['measure']   = 'peak_lat_ms'
            result['analysis']  = 'all_participants'
            outlier_pids = OUTLIER_PIDS.get(cond,[]) + OUTLIER_PIDS.get('all',[])
            result['outlier_participants_included'] = \
                ','.join(sorted(set(p for p in outlier_pids
                                    if p in df['participant_id'].unique())))
            primary_results.append(result)
            print_primary(result)

    # P3b vs P300 divergence
    print(f"\n── P3b vs P300 latency divergence (per participant) ──")
    print(f"\n{'PID':5s} {'Group':10s} {'Condition':25s} "
          f"{'P300 lat':>10} {'P3b lat':>10} {'Diverge?':>10}")
    print("-" * 65)
    for pid in sorted(df['participant_id'].unique()):
        grp = group_map.get(pid,'?')
        for cond in conditions:
            p300 = df[(df['participant_id']==pid) &
                      (df['component']=='P300') &
                      (df['condition']==cond)]['peak_lat_ms'].values
            p3b  = df[(df['participant_id']==pid) &
                      (df['component']=='P3b') &
                      (df['condition']==cond)]['peak_lat_ms'].values
            if len(p300)>0 and len(p3b)>0:
                diff    = abs(p300[0]-p3b[0])
                diverge = 'YES' if diff > 20 else 'no'
                print(f"  {pid:3s}  {grp:10s} {cond:25s} "
                      f"{p300[0]:>10.1f} {p3b[0]:>10.1f} {diverge:>10}")

    # ── Step 4: PRIMARY — task effect ───────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"STEP 4: PRIMARY — {effect_label.upper()}")
    print(f"{'='*65}")

    effect_results = []

    for comp in ['N200','P300']:
        for measure, mcol in [('mean amplitude','mean_amp_uv'),
                               ('peak amplitude','peak_amp_uv'),
                               ('peak latency','peak_lat_ms')]:
            print(f"\n── {comp} {measure} {effect_name.replace('_', ' ')} ──")
            g1_eff, g2_eff = [], []
            pid_effects = []
            for pid in sorted(df['participant_id'].unique()):
                sub = df[(df['participant_id']==pid) & (df['component']==comp)]
                grp = group_map.get(pid)
                cond_a_vals = sub[sub['condition']==effect_cond_a][mcol].values
                cond_b_vals = sub[sub['condition']==effect_cond_b][mcol].values
                if len(cond_a_vals)>0 and len(cond_b_vals)>0:
                    eff = cond_a_vals[0]-cond_b_vals[0]
                    pid_effects.append((pid, grp, eff))
                    if grp==g1:    g1_eff.append(eff)
                    elif grp==g2: g2_eff.append(eff)

            label  = f"{comp} {measure} {effect_label}"
            result = run_tests(np.array(g1_eff), np.array(g2_eff), label)
            result['component'] = comp
            result['condition'] = f'{effect_cond_a}-{effect_cond_b}'.replace('/correct', '').replace('/hit', '')
            result['measure']   = f'{effect_name}_{mcol}'
            result['analysis']  = 'all_participants'
            
            excl_a = OUTLIER_PIDS.get(effect_cond_a, [])
            excl_b = OUTLIER_PIDS.get(effect_cond_b, [])
            outlier_pids = list(set(excl_a + excl_b))
            
            result['outlier_participants_included'] = ','.join(sorted(set(p for p in outlier_pids if p in df['participant_id'].unique())))
            effect_results.append(result)
            print_primary(result)

            print(f"\n    Individual {effect_name.replace('_', ' ')}s:")
            print(f"    {'PID':5s} {'Group':10s} {'Effect':>10}")
            for pid, grp, eff in pid_effects:
                print(f"    {pid:5s} {grp:10s} {eff:>+10.3f}")

    primary_results.extend(effect_results)

    # ── Step 5: EXPLORATORY ───────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"STEP 5: EXPLORATORY — ERP AMPLITUDE")
    print(f"{'='*65}")

    exploratory_results = []

    # N1 — all conditions
    print(f"\n── N1 mean amplitude (µV) ──")
    for cond in conditions:
        ctrl  = get_vals(df, 'N1', cond, g1,  'mean_amp_uv')
        creat = get_vals(df, 'N1', cond, g2, 'mean_amp_uv')
        if len(ctrl)==0 and len(creat)==0: continue
        label  = f"N1 mean amplitude — {cond_labels[cond]}"
        result = run_tests(ctrl, creat, label)
        result['component']='N1'; result['condition']=cond
        result['measure']='mean_amp_uv'
        result['analysis']='all_participants'
        result['outlier_participants_included']=''
        exploratory_results.append(result)
        print_primary(result)

    if task == 'nback':
        # P2 — all conditions
        print(f"\n── P2 mean amplitude (µV) ──")
        for cond in conditions:
            ctrl  = get_vals(df, 'P2', cond, g1,  'mean_amp_uv')
            creat = get_vals(df, 'P2', cond, g2, 'mean_amp_uv')
            if len(ctrl)==0 and len(creat)==0: continue
            label  = f"P2 mean amplitude — {cond_labels[cond]}"
            result = run_tests(ctrl, creat, label)
            result['component']='P2'; result['condition']=cond
            result['measure']='mean_amp_uv'
            result['analysis']='all_participants'
            result['outlier_participants_included']=''
            exploratory_results.append(result)
            print_primary(result)

        # FSW — target/hit only
        print(f"\n── FSW mean amplitude (µV) — target/hit only ──")
        ctrl  = get_vals(df, 'FSW', 'target/hit', g1,  'mean_amp_uv')
        creat = get_vals(df, 'FSW', 'target/hit', g2, 'mean_amp_uv')
        if len(ctrl)>0 or len(creat)>0:
            label  = "FSW mean amplitude — Target hit (frontal slow wave)"
            result = run_tests(ctrl, creat, label)
            result['component']='FSW'; result['condition']='target/hit'
            result['measure']='mean_amp_uv'
            result['analysis']='all_participants'
            result['outlier_participants_included']=''
            exploratory_results.append(result)
            print_primary(result)
    else:
        # CSW — incongruent only
        print(f"\n── CSW mean amplitude (µV) — incongruent only ──")
        ctrl  = get_vals(df, 'CSW', 'incongruent/correct', g1,  'mean_amp_uv')
        creat = get_vals(df, 'CSW', 'incongruent/correct', g2, 'mean_amp_uv')
        if len(ctrl)>0 or len(creat)>0:
            label  = "CSW mean amplitude — Incongruent (conflict slow wave)"
            result = run_tests(ctrl, creat, label)
            result['component'] = 'CSW'
            result['condition'] = 'incongruent/correct'
            result['measure']   = 'mean_amp_uv'
            result['analysis']  = 'all_participants'
            result['outlier_participants_included'] = ''
            exploratory_results.append(result)
            print_primary(result)

    # ── Step 6: BEHAVIOURAL ───────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"STEP 6: BEHAVIOURAL STATISTICS")
    print(f"{'='*65}")

    behav_results = []
    behav_dfs     = []

    for pid, grp in group_map.items():
        bfile = behav_folder / f'{pid}_{task}_clean.csv'
        if not bfile.exists():
            print(f"  WARNING: {bfile} not found"); continue
        bdf = pd.read_csv(bfile)
        bdf['participant_id'] = pid
        bdf['group']          = grp
        behav_dfs.append(bdf)

    if behav_dfs:
        behav      = pd.concat(behav_dfs, ignore_index=True)
        if task == 'nback':
            behav_exp  = behav[behav.get('is_practice', pd.Series(0,index=behav.index))==0].copy()
        else:
            behav_exp = behav
            
        behav_resp = behav_exp[behav_exp['missed']==0].copy()

        print(f"\n── Reaction Time (ms) — correct trials only ──")
        for tt_lbl, tt_val in behav_conds:
            print(f"\n  {tt_lbl}:")
            g1_rt  = behav_resp[(behav_resp['group']==g1) &
                                  (behav_resp[tt_col]==tt_val) &
                                  (behav_resp[acc_col]==1)
                                 ].groupby('participant_id')[rt_col].mean().values
            g2_rt = behav_resp[(behav_resp['group']==g2) &
                                  (behav_resp[tt_col]==tt_val) &
                                  (behav_resp[acc_col]==1)
                                 ].groupby('participant_id')[rt_col].mean().values
            result = run_tests(g1_rt, g2_rt, f"RT — {tt_lbl}")
            result['component']='behaviour'; result['condition']=tt_lbl.lower()
            result['measure']='RT_ms'; result['analysis']='all_participants'
            result['outlier_participants_included']=''
            behav_results.append(result)
            print_primary(result)

        print(f"\n── {behav_eff_lbl} ──")
        g1_eff, g2_eff = [], []
        for pid in sorted(behav['participant_id'].unique()):
            sub  = behav_resp[behav_resp['participant_id']==pid]
            grp  = group_map.get(pid)
            cond_a = sub[(sub[tt_col]==behav_conds[1][1])&(sub[acc_col]==1)][rt_col].mean()
            cond_b = sub[(sub[tt_col]==behav_conds[0][1])&(sub[acc_col]==1)][rt_col].mean()
            if not (np.isnan(cond_a) or np.isnan(cond_b)):
                if grp==g1:    g1_eff.append(cond_a-cond_b)
                elif grp==g2: g2_eff.append(cond_a-cond_b)
        result = run_tests(np.array(g1_eff), np.array(g2_eff), behav_eff_lbl)
        result['component']='behaviour'; result['condition']=behav_eff_cond
        result['measure']='RT_effect_ms'; result['analysis']='all_participants'
        result['outlier_participants_included']=''
        behav_results.append(result)
        print_primary(result)

        print(f"\n── Accuracy ──")
        for tt_lbl, tt_val in behav_conds:
            print(f"\n  {tt_lbl}:")
            g1_acc  = behav_exp[(behav_exp['group']==g1) &
                                  (behav_exp[tt_col]==tt_val)
                                 ].groupby('participant_id')[acc_col].mean().values
            g2_acc = behav_exp[(behav_exp['group']==g2) &
                                  (behav_exp[tt_col]==tt_val)
                                 ].groupby('participant_id')[acc_col].mean().values
            result = run_tests(g1_acc, g2_acc, f"Accuracy — {tt_lbl}")
            result['component']='behaviour'; result['condition']=tt_lbl.lower()
            result['measure']='accuracy'; result['analysis']='all_participants'
            result['outlier_participants_included']=''
            behav_results.append(result)
            print_primary(result)

        if task == 'nback':
            print(f"\n── N-back load breakdown (RT by list) ──")
            exp_lists = [l for l in behav_exp['ListName'].unique()
                         if 'Prac' not in str(l)]
            for lst in sorted(exp_lists):
                g1_rt  = behav_resp[(behav_resp['group']==g1) &
                                      (behav_resp['ListName']==lst)
                                     ].groupby('participant_id')[rt_col].mean().values
                g2_rt = behav_resp[(behav_resp['group']==g2) &
                                      (behav_resp['ListName']==lst)
                                     ].groupby('participant_id')[rt_col].mean().values
                if len(g1_rt)==0 and len(g2_rt)==0: continue
                result = run_tests(g1_rt, g2_rt, f"RT — {lst}")
                result['component']='behaviour'; result['condition']=str(lst)
                result['measure']='RT_by_load_ms'; result['analysis']='all_participants'
                result['outlier_participants_included']=''
                behav_results.append(result)
                print_primary(result)

    primary_results.extend(behav_results)

    # ── Step 7: SENSITIVITY analysis ─────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"STEP 7: SENSITIVITY ANALYSIS")
    print(f"Re-running primary ERP statistics without outlier participants")
    print(f"{'='*65}")

    sensitivity_results = []

    for comp in ['N200','P300']:
        for measure, mcol in [('mean amplitude','mean_amp_uv'),
                               ('peak latency','peak_lat_ms')]:
            for cond in conditions:
                excl    = outlier_exclusions.get(cond,[])
                df_sens = df[~df['participant_id'].isin(excl)]
                ctrl    = get_vals(df_sens, comp, cond, g1,  mcol)
                creat   = get_vals(df_sens, comp, cond, g2, mcol)
                label   = f"{comp} {measure} — {cond_labels[cond]} (excl {excl})"
                result  = run_tests(ctrl, creat, label)
                result['component']        = comp
                result['condition']        = cond
                result['measure']          = mcol
                result['analysis']         = 'outliers_excluded'
                result['outliers_removed'] = ','.join(excl)
                sensitivity_results.append(result)

    # Apply FDR corrections
    primary_results = apply_fdr_corrections(primary_results)
    sensitivity_results = apply_fdr_corrections(sensitivity_results)
    exploratory_results = apply_fdr_corrections(exploratory_results)

    # Add transparency column to all results
    for r in primary_results:
        r['multiple_comparisons_correction'] = 'FDR (Benjamini-Hochberg) applied to fdr columns'
    for r in sensitivity_results:
        r['multiple_comparisons_correction'] = 'FDR (Benjamini-Hochberg) applied to fdr columns'
    for r in exploratory_results:
        r['multiple_comparisons_correction'] = 'FDR (Benjamini-Hochberg) applied to fdr columns'

    print(f"\n{'Comparison':52s} {'All p_FDR':>9} {'Excl p_FDR':>10} "
          f"{'All d':>8} {'Excl d':>8}")
    print("-" * 90)
    for sens in sensitivity_results:
        comp = sens['component']; cond = sens['condition']; meas = sens['measure']
        match = [r for r in primary_results
                 if r.get('component')==comp and
                    r.get('condition')==cond and
                    r.get('measure')==meas and
                    r.get('analysis')=='all_participants']
        if match:
            prim = match[0]
            def fmt(v):
                return f"{v:.4f}" if not (isinstance(v,float) and np.isnan(v)) else 'n/a'
            def fmtd(v):
                return f"{v:.3f}" if not (isinstance(v,float) and np.isnan(v)) else 'n/a'
            print(f"  {comp} {meas[:18]} {cond_labels.get(cond,cond)[:20]:20s} "
                  f"{fmt(prim['t_p_fdr']):>9} {fmt(sens['t_p_fdr']):>10} "
                  f"{fmtd(prim['cohens_d']):>8} {fmtd(sens['cohens_d']):>8}")

    # ── Step 8: Save CSVs ─────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"STEP 8: Saving statistics output")
    print(f"{'='*65}")

    # Specify explicit column order including FDR adjusted columns
    primary_cols = [
        'comparison', f'n_{g1}', f'n_{g2}', f'mean_{g1}', f'mean_{g2}',
        f'sd_{g1}', f'sd_{g2}', f'sem_{g1}', f'sem_{g2}',
        't_stat', 't_p', 't_p_fdr', 't_sig', 't_sig_fdr',
        'U_stat', 'U_p', 'U_p_fdr', 'U_sig', 'U_sig_fdr',
        'cohens_d', 'd_interp', 'component', 'condition', 'measure',
        'analysis', 'outlier_participants_included', 'multiple_comparisons_correction'
    ]

    exploratory_cols = [
        'comparison', f'n_{g1}', f'n_{g2}', f'mean_{g1}', f'mean_{g2}',
        f'sd_{g1}', f'sd_{g2}', f'sem_{g1}', f'sem_{g2}',
        't_stat', 't_p', 't_p_fdr', 't_sig', 't_sig_fdr',
        'U_stat', 'U_p', 'U_p_fdr', 'U_sig', 'U_sig_fdr',
        'cohens_d', 'd_interp', 'component', 'condition', 'measure',
        'analysis', 'outlier_participants_included', 'multiple_comparisons_correction'
    ]

    sensitivity_cols = [
        'comparison', f'n_{g1}', f'n_{g2}', f'mean_{g1}', f'mean_{g2}',
        f'sd_{g1}', f'sd_{g2}', f'sem_{g1}', f'sem_{g2}',
        't_stat', 't_p', 't_p_fdr', 't_sig', 't_sig_fdr',
        'U_stat', 'U_p', 'U_p_fdr', 'U_sig', 'U_sig_fdr',
        'cohens_d', 'd_interp', 'component', 'condition', 'measure',
        'analysis', 'outliers_removed', 'multiple_comparisons_correction'
    ]

    # Ensure all keys exist in result dicts to prevent KeyErrors
    for r in primary_results:
        for col in primary_cols:
            if col not in r:
                r[col] = np.nan
    for r in exploratory_results:
        for col in exploratory_cols:
            if col not in r:
                r[col] = np.nan
    for r in sensitivity_results:
        for col in sensitivity_cols:
            if col not in r:
                r[col] = np.nan

    df_primary     = pd.DataFrame(primary_results)[primary_cols]
    df_exploratory = pd.DataFrame(exploratory_results)[exploratory_cols]
    df_sensitivity = pd.DataFrame(sensitivity_results)[sensitivity_cols]

    primary_path     = stats_folder / f'{task}_statistics_primary.csv'
    exploratory_path = stats_folder / f'{task}_statistics_exploratory.csv'
    sensitivity_path = stats_folder / f'{task}_statistics_sensitivity.csv'

    df_primary.to_csv(str(primary_path),     index=False, encoding='utf-8-sig')
    df_exploratory.to_csv(str(exploratory_path), index=False, encoding='utf-8-sig')
    df_sensitivity.to_csv(str(sensitivity_path), index=False, encoding='utf-8-sig')

    print(f"\nSaved: {primary_path.name}")
    print(f"Saved: {exploratory_path.name}")
    print(f"Saved: {sensitivity_path.name}")

    # ── Step 9: Formatted summary ─────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"STEP 9: FORMATTED RESULTS TABLE — PRIMARY OUTCOMES (FDR-Corrected)")
    print(f"{'='*65}")
    print(f"\nSignificance: *** p_FDR<.001  ** p_FDR<.01  * p_FDR<.05  † p_FDR<.10  ns")
    print(f"Effect size : small |d|<0.5  medium 0.5≤|d|<0.8  large |d|≥0.8")
    print(f"\n{'Comparison':48s} {'Ctrl M':>8} {'Creat M':>8} "
          f"{'t':>7} {'t-p':>7} {'t-p_FDR':>9} {'U-p':>7} {'U-p_FDR':>9} {'d':>7} {'Size':>7}")
    print("-" * 135)

    for r in primary_results:
        if r.get('analysis') != 'all_participants': continue
        def fmt(v, dec=3):
            return f"{v:.{dec}f}" if not (isinstance(v,float) and np.isnan(v)) else 'n/a'
        print(f"  {r['comparison'][:46]:46s} "
              f"{fmt(r[f'mean_{g1}']):>8} {fmt(r[f'mean_{g2}']):>8} "
              f"{fmt(r['t_stat']):>7} {fmt(r['t_p'],4):>7} {fmt(r['t_p_fdr'],4):>9} "
              f"{fmt(r['U_p'],4):>7} {fmt(r['U_p_fdr'],4):>9} {fmt(r['cohens_d']):>7} "
              f"{r['d_interp']:>7}")

    print(f"\n{'='*65}")
    print(f"EXPLORATORY OUTCOMES — Welch t-test, Mann-Whitney U, Cohen's d")
    print(f"{'='*65}")
    print(f"\n{'Comparison':48s} {'Ctrl M':>8} {'Creat M':>8} "
          f"{'t':>7} {'t-p':>7} {'t-p_FDR':>9} {'U-p':>7} {'U-p_FDR':>9} {'d':>7} {'Size':>7}")
    print("-" * 135)
    for r in exploratory_results:
        def fmt(v, dec=3):
            return f"{v:.{dec}f}" if not (isinstance(v,float) and np.isnan(v)) else 'n/a'
        print(f"  {r['comparison'][:46]:46s} "
              f"{fmt(r[f'mean_{g1}']):>8} {fmt(r[f'mean_{g2}']):>8} "
              f"{fmt(r['t_stat']):>7} {fmt(r['t_p'],4):>7} {fmt(r['t_p_fdr'],4):>9} "
              f"{fmt(r['U_p'],4):>7} {fmt(r['U_p_fdr'],4):>9} {fmt(r['cohens_d']):>7} "
              f"{r['d_interp']:>7}")

    # ── Summary ───────────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"COMPLETE — {task.upper()} statistics")
    print(f"Finished : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*65}")

if __name__ == '__main__':
    main()
