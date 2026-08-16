"""
Purpose:
- model a resource-constrained RMG scenario in which the remaining blocks absorb displaced workload;
- study how a reduced resource pool propagates into service times and turnaround times;
- provide a second what-if case beyond the T22 closure scenario.

Inputs:
- the discovered baseline simulation bundle from the active baseline discovery flow;
- the Prosit simulator engine for a reduced effective RMG resource set.

Outputs:
- baseline and reduced-resource simulation logs;
- scenario comparison summaries for turnaround time and RMG activity performance.

Methodological note:
- the experiment is meant to test how the model behaves under plausible but degraded capacity assumptions.
"""

import os
import pickle
from copy import deepcopy

import pandas as pd
from prosit import SimulatorEngine


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
BASELINE_PICKLE = os.path.join(REPO_ROOT, 'scenarios', 'Archiv', 'scenarios', 'baseline_parameters.pkl')
OUT_ROOT = os.path.join(REPO_ROOT, 'data', 'processed', 'CTB', 'prosit_simulations', 'what_if_reduced_rmg_resources')
os.makedirs(OUT_ROOT, exist_ok=True)


def load_baseline_params():
    with open(BASELINE_PICKLE, 'rb') as f:
        return pickle.load(f)


def apply_reduced_rmg_pool(params, blocked_blocks=None, retained_blocks=None):
    params = deepcopy(params)
    blocked_blocks = set(blocked_blocks or ['T22'])
    retained_blocks = set(retained_blocks or ['T13', 'T14', 'T16', 'T17', 'T18', 'T19', 'T20', 'T21', 'T23', 'T24', 'T25', 'T26', 'T27'])

    for act, resources in list(params.act_to_resources.items()):
        if act.startswith('RMG_') or act in {'RMG_receive', 'RMG_delivery', 'RMG_mixed'}:
            filtered = [r for r in resources if r not in blocked_blocks and r in retained_blocks]
            if filtered:
                params.act_to_resources[act] = filtered

    for block in blocked_blocks:
        if block in params.calendars:
            params.calendars[block] = {day: {hour: False for hour in range(24)} for day in range(7)}

    return params


def add_kpis(df):
    df = df.copy()
    for col in ['enabled:timestamp', 'start:timestamp', 'time:timestamp']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    if 'enabled:timestamp' in df.columns and 'start:timestamp' in df.columns:
        df['waiting_time_min'] = (df['start:timestamp'] - df['enabled:timestamp']).dt.total_seconds() / 60.0
    else:
        df['waiting_time_min'] = pd.NA
    if 'start:timestamp' in df.columns and 'time:timestamp' in df.columns:
        df['service_time_min'] = (df['time:timestamp'] - df['start:timestamp']).dt.total_seconds() / 60.0
    else:
        df['service_time_min'] = pd.NA
    if {'case:concept:name', 'start:timestamp', 'time:timestamp'}.issubset(df.columns):
        case_start = df.groupby('case:concept:name')['start:timestamp'].transform('min')
        case_end = df.groupby('case:concept:name')['time:timestamp'].transform('max')
        df['turnaround_time_min'] = ((case_end - case_start).dt.total_seconds() / 60.0)
    else:
        df['turnaround_time_min'] = pd.NA
    return df


def summarize(df, label):
    df = add_kpis(df)
    turnaround = df.groupby('case:concept:name')['turnaround_time_min'].first().dropna()
    rmg = df[df['concept:name'].isin(['RMG_receive', 'RMG_delivery', 'RMG_mixed'])].copy()
    summary = {
        'scenario': label,
        'n_cases': int(df['case:concept:name'].nunique()),
        'n_events': int(len(df)),
        'mean_turnaround_min': float(turnaround.mean()) if not turnaround.empty else float('nan'),
        'p90_turnaround_min': float(turnaround.quantile(0.9)) if not turnaround.empty else float('nan'),
        'median_turnaround_min': float(turnaround.median()) if not turnaround.empty else float('nan'),
        'mean_rmg_service_min': float(rmg['service_time_min'].mean()) if not rmg.empty else float('nan'),
        'p90_rmg_service_min': float(rmg['service_time_min'].quantile(0.9)) if not rmg.empty else float('nan'),
        'mean_rmg_waiting_min': float(rmg['waiting_time_min'].mean()) if 'waiting_time_min' in rmg.columns and not rmg.empty else float('nan'),
        'mean_block_utilization': float(df['block_utilization'].mean()) if 'block_utilization' in df.columns else float('nan'),
    }
    act_summary = rmg.groupby('concept:name')['service_time_min'].agg([
        'mean', 'median', lambda s: s.quantile(0.9), 'max'
    ]).reset_index()
    act_summary.columns = ['activity', 'mean_service_time_min', 'median_service_time_min', 'p90_service_time_min', 'max_service_time_min']
    return summary, act_summary


def main():
    baseline_params = load_baseline_params()
    reduced_params = apply_reduced_rmg_pool(baseline_params, blocked_blocks=['T22'])

    baseline_log = SimulatorEngine(baseline_params).apply(n_traces=2000)
    reduced_log = SimulatorEngine(reduced_params).apply(n_traces=2000)

    baseline_log.to_csv(os.path.join(OUT_ROOT, 'baseline_reference_sim_log.csv'), index=False)
    reduced_log.to_csv(os.path.join(OUT_ROOT, 't22_closed_reduced_rmg_pool_sim_log.csv'), index=False)

    baseline_summary, baseline_acts = summarize(baseline_log, 'baseline')
    reduced_summary, reduced_acts = summarize(reduced_log, 't22_closed_reduced_rmg_pool')

    summary_df = pd.DataFrame([baseline_summary, reduced_summary])
    summary_df.to_csv(os.path.join(OUT_ROOT, 'scenario_comparison_summary.csv'), index=False)
    baseline_acts.to_csv(os.path.join(OUT_ROOT, 'baseline_rmg_activity_summary.csv'), index=False)
    reduced_acts.to_csv(os.path.join(OUT_ROOT, 't22_closed_reduced_rmg_pool_activity_summary.csv'), index=False)

    print('Reduced-RMG resource experiment complete.')
    print(summary_df.to_string(index=False))
    print('\nRMG activity summary (reduced resource pool):')
    print(reduced_acts.to_string(index=False))


if __name__ == '__main__':
    main()
