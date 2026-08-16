"""
Purpose:
- reuse the discovered baseline simulation model under a blocked-block scenario;
- implement a T22 closure experiment for the thesis what-if analysis;
- compare the baseline simulation against the adapted scenario on turnaround time and block utilization.

Inputs:
- the discovered baseline parameter bundle stored in the archived scenario artifacts;
- the Prosit simulator engine available in the project environment.

Outputs:
- baseline and scenario simulation logs;
- a comparison summary CSV covering turnaround time, RMG service time, and utilization effects.

Methodological note:
- this experiment is designed to test white-box adaptation behavior under a concrete operational disruption.
"""

import os
import pickle
from copy import deepcopy

import pandas as pd
from prosit import SimulatorEngine


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
BASELINE_PICKLE = os.path.join(REPO_ROOT, 'scenarios', 'Archiv', 'scenarios', 'baseline_parameters.pkl')
OUT_ROOT = os.path.join(REPO_ROOT, 'data', 'processed', 'CTB', 'prosit_simulations', 'what_if_T22_closed')
os.makedirs(OUT_ROOT, exist_ok=True)


def load_baseline_params():
    with open(BASELINE_PICKLE, 'rb') as f:
        return pickle.load(f)


def apply_block_closure(params, blocked_blocks=None):
    params = deepcopy(params)
    blocked_blocks = set(blocked_blocks or ['T22'])
    fallback_pool = ['T13', 'T14', 'T16', 'T17', 'T18', 'T19', 'T20', 'T21', 'T23', 'T24', 'T25', 'T26', 'T27']

    for act, resources in list(params.act_to_resources.items()):
        kept = [r for r in resources if r not in blocked_blocks]
        if not kept:
            for candidate in fallback_pool:
                if candidate not in blocked_blocks:
                    kept = [candidate]
                    break
        if kept:
            params.act_to_resources[act] = kept

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
        'mean_block_utilization_rmg': float(df[df['block'].isin(['T22','T13','T14','T17','T16','T21','T24'])]['block_utilization'].mean()) if 'block_utilization' in df.columns and 'block' in df.columns else float('nan'),
    }

    act_summary = rmg.groupby('concept:name')['service_time_min'].agg([
        'mean', 'median', lambda s: s.quantile(0.9), 'max'
    ]).reset_index()
    act_summary.columns = ['activity', 'mean_service_time_min', 'median_service_time_min', 'p90_service_time_min', 'max_service_time_min']

    block_summary = df[df['block'].notna()].groupby('block').agg(
        mean_block_utilization=('block_utilization', 'mean'),
        p90_block_utilization=('block_utilization', lambda s: s.quantile(0.9)),
        event_count=('case:concept:name', 'count')
    ).reset_index()

    return summary, act_summary, block_summary


def main():
    baseline_params = load_baseline_params()
    baseline_engine = SimulatorEngine(baseline_params)
    baseline_log = baseline_engine.apply(n_traces=2000)
    baseline_log.to_csv(os.path.join(OUT_ROOT, 'baseline_reference_sim_log.csv'), index=False)

    scenario_params = apply_block_closure(baseline_params, blocked_blocks=['T22'])
    scenario_engine = SimulatorEngine(scenario_params)
    scenario_log = scenario_engine.apply(n_traces=2000)
    scenario_log.to_csv(os.path.join(OUT_ROOT, 't22_closed_sim_log.csv'), index=False)

    baseline_summary, baseline_acts, baseline_blocks = summarize(baseline_log, 'baseline')
    scenario_summary, scenario_acts, scenario_blocks = summarize(scenario_log, 't22_closed')

    summary_df = pd.DataFrame([baseline_summary, scenario_summary])
    summary_df.to_csv(os.path.join(OUT_ROOT, 'scenario_comparison_summary.csv'), index=False)

    baseline_acts.to_csv(os.path.join(OUT_ROOT, 'baseline_rmg_activity_summary.csv'), index=False)
    scenario_acts.to_csv(os.path.join(OUT_ROOT, 't22_closed_rmg_activity_summary.csv'), index=False)
    baseline_blocks.to_csv(os.path.join(OUT_ROOT, 'baseline_block_utilization_summary.csv'), index=False)
    scenario_blocks.to_csv(os.path.join(OUT_ROOT, 't22_closed_block_utilization_summary.csv'), index=False)

    print('Baseline vs T22 closure experiment complete.')
    print(summary_df.to_string(index=False))

    print('\nRMG activity summary (T22 closed):')
    print(scenario_acts.to_string(index=False))


if __name__ == '__main__':
    main()
