"""
Purpose:
- reuse the discovered ProSiT baseline (workload-features variant, adopted
  as default after the sensitivity comparison in Chapter 5) under a
  reduced RMG resource pool;
- study how the remaining blocks absorb the workload displaced from T22;
- provide a second what-if case beyond the T22 closure scenario, with the
  same simulation window as the held-out validation so both experiments
  are directly comparable.

Inputs:
- baseline/discovery_params/params_<latest>_train80/prosit_discovery_workload/
  prosit_params.pkl        (ProSiT parameters fit on s6_train.csv with
                            use_workload_features=True; override via
                            --params);
- validation/results/split_manifest.json (test-window cutoff & size).

Outputs:
- baseline and reduced-resource simulation logs;
- scenario comparison summaries for turnaround time and RMG activity performance.

Methodological note:
- unlike the plain T22 closure, this scenario also removes the RMG capacity
  associated with T22 from the discovered resource pool, testing how the
  workload-aware ProSiT model reallocates work in the presence of a
  hard capacity reduction.
"""

import argparse
import json
import os
import pickle
from copy import deepcopy
from datetime import datetime

import pandas as pd
from prosit import SimulatorEngine


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def _resolve_default_pickle() -> str:
    discovery_root = os.path.join(REPO_ROOT, 'baseline', 'discovery_params')
    if os.path.isdir(discovery_root):
        candidates = sorted(
            (d for d in os.listdir(discovery_root)
             if d.endswith('_train80') and os.path.isdir(os.path.join(discovery_root, d))),
            reverse=True,
        )
        for cand in candidates:
            pkl = os.path.join(discovery_root, cand, 'prosit_discovery_workload', 'prosit_params.pkl')
            if os.path.exists(pkl):
                return pkl
    return os.path.join(REPO_ROOT, 'scenarios', 'Archiv', 'scenarios', 'baseline_parameters.pkl')


DEFAULT_BASELINE_PICKLE = _resolve_default_pickle()
DEFAULT_MANIFEST = os.path.join(REPO_ROOT, 'validation', 'results', 'split_manifest.json')
OUT_ROOT = os.path.join(REPO_ROOT, 'data', 'processed', 'CTB', 'prosit_simulations', 'what_if_reduced_rmg_resources')
os.makedirs(OUT_ROOT, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--params', default=DEFAULT_BASELINE_PICKLE,
                        help='Path to a pickled prosit SimulatorParameters object.')
    parser.add_argument('--manifest', default=DEFAULT_MANIFEST,
                        help='split_manifest.json used to size the simulation window.')
    parser.add_argument('--n-traces', type=int, default=None,
                        help='Number of cases to simulate. Defaults to the size of the held-out test set.')
    parser.add_argument('--t-start', default=None,
                        help='ISO timestamp for the simulation start. Defaults to cutoff_arrival_ts '
                             'from the manifest.')
    parser.add_argument('--blocked-blocks', nargs='+', default=['T22'],
                        help='RMG blocks to remove from the resource pool.')
    parser.add_argument('--retained-blocks', nargs='+',
                        default=['T13', 'T14', 'T16', 'T17', 'T18', 'T19', 'T20', 'T21', 'T23', 'T24', 'T25', 'T26', 'T27'],
                        help='RMG blocks retained after the reduction (order-independent).')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def load_baseline_params(pickle_path: str):
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(
            f'ProSiT baseline pickle not found at {pickle_path}. '
            'Run baseline/07_run_prosit_discovery.py first.'
        )
    print(f'[what_if_reduced_rmg] Loading params from {pickle_path}')
    with open(pickle_path, 'rb') as f:
        return pickle.load(f)


def _resolve_simulation_window(args) -> tuple[int, datetime | None]:
    n_traces = args.n_traces
    t_start = None
    if os.path.exists(args.manifest):
        with open(args.manifest, 'r') as fh:
            manifest = json.load(fh)
        if n_traces is None:
            n_traces = int(manifest.get('test', {}).get('n_cases', 2000))
        cutoff = args.t_start or manifest.get('cutoff_arrival_ts')
        if cutoff:
            t_start = datetime.fromisoformat(cutoff)
    else:
        n_traces = n_traces or 2000
    if args.t_start and t_start is None:
        t_start = datetime.fromisoformat(args.t_start)
    return int(n_traces), t_start


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
    # ProSiT 1.0.3 sometimes emits the same column twice in the simulated log
    # (e.g. enabled:timestamp appears once as an event attribute and once as a
    # case-level attribute). Deduplicate before any coercion so pd.to_datetime
    # sees a Series, not a DataFrame slice.
    df = df.loc[:, ~df.columns.duplicated()]
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
    args = parse_args()
    n_traces, t_start = _resolve_simulation_window(args)
    print(f'[what_if_reduced_rmg] n_traces={n_traces:,}  t_start={t_start}  '
          f'blocked={args.blocked_blocks}  retained={len(args.retained_blocks)} blocks')

    baseline_params = load_baseline_params(args.params)
    reduced_params = apply_reduced_rmg_pool(
        baseline_params,
        blocked_blocks=args.blocked_blocks,
        retained_blocks=args.retained_blocks,
    )

    baseline_engine = SimulatorEngine(baseline_params)
    reduced_engine = SimulatorEngine(reduced_params)
    if t_start is None:
        baseline_log = baseline_engine.apply(n_traces=n_traces)
        reduced_log = reduced_engine.apply(n_traces=n_traces)
    else:
        baseline_log = baseline_engine.apply(n_traces=n_traces, t_start=t_start)
        reduced_log = reduced_engine.apply(n_traces=n_traces, t_start=t_start)

    baseline_log.to_csv(os.path.join(OUT_ROOT, 'baseline_reference_sim_log.csv'), index=False)
    reduced_log.to_csv(os.path.join(OUT_ROOT, 't22_closed_reduced_rmg_pool_sim_log.csv'), index=False)

    baseline_summary, baseline_acts = summarize(baseline_log, 'baseline')
    reduced_summary, reduced_acts = summarize(reduced_log, 't22_closed_reduced_rmg_pool')

    summary_df = pd.DataFrame([baseline_summary, reduced_summary])
    summary_df.to_csv(os.path.join(OUT_ROOT, 'scenario_comparison_summary.csv'), index=False)
    baseline_acts.to_csv(os.path.join(OUT_ROOT, 'baseline_rmg_activity_summary.csv'), index=False)
    reduced_acts.to_csv(os.path.join(OUT_ROOT, 't22_closed_reduced_rmg_pool_activity_summary.csv'), index=False)

    with open(os.path.join(OUT_ROOT, 'scenario_run_summary.json'), 'w') as fh:
        json.dump({
            'params_pickle': args.params,
            'manifest': args.manifest,
            'n_traces': int(n_traces),
            't_start': None if t_start is None else t_start.isoformat(),
            'blocked_blocks': list(args.blocked_blocks),
            'retained_blocks': list(args.retained_blocks),
            'seed': args.seed,
        }, fh, indent=2)

    print('Reduced-RMG resource experiment complete.')
    print(summary_df.to_string(index=False))
    print('\nRMG activity summary (reduced resource pool):')
    print(reduced_acts.to_string(index=False))


if __name__ == '__main__':
    main()
