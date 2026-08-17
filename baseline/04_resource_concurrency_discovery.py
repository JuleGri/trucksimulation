"""
Purpose:
- infer effective resource capacities from the temporal overlap of pooled activities;
- summarize median, p90, and maximum observed concurrency per resource pool;
- construct a capacity recommendation that is transparent and compatible with the simulation model.

Inputs:
- the processed CTB event log with org:resource, start:timestamp, and time:timestamp;
- the resource pooling conventions used in the baseline discovery pipeline.

Outputs:
- a discovered_resource_capacities.csv summary for the simulation bundle;
- a capacity candidate set that can be used for robustness and scenario sensitivity checks.

Methodological note:
- resource discovery is based on observed overlap behavior rather than exact worker schedules;
- the resulting capacities are effective capacities under the pooled-resource event-log abstraction.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from _eventlog_source import add_input_arg, paramset_suffix_for, resolve_event_csv


def map_resource_pool(resource_name):
    if resource_name is None or pd.isna(resource_name):
        return 'UNKNOWN'
    value = str(resource_name).upper()
    if 'LL' in value:
        return 'LL'
    if 'HO2' in value or 'HO' in value:
        return 'HO2'
    if 'VC' in value:
        return 'VC'
    if 'RMG' in value:
        return 'RMG'
    if 'GATEIN' in value or 'GATE IN' in value:
        return 'GateIn'
    if 'GATEOUT' in value or 'GATE OUT' in value:
        return 'GateOut'
    return 'OTHER'


def compute_pool_concurrency(df):
    required = {'org:resource', 'start:timestamp', 'time:timestamp'}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f'Missing expected columns for concurrency discovery: {sorted(missing)}')

    subset = df[['org:resource', 'start:timestamp', 'time:timestamp']].copy()
    subset['resource_pool'] = subset['org:resource'].map(map_resource_pool)
    subset = subset.dropna(subset=['start:timestamp', 'time:timestamp'])
    subset['start:timestamp'] = pd.to_datetime(subset['start:timestamp'], errors='coerce')
    subset['time:timestamp'] = pd.to_datetime(subset['time:timestamp'], errors='coerce')
    subset = subset.dropna(subset=['start:timestamp', 'time:timestamp'])
    subset = subset[subset['time:timestamp'] >= subset['start:timestamp']].copy()

    rows = []
    for pool, group in subset.groupby('resource_pool', sort=True):
        if pool == 'UNKNOWN':
            continue
        start_ts = group['start:timestamp'].dt.floor('s')
        end_ts = group['time:timestamp'].dt.floor('s')
        if start_ts.empty:
            continue

        min_ts = start_ts.min()
        max_ts = end_ts.max()
        if pd.isna(min_ts) or pd.isna(max_ts):
            continue

        seconds = pd.date_range(start=min_ts.floor('s'), end=max_ts.floor('s'), freq='1s')
        counts = pd.Series(0, index=seconds, dtype='int64')
        for s, e in zip(start_ts, end_ts):
            if pd.isna(s) or pd.isna(e):
                continue
            if e < s:
                continue
            counts.loc[s:e] += 1

        q = counts.quantile([0.5, 0.9, 1.0])
        rows.append({
            'resource_pool': pool,
            'median_concurrency': int(round(float(q.loc[0.5]))),
            'p90_concurrency': int(round(float(q.loc[0.9]))),
            'max_concurrency': int(round(float(q.loc[1.0]))),
            'night_cap': 1,
            'candidate_capacities': [1, 2, 3],
        })

    summary = pd.DataFrame(rows)
    if summary.empty:
        raise RuntimeError('No concurrency statistics were produced from the event log.')
    summary['chosen_capacity_baseline'] = summary['p90_concurrency'].clip(lower=1, upper=3)
    summary['chosen_capacity_baseline'] = np.where(summary['resource_pool'].isin(['RMG']), 1, summary['chosen_capacity_baseline'])
    return summary


def main():
    parser = argparse.ArgumentParser(description='Resource-concurrency discovery on the held-out training log.')
    add_input_arg(parser)
    args = parser.parse_args()

    event_csv = resolve_event_csv(args.input_csv)
    df = pd.read_csv(event_csv)
    summary = compute_pool_concurrency(df)
    summary['source_eventlog'] = os.path.basename(event_csv)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    from _eventlog_source import resolve_paramset_dir
    param_dir = resolve_paramset_dir(script_dir, paramset_suffix_for(event_csv))
    output_path = os.path.join(param_dir, 'discovered_resource_capacities.csv')
    summary.to_csv(output_path, index=False)

    print('Resource concurrency discovery complete.')
    print(f'Wrote: {output_path}')
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
