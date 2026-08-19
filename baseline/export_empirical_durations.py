"""
Purpose:
- export empirical service-time and waiting-time summaries from the processed CTB event log;
- generate the statistics needed for distribution fitting and simulation-parameter generation;
- create a reproducible baseline statistics artifact for the thesis pipeline.

Inputs:
- the processed CTB event log in data/processed/CTB/;
- the case and activity structure contained in the baseline event log.

Outputs:
- empirical_activity_durations.csv;
- empirical_resource_waiting_times.csv.

Methodological note:
- this script is a preprocessing bridge between raw event data and the fitted simulator parameters.
"""
import os
import pandas as pd
import numpy as np

from _eventlog_source import parse_input_arg, resolve_event_csv

script_dir = os.path.dirname(os.path.abspath(__file__))
# Route through the shared resolver so this preprocessing step also honours
# the s6_train.csv hold-out (see validation/01_train_test_split.py).
input_csv = resolve_event_csv(parse_input_arg())

if not os.path.exists(input_csv):
    raise FileNotFoundError(f"Eventlog CSV not found at expected location: {input_csv}")

print('Reading event CSV from', input_csv)
df = pd.read_csv(input_csv)

# Convert relevant timestamp columns to datetime
for col in ["start:timestamp", "time:timestamp"]:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    else:
        print(f"Warning: column {col} not found in input CSV")

# When reading from the held-out training log (s6_train.csv), every case
# already belongs to the training partition — no additional random sub-
# sampling is applied so that empirical stats reflect the full training
# distribution. The historical 24 000-case sample is preserved only when
# reading from the full log for legacy comparisons.
from _eventlog_source import is_train_log

case_ids = df['case:concept:name'].drop_duplicates()
if is_train_log(input_csv):
    print(f'Held-out training log detected: keeping all {len(case_ids)} cases.')
else:
    n_cases = 24000
    if len(case_ids) > n_cases:
        case_sample = case_ids.sample(n_cases, random_state=42)
        df = df[df['case:concept:name'].isin(case_sample)].copy()
        print(f'Sampled {len(case_sample)} cases for empirical stats (legacy full-log mode)')
    else:
        print(f'Using all {len(case_ids)} cases for empirical stats')

# Compute empirical durations in minutes
if 'time:timestamp' in df.columns and 'start:timestamp' in df.columns:
    df['service_time_min'] = (df['time:timestamp'] - df['start:timestamp']).dt.total_seconds() / 60
else:
    df['service_time_min'] = np.nan

# Note (2026-08 revision): waiting_time_min is no longer computed from
# a handcrafted enabled:timestamp. Pre-service delay is a model concept
# derived by ProSiT during discovery. We retain the column as NaN for
# schema compatibility with downstream scripts.
df['waiting_time_min'] = np.nan

# Remove extreme outliers per-activity/service and per-resource/waiting: 99.9 percentile
# Service: per activity
service_trimmed = df.copy()
service_stats = []
for act, g in df.groupby('concept:name'):
    s = g['service_time_min']
    if s.dropna().empty:
        continue
    cutoff = s.quantile(0.999)
    sg = g[s <= cutoff]
    # recompute stats
    service_stats.append({
        'activity': act,
        'service_mean': float(sg['service_time_min'].mean()),
        'service_std': float(sg['service_time_min'].std(ddof=0)),
        'service_median': float(sg['service_time_min'].median()),
        'service_min': float(sg['service_time_min'].min()),
        'service_max': float(sg['service_time_min'].max()),
        'service_count': int(sg['service_time_min'].count())
    })

# Waiting: per resource (org:resource)
waiting_stats = []
if 'org:resource' in df.columns:
    for res, g in df.groupby('org:resource'):
        w = g['waiting_time_min']
        if w.dropna().empty:
            continue
        cutoff = w.quantile(0.999)
        wg = g[w <= cutoff]
        waiting_stats.append({
            'resource': res,
            'waiting_mean': float(wg['waiting_time_min'].mean()),
            'waiting_std': float(wg['waiting_time_min'].std(ddof=0)),
            'waiting_median': float(wg['waiting_time_min'].median()),
            'waiting_min': float(wg['waiting_time_min'].min()),
            'waiting_max': float(wg['waiting_time_min'].max()),
            'waiting_count': int(wg['waiting_time_min'].count())
        })
else:
    print('Warning: org:resource column not found; waiting stats per resource will be empty')

# Save results to baseline folder (not discovery_params)
out_activity = os.path.join(script_dir, 'empirical_activity_durations.csv')
out_resource = os.path.join(script_dir, 'empirical_resource_waiting_times.csv')

pd.DataFrame(service_stats).sort_values('activity').to_csv(out_activity, index=False)
print('Wrote', out_activity)

if waiting_stats:
    pd.DataFrame(waiting_stats).sort_values('resource').to_csv(out_resource, index=False)
    print('Wrote', out_resource)
else:
    print('No waiting stats written (org:resource missing or empty)')

print('Done')
