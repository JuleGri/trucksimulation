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

script_dir = os.path.dirname(os.path.abspath(__file__))
# Input CSV relative to script_dir
input_csv = os.path.join(script_dir, "..", "..", "data", "processed", "CTB", "s6_eventlog_target_rank_features.csv")
input_csv = os.path.normpath(input_csv)

if not os.path.exists(input_csv):
    raise FileNotFoundError(f"Eventlog CSV not found at expected location: {input_csv}")

print('Reading event CSV from', input_csv)
df = pd.read_csv(input_csv)

# Convert relevant timestamp columns to datetime
for col in ["enabled:timestamp", "start:timestamp", "time:timestamp"]:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    else:
        print(f"Warning: column {col} not found in input CSV")

# Sample training cases like the notebook uses (24k) if available
n_cases = 24000
case_ids = df['case:concept:name'].drop_duplicates()
if len(case_ids) > n_cases:
    case_sample = case_ids.sample(n_cases, random_state=42)
    df = df[df['case:concept:name'].isin(case_sample)].copy()
    print(f'Sampled {len(case_sample)} cases for empirical stats')
else:
    print(f'Using all {len(case_ids)} cases for empirical stats')

# Compute empirical durations in minutes
if 'time:timestamp' in df.columns and 'start:timestamp' in df.columns:
    df['service_time_min'] = (df['time:timestamp'] - df['start:timestamp']).dt.total_seconds() / 60
else:
    df['service_time_min'] = np.nan

if 'start:timestamp' in df.columns and 'enabled:timestamp' in df.columns:
    df['waiting_time_min'] = (df['start:timestamp'] - df['enabled:timestamp']).dt.total_seconds() / 60
else:
    df['waiting_time_min'] = np.nan

# Clip negative waiting times to zero (common timestamp artifact)
df['waiting_time_min'] = df['waiting_time_min'].clip(lower=0)

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
