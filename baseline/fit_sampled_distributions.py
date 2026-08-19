"""
Purpose:
- fit parametric service and waiting-time distributions to the sampled event log;
- compare candidate families per activity and per resource with distribution-quality criteria;
- update the simulator bundle with the best available sampled-fit families.

Inputs:
- the processed CTB event log;
- the sampled baseline cases used for distribution fitting;
- the current simulator bundle under discovery_params/<paramset>/.

Outputs:
- sampled service-fit summaries;
- sampled waiting-fit summaries;
- updated simulator bundle files that reflect the fitted families.

Methodological note:
- this script supports the distribution-fitting layer used before simulation parameter assembly.
"""

import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
from scipy import stats

from _eventlog_source import is_train_log, parse_input_arg, resolve_event_csv

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
# Route through the shared resolver so the sampled-distribution fits also
# consume the held-out training log by default.
raw_csv = resolve_event_csv(parse_input_arg())

if not os.path.exists(raw_csv):
    raise FileNotFoundError(f"Event CSV not found at {raw_csv}")

print('Reading raw event CSV from', raw_csv)
df = pd.read_csv(raw_csv)

# timestamps
for col in ['start:timestamp','time:timestamp']:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

# sample cases (only when reading the full log — the training log is
# already the correct hold-out and does not need extra sub-sampling)
case_ids = df['case:concept:name'].drop_duplicates()
if is_train_log(raw_csv):
    print(f'Held-out training log detected: keeping all {len(case_ids)} cases')
else:
    n_cases = 24000
    if len(case_ids) > n_cases:
        sample_ids = case_ids.sample(n_cases, random_state=42)
        df = df[df['case:concept:name'].isin(sample_ids)].copy()
        print(f'Sampled {len(sample_ids)} cases (legacy full-log mode)')
    else:
        print(f'Using all {len(case_ids)} cases')

# compute durations in minutes
df['service_time_min'] = (df['time:timestamp'] - df['start:timestamp']).dt.total_seconds() / 60
# Note (2026-08): waiting_time_min is no longer computed from a handcrafted
# enabled:timestamp. Pre-service delay is a model concept derived by ProSiT.
df['waiting_time_min'] = np.nan

# candidate distributions
candidates = ['lognorm','gamma','weibull_min','expon','norm']

min_samples = 30

service_results = []
waiting_results = []

# helper AIC
def compute_aic(dist, params, data):
    # number of parameters k
    argcount = len(params) if isinstance(params, (list, tuple)) else 1
    # compute log-likelihood
    with np.errstate(divide='ignore'):
        ll = np.sum(dist.logpdf(data, *params))
    k = argcount
    aic = 2*k - 2*ll
    return float(aic)

# Fit per activity (service)
for act, g in df.groupby('concept:name'):
    data = g['service_time_min'].dropna().values
    if len(data) < min_samples:
        print(f'Skipping activity {act} - insufficient samples ({len(data)})')
        continue
    # remove zeros and negative for positive-support dists
    data_pos = data[data>0]
    best = None
    best_aic = None
    best_p = None
    for cname in candidates:
        dist = getattr(stats, cname)
        try:
            if cname in ['lognorm','gamma','weibull_min']:
                if len(data_pos) < min_samples:
                    continue
                # fix loc=0 for positive distributions
                params = dist.fit(data_pos, floc=0)
                args = params
                # ks test on data_pos
                ks = stats.kstest(data_pos, cname, args=args)
                aic = compute_aic(dist, params, data_pos)
                pval = ks.pvalue
            else:
                params = dist.fit(data)
                ks = stats.kstest(data, cname, args=params)
                aic = compute_aic(dist, params, data)
                pval = ks.pvalue
            if best is None or (pval > best_p) or (best_p is not None and pval==best_p and aic < best_aic):
                best = (cname, params, pval, aic)
                best_p = pval
                best_aic = aic
        except Exception as e:
            # fitting failed for this dist
            continue
    if best is None:
        chosen_family = 'empirical'
        chosen_params = None
        ks_p = None
        aic = None
    else:
        chosen_family, chosen_params, ks_p, aic = best
    service_results.append({'activity': act, 'chosen_family': chosen_family, 'chosen_params': str(chosen_params), 'ks_pvalue': ks_p, 'aic': aic, 'sample_count': len(data)})
    print(f'Activity {act}: chosen {chosen_family} p={ks_p} aic={aic}')

# Fit per resource (waiting)
if 'org:resource' in df.columns:
    for res, g in df.groupby('org:resource'):
        data = g['waiting_time_min'].dropna().values
        if len(data) < min_samples:
            print(f'Skipping resource {res} - insufficient samples ({len(data)})')
            continue
        data_pos = data[data>0]
        best = None
        best_aic = None
        best_p = None
        for cname in candidates:
            dist = getattr(stats, cname)
            try:
                if cname in ['lognorm','gamma','weibull_min']:
                    if len(data_pos) < min_samples:
                        continue
                    params = dist.fit(data_pos, floc=0)
                    ks = stats.kstest(data_pos, cname, args=params)
                    aic = compute_aic(dist, params, data_pos)
                    pval = ks.pvalue
                else:
                    params = dist.fit(data)
                    ks = stats.kstest(data, cname, args=params)
                    aic = compute_aic(dist, params, data)
                    pval = ks.pvalue
                if best is None or (pval > best_p) or (best_p is not None and pval==best_p and aic < best_aic):
                    best = (cname, params, pval, aic)
                    best_p = pval
                    best_aic = aic
            except Exception:
                continue
        if best is None:
            chosen_family = 'empirical'
            chosen_params = None
            ks_p = None
            aic = None
        else:
            chosen_family, chosen_params, ks_p, aic = best
        waiting_results.append({'resource': res, 'chosen_family': chosen_family, 'chosen_params': str(chosen_params), 'ks_pvalue': ks_p, 'aic': aic, 'sample_count': len(data)})
        print(f'Resource {res}: chosen {chosen_family} p={ks_p} aic={aic}')
else:
    print('No org:resource column found; skipping waiting fits')

# Save results
out_dir_candidates = [
    os.path.join(script_dir, 'discovery_params'),
    script_dir
]
paramset_dir = None
if os.path.exists(os.path.join(script_dir, 'discovery_params')):
    candidates = [p for p in os.listdir(os.path.join(script_dir, 'discovery_params')) if os.path.isdir(os.path.join(script_dir, 'discovery_params', p))]
    if candidates:
        paramset_dir = os.path.join(script_dir, 'discovery_params', candidates[0])

if paramset_dir is None:
    paramset_dir = os.path.join(script_dir, 'discovery_params', 'params_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(paramset_dir, exist_ok=True)

service_df = pd.DataFrame(service_results)
waiting_df = pd.DataFrame(waiting_results)
service_df.to_csv(os.path.join(paramset_dir, 'distribution_fits_service_sampled.csv'), index=False)
waiting_df.to_csv(os.path.join(paramset_dir, 'distribution_fits_waiting_sampled.csv'), index=False)
print('Saved sampled distribution fits to', paramset_dir)

# Update simulator bundle if exists
bundle_path = os.path.join(paramset_dir, 'simulator_parameters_bundle.json')
if not os.path.exists(bundle_path):
    # try root paramset
    bundle_path = os.path.join(script_dir, 'discovery_params', candidates[0], 'simulator_parameters_bundle.json') if 'candidates' in locals() and candidates else bundle_path

if os.path.exists(bundle_path):
    with open(bundle_path, 'r') as f:
        bundle = json.load(f)
    # update activities
    for _, r in service_df.iterrows():
        act = r['activity']
        fam = r['chosen_family']
        params = r['chosen_params'] if not pd.isna(r['chosen_params']) else None
        if act in bundle.get('activities', {}):
            bundle['activities'][act]['service'] = {'family': fam, 'params': params}
    # update resources waiting if present
    for _, r in waiting_df.iterrows():
        res = r['resource']
        fam = r['chosen_family']
        params = r['chosen_params'] if not pd.isna(r['chosen_params']) else None
        # find activities referencing this resource — Prosit bundle earlier stored waiting under activities; we attach resource-level waiting under bundle['resources_waiting'] for clarity
        bundle.setdefault('resources_waiting', {})
        bundle['resources_waiting'][res] = {'family': fam, 'params': params}
    # write back
    with open(os.path.join(paramset_dir, 'simulator_parameters_bundle_sampled.json'), 'w') as f:
        json.dump(bundle, f, indent=2)
    print('Wrote updated simulator bundle with sampled fits to', os.path.join(paramset_dir, 'simulator_parameters_bundle_sampled.json'))
else:
    print('No simulator bundle found to update in', paramset_dir)

print('Done')
