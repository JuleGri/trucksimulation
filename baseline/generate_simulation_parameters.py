"""
Purpose:
- transform the empirical discovery outputs into a simulator-ready parameter bundle;
- fit candidate duration families for each activity and resource pattern;
- store the final parameter set in a reproducible discovery_params/<paramset>/ folder;
- create the artifacts needed for downstream Prosit experiments and scenario analysis.

Inputs:
- empirical activity-duration summaries;
- discovered resource capacities and overlap statistics;
- the newest processed event log used for the baseline discovery workflow.

Outputs:
- fitted service-time summaries per activity;
- resource-capacity recommendations;
- simulator_parameters_bundle.json and related paramset exports.

Methodological note:
- this script consolidates the discovered baseline into a reusable simulation specification for white-box scenario testing.
"""

import os
import json
from datetime import datetime
import pandas as pd
import numpy as np
from math import log, sqrt

# Helpers for moment-based parameter estimation

def fit_lognormal_from_moments(mean, std):
    if mean <= 0 or std <= 0:
        return None
    sigma = sqrt(np.log(1 + (std ** 2) / (mean ** 2)))
    mu = np.log(mean) - 0.5 * sigma ** 2
    return {'mu': float(mu), 'sigma': float(sigma)}

def fit_gamma_from_moments(mean, std):
    if mean <= 0 or std <= 0:
        return None
    var = std ** 2
    k = (mean ** 2) / var
    theta = var / mean
    return {'shape': float(k), 'scale': float(theta)}

# Locate project and discovery params
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
discovery_root = os.path.join(script_dir, 'discovery_params')
if not os.path.exists(discovery_root):
    os.makedirs(discovery_root)

# Route the paramset folder through the shared resolver so this script
# writes into the same folder as the discovery scripts (train80 by default,
# full only when the caller explicitly points at the full log). The
# --input / TRUCKSIM_EVENTLOG selection is honoured transparently.
from _eventlog_source import parse_input_arg, paramset_suffix_for, resolve_paramset_dir

_source_hint = parse_input_arg()
# When no --input is passed, prefer the training log if it exists so that
# suffix defaults align with the discovery scripts.
_source_hint = _source_hint or os.path.join(script_dir, '..', '..', 'data', 'processed', 'CTB', 's6_train.csv')
suffix = paramset_suffix_for(_source_hint)
param_dir = resolve_paramset_dir(script_dir, suffix)
paramset = os.path.basename(param_dir)
print('Using paramset folder:', param_dir)

# Helper to find cleaned empirical durations CSV (try several locations and recursive search)
possible_paths = [
    os.path.join(param_dir, 'empirical_activity_durations_cleaned.csv'),
    os.path.join(param_dir, 'empirical_activity_durations.csv'),
    os.path.join(param_dir, 'empirical_activity_durations_cleaned.xlsx'),
    os.path.join(param_dir, 'archiv', 'empirical_activity_durations_cleaned.csv'),
    os.path.join(script_dir, 'empirical_activity_durations.csv'),
]
emp_path = None
for p in possible_paths:
    if os.path.exists(p):
        emp_path = p
        break

# If still not found, search recursively from script_dir and project_root for files matching empirical_activity_durations*.csv
if emp_path is None:
    def recursive_search(root, pattern_prefix):
        for root_dir, dirs, files in os.walk(root):
            for fname in files:
                if fname.lower().startswith(pattern_prefix) and fname.lower().endswith('.csv'):
                    return os.path.join(root_dir, fname)
        return None

    emp_path = recursive_search(script_dir, 'empirical_activity_durations')
    if emp_path is None:
        emp_path = recursive_search(project_root, 'empirical_activity_durations')

if emp_path is None:
    # As a last resort, list candidate CSVs to help debugging
    candidates = []
    for root_dir, dirs, files in os.walk(script_dir):
        for fname in files:
            if fname.lower().endswith('.csv'):
                candidates.append(os.path.join(root_dir, fname))
    raise FileNotFoundError('Could not find empirical activity durations CSV. Checked typical locations. Nearby CSV files (first 20): {}' .format(candidates[:20]))

print('Reading empirical durations from', emp_path)

# Read CSV robustly: try header=[0,1] then flatten
try:
    emp = pd.read_csv(emp_path, header=[0,1], index_col=0)
    # flatten
    emp.columns = ['_'.join(map(str, c)).strip() for c in emp.columns.values]
except Exception:
    emp = pd.read_csv(emp_path, index_col=0)
    emp.columns = [str(c).strip() for c in emp.columns]

# Ensure activity names are a column instead of index if necessary
if emp.index.name is None:
    emp = emp

# Function to extract mean and std for service and waiting from columns
import re

def find_stat_cols(cols, prefix):
    # find name patterns containing prefix and mean/std
    prefix = prefix.lower()
    mean_col = None
    std_col = None
    for c in cols:
        lc = c.lower()
        if prefix in lc and 'mean' in lc and mean_col is None:
            mean_col = c
        if prefix in lc and ('std' in lc or 'sigma' in lc) and std_col is None:
            std_col = c
    return mean_col, std_col

cols = emp.columns.tolist()
service_mean_col, service_std_col = find_stat_cols(cols, 'service')
waiting_mean_col, waiting_std_col = find_stat_cols(cols, 'waiting')

if service_mean_col is None or waiting_mean_col is None:
    # Try alternative names that appeared in original CSV variants
    # e.g., columns like 'service_time_min_mean' or 'mean'
    for c in cols:
        lc = c.lower()
        if 'service' in lc and 'mean' in lc:
            service_mean_col = service_mean_col or c
        if 'service' in lc and 'std' in lc:
            service_std_col = service_std_col or c
        if 'waiting' in lc and 'mean' in lc:
            waiting_mean_col = waiting_mean_col or c
        if 'waiting' in lc and 'std' in lc:
            waiting_std_col = waiting_std_col or c

print('Found columns:', service_mean_col, service_std_col, waiting_mean_col, waiting_std_col)

# Build distribution fits per activity
service_rows = []
waiting_rows = []

for activity, row in emp.iterrows():
    try:
        svc_mean = float(row[service_mean_col]) if service_mean_col in row.index else np.nan
    except Exception:
        svc_mean = np.nan
    try:
        svc_std = float(row[service_std_col]) if service_std_col in row.index else np.nan
    except Exception:
        svc_std = np.nan
    try:
        w_mean = float(row[waiting_mean_col]) if waiting_mean_col in row.index else np.nan
    except Exception:
        w_mean = np.nan
    try:
        w_std = float(row[waiting_std_col]) if waiting_std_col in row.index else np.nan
    except Exception:
        w_std = np.nan

    # Service fits
    svc_fit = None
    if not np.isnan(svc_mean) and not np.isnan(svc_std) and svc_mean>0 and svc_std>0:
        ln = fit_lognormal_from_moments(svc_mean, svc_std)
        gm = fit_gamma_from_moments(svc_mean, svc_std)
        svc_fit = {
            'activity': activity,
            'service_mean': svc_mean,
            'service_std': svc_std,
            'lognormal_params': ln,
            'gamma_params': gm,
            'chosen_family': 'lognormal' if ln is not None else ('gamma' if gm is not None else 'empirical'),
            'chosen_params': ln if ln is not None else gm if gm is not None else None,
            'source': 'moments'
        }
    else:
        svc_fit = {
            'activity': activity,
            'service_mean': svc_mean,
            'service_std': svc_std,
            'lognormal_params': None,
            'gamma_params': None,
            'chosen_family': 'empirical',
            'chosen_params': None,
            'source': 'insufficient_stats'
        }
    service_rows.append(svc_fit)

    # Waiting fits
    w_fit = None
    if not np.isnan(w_mean) and not np.isnan(w_std) and w_mean>0 and w_std>0:
        lnw = fit_lognormal_from_moments(w_mean, w_std)
        gmw = fit_gamma_from_moments(w_mean, w_std)
        w_fit = {
            'activity': activity,
            'waiting_mean': w_mean,
            'waiting_std': w_std,
            'lognormal_params': lnw,
            'gamma_params': gmw,
            'chosen_family': 'lognormal' if lnw is not None else ('gamma' if gmw is not None else 'empirical'),
            'chosen_params': lnw if lnw is not None else gmw if gmw is not None else None,
            'source': 'moments'
        }
    else:
        w_fit = {
            'activity': activity,
            'waiting_mean': w_mean,
            'waiting_std': w_std,
            'lognormal_params': None,
            'gamma_params': None,
            'chosen_family': 'empirical',
            'chosen_params': None,
            'source': 'insufficient_stats'
        }
    waiting_rows.append(w_fit)

# Save fits to CSVs
service_df = pd.DataFrame(service_rows)
waiting_df = pd.DataFrame(waiting_rows)
service_df.to_csv(os.path.join(param_dir, 'distribution_fits_service.csv'), index=False)
waiting_df.to_csv(os.path.join(param_dir, 'distribution_fits_waiting.csv'), index=False)
print('Saved distribution_fits_service.csv and distribution_fits_waiting.csv to', param_dir)

# Resource capacities: try to read resource_estimated_concurrency.csv from param_dir or archiv
res_paths = [
    os.path.join(param_dir, 'resource_estimated_concurrency.csv'),
    os.path.join(param_dir, 'archiv', 'resource_estimated_concurrency.csv'),
]
res_est_path = next((p for p in res_paths if os.path.exists(p)), None)

if res_est_path is not None:
    res_df = pd.read_csv(res_est_path)
    # Map heuristic pools
    def map_pool(rname):
        rn = str(rname).upper()
        if 'LL' in rn:
            return 'LL'
        if 'HO2' in rn or 'HO' in rn:
            return 'HO2'
        if 'VC' in rn:
            return 'VC'
        if 'GATEIN' in rn or 'GATE IN' in rn:
            return 'GateIn'
        if 'GATEOUT' in rn or 'GATE OUT' in rn:
            return 'GateOut'
        if 'RMG' in rn:
            return 'RMG'
        return 'OTHER'

    res_df['pool'] = res_df['resource'].apply(map_pool)
    agg = res_df.groupby('pool').agg({
        'median_concurrency': 'median',
        'p90_concurrency': 'median',
        'max_concurrency': 'max'
    }).reset_index()
    # cap and night handling: attempt to find night p90 by checking if 'time' column exists in raw file (skip otherwise)
    # For simplicity here, set night_p90 to 1 if pool observed at night else 0: try to read resource timestamps not available -> conservative choice
    rows = []
    for _, r in agg.iterrows():
        pool = r['pool']
        med = int(r['median_concurrency']) if not np.isnan(r['median_concurrency']) else 1
        p90 = int(r['p90_concurrency']) if not np.isnan(r['p90_concurrency']) else med
        mx = int(r['max_concurrency']) if not np.isnan(r['max_concurrency']) else p90
        chosen = max(1, p90)
        chosen = min(3, chosen)
        # conservative night policy: cap night to 1
        night_capacity = 1
        chosen = max(chosen, night_capacity)
        candidates = sorted(list({max(1, med), max(1, p90), max(1, mx)}))
        candidates = [min(3, int(x)) for x in candidates]
        rows.append({
            'resource_pool': pool,
            'median_concurrency': int(med),
            'p90_concurrency': int(p90),
            'max_concurrency': int(mx),
            'night_p90': int(night_capacity),
            'chosen_capacity_baseline': int(chosen),
            'capacity_candidates': str(candidates)
        })
    capacities_df = pd.DataFrame(rows)
else:
    # fallback default capacities (conservative)
    rows = [
        {'resource_pool':'LL','median_concurrency':1,'p90_concurrency':2,'max_concurrency':3,'night_p90':1,'chosen_capacity_baseline':2,'capacity_candidates':'[1,2,3]'},
        {'resource_pool':'HO2','median_concurrency':1,'p90_concurrency':2,'max_concurrency':3,'night_p90':1,'chosen_capacity_baseline':2,'capacity_candidates':'[1,2,3]'},
        {'resource_pool':'RMG','median_concurrency':1,'p90_concurrency':1,'max_concurrency':1,'night_p90':1,'chosen_capacity_baseline':1,'capacity_candidates':'[1]'},
    ]
    capacities_df = pd.DataFrame(rows)

capacities_df.to_csv(os.path.join(param_dir, 'discovered_resource_capacities.csv'), index=False)
print('Saved discovered_resource_capacities.csv to', param_dir)

# Create simulator bundle JSON
bundle = {
    'paramset': paramset,
    'eventlog_source': '../../data/processed/CTB/xes_files/s6_eventlog_target_rank_features.xes',
    'created_by': os.path.basename(__file__),
    'timestamp': datetime.now().isoformat(),
    'activities': {},
    'resources': {},
    'routing_table': None
}

# attach routing table if present
routing_paths = [
    os.path.join(param_dir, 'discovered_routing_probabilities.csv'),
    os.path.join(param_dir, 'archiv', 'discovered_routing_probabilities.csv'),
]
routing_path = next((p for p in routing_paths if os.path.exists(p)), None)
if routing_path is not None:
    bundle['routing_table'] = os.path.relpath(routing_path, start=param_dir)

# activities mapping: prefer chosen_family from fits
for _, r in service_df.iterrows():
    act = r['activity']
    svc_family = r['chosen_family']
    svc_params = r['chosen_params']
    wrow = waiting_df[waiting_df['activity'] == act]
    if len(wrow) > 0:
        wrow = wrow.iloc[0]
        wait_family = wrow['chosen_family']
        wait_params = wrow['chosen_params']
    else:
        wait_family = 'empirical'
        wait_params = None
    bundle['activities'][act] = {
        'service': {'family': svc_family, 'params': svc_params},
        'waiting': {'family': wait_family, 'params': wait_params}
    }

for _, r in capacities_df.iterrows():
    pool = r['resource_pool']
    bundle['resources'][pool] = {
        'capacity': int(r['chosen_capacity_baseline']),
        'capacity_candidates': eval(r['capacity_candidates']) if isinstance(r['capacity_candidates'], str) else r['capacity_candidates']
    }

bundle_path = os.path.join(param_dir, 'simulator_parameters_bundle.json')
with open(bundle_path, 'w') as f:
    json.dump(bundle, f, indent=2)

print('Saved simulator_parameters_bundle.json to', bundle_path)
print('\nSummary:')
print('- activities:', len(bundle['activities']))
print('- resources:', len(bundle['resources']))
print('- routing_table (relative to param dir):', bundle['routing_table'])
print('\nDone. You can now inspect files under', param_dir)
