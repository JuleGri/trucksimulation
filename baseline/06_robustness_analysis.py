"""
Purpose:
- evaluate the robustness of the arrival model under repeated chronological validation;
- assess sensitivity of the context-aware arrival tree to bucket size and tree hyperparameters;
- test whether duration models remain stable under time-based cross-validation;
- quantify how alternative capacity assumptions change the expected stress on pooled resources.

Inputs:
- the current event log used by the baseline discovery project;
- the capacity estimates already produced in the resource-discovery stage;
- the duration-feature set used in the data-aware evaluation.

Outputs:
- arrival_cv_summary.csv and arrival_hyperparameter_sensitivity.csv;
- duration_cv_summary.csv;
- capacity_sensitivity_summary.csv and a consolidated robustness_summary.csv.

Methodological note:
- this script closes the gap between model selection and a more formal robustness discussion in the thesis.
"""

import argparse
import os
from itertools import product

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.tree import DecisionTreeRegressor

from _eventlog_source import add_input_arg, paramset_suffix_for, resolve_event_csv


def resolve_discovery_dir(suffix: str = ''):
    """Resolve or create the paramset folder used to store robustness artefacts.

    Delegates to the shared resolver in ``_eventlog_source`` so the results
    end up in the same paramset directory as the corresponding discovery
    outputs (train80 or full).
    """
    from _eventlog_source import resolve_paramset_dir
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return resolve_paramset_dir(script_dir, suffix)


def build_arrival_slots(df):
    ts_col = 'start:timestamp'
    df = df[['case:concept:name', ts_col]].dropna().copy()
    df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce')
    df = df.dropna(subset=[ts_col]).sort_values(ts_col)
    case_arrivals = (
        df.groupby('case:concept:name', as_index=False)[ts_col]
        .min()
        .rename(columns={ts_col: 'arrival_ts'})
        .sort_values('arrival_ts')
        .reset_index(drop=True)
    )
    case_arrivals['arrival_hour'] = case_arrivals['arrival_ts'].dt.hour
    case_arrivals['day_of_week'] = case_arrivals['arrival_ts'].dt.dayofweek
    case_arrivals['is_weekend'] = (case_arrivals['day_of_week'] >= 5).astype(int)
    return case_arrivals


def build_service_time_df(df):
    out = df.copy()
    for col in ['start:timestamp', 'time:timestamp']:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors='coerce')
    if 'start:timestamp' not in out.columns or 'time:timestamp' not in out.columns:
        raise KeyError('The event log must contain start:timestamp and time:timestamp.')
    out['service_time_min'] = (out['time:timestamp'] - out['start:timestamp']).dt.total_seconds().div(60.0)
    return out.dropna(subset=['service_time_min'])


def make_feature_frame(df):
    out = df.copy()
    out['hour'] = pd.to_datetime(out['start:timestamp'], errors='coerce').dt.hour
    out['weekday'] = pd.to_datetime(out['start:timestamp'], errors='coerce').dt.dayofweek
    out['is_weekend'] = out['weekday'].ge(5).astype(int)
    out['n_receives'] = out.get('n_receives', 0)
    out['n_deliveries'] = out.get('n_deliveries', 0)
    out['visit_complexity'] = out.get('visit_complexity', 0)
    out['target_rank'] = out.get('target_rank', 0)
    out['target_demand'] = out.get('target_demand', 0)
    out['target_utilization'] = out.get('target_utilization', 0)
    return out


def arrival_cv_summary(arrivals_by_slot):
    rows = []
    feature_cols = ['hour', 'day_of_week', 'is_weekend']
    tscv = TimeSeriesSplit(n_splits=5)
    for bucket_minutes in [15, 30, 60]:
        resampled = arrivals_by_slot.set_index('arrival_ts').resample(f'{bucket_minutes}min').size().rename('arrivals').to_frame()
        resampled['hour'] = resampled.index.hour
        resampled['day_of_week'] = resampled.index.dayofweek
        resampled['is_weekend'] = (resampled['day_of_week'] >= 5).astype(int)
        X = resampled[feature_cols]
        y = resampled['arrivals']
        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            pooled_pred = np.repeat(float(y_train.mean()), len(y_test))
            rows.append({
                'bucket_minutes': bucket_minutes,
                'fold': fold_idx,
                'model': 'pooled_mean',
                'mae': mean_absolute_error(y_test, pooled_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, pooled_pred)),
                'r2': r2_score(y_test, pooled_pred),
            })
            for max_depth, min_samples_leaf in [(3, 5), (5, 5), (5, 10), (8, 5), (None, 5)]:
                model = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=42)
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                rows.append({
                    'bucket_minutes': bucket_minutes,
                    'fold': fold_idx,
                    'model': 'time_aware_tree',
                    'max_depth': max_depth,
                    'min_samples_leaf': min_samples_leaf,
                    'mae': mean_absolute_error(y_test, pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, pred)),
                    'r2': r2_score(y_test, pred),
                })
    arrival_cv_df = pd.DataFrame(rows)
    agg = arrival_cv_df.groupby(['bucket_minutes', 'model', 'max_depth', 'min_samples_leaf'], dropna=False).agg(
        mae_mean=('mae', 'mean'),
        mae_std=('mae', 'std'),
        rmse_mean=('rmse', 'mean'),
        rmse_std=('rmse', 'std'),
        r2_mean=('r2', 'mean'),
        r2_std=('r2', 'std'),
        n_folds=('fold', 'count'),
    ).reset_index()
    return arrival_cv_df, agg


def evaluate_duration_cv(df):
    feature_cols = ['target_demand', 'target_utilization', 'target_rank', 'visit_complexity', 'n_receives', 'n_deliveries', 'hour', 'weekday', 'is_weekend']
    rows = []
    activities = sorted(df['concept:name'].dropna().unique())
    tscv = TimeSeriesSplit(n_splits=5)
    for activity in activities:
        subset = df[df['concept:name'].eq(activity)].copy().sort_values('start:timestamp')
        if subset.empty:
            continue
        X = subset[feature_cols]
        y = subset['service_time_min']
        for max_depth, min_samples_leaf in [(3, 5), (5, 5), (5, 10), (8, 5), (None, 5)]:
            fold_metrics = []
            for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                if len(y_train) < 2 or len(y_test) < 1:
                    continue
                baseline_pred = np.repeat(float(y_train.mean()), len(y_test))
                model = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=42)
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                fold_metrics.append({
                    'activity': activity,
                    'max_depth': max_depth,
                    'min_samples_leaf': min_samples_leaf,
                    'baseline_mae': mean_absolute_error(y_test, baseline_pred),
                    'baseline_rmse': np.sqrt(mean_squared_error(y_test, baseline_pred)),
                    'baseline_r2': r2_score(y_test, baseline_pred),
                    'tree_mae': mean_absolute_error(y_test, pred),
                    'tree_rmse': np.sqrt(mean_squared_error(y_test, pred)),
                    'tree_r2': r2_score(y_test, pred),
                    'fold': fold_idx,
                })
            if not fold_metrics:
                continue
            metric_df = pd.DataFrame(fold_metrics)
            rows.append({
                'activity': activity,
                'max_depth': max_depth,
                'min_samples_leaf': min_samples_leaf,
                'baseline_mae_mean': metric_df['baseline_mae'].mean(),
                'baseline_rmse_mean': metric_df['baseline_rmse'].mean(),
                'baseline_r2_mean': metric_df['baseline_r2'].mean(),
                'tree_mae_mean': metric_df['tree_mae'].mean(),
                'tree_rmse_mean': metric_df['tree_rmse'].mean(),
                'tree_r2_mean': metric_df['tree_r2'].mean(),
                'tree_mae_std': metric_df['tree_mae'].std(ddof=0),
                'tree_rmse_std': metric_df['tree_rmse'].std(ddof=0),
                'tree_r2_std': metric_df['tree_r2'].std(ddof=0),
                'n_folds': len(metric_df),
            })
    return pd.DataFrame(rows)


def resource_capacity_sensitivity_summary(suffix: str = ''):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    discovery_dir = resolve_discovery_dir(suffix)
    capacity_csv = os.path.join(discovery_dir, 'discovered_resource_capacities.csv')
    if os.path.exists(capacity_csv):
        cap_df = pd.read_csv(capacity_csv)
    else:
        cap_df = pd.DataFrame({
            'resource_pool': ['GateIn', 'GateOut', 'HO2', 'LL', 'OTHER'],
            'median_concurrency': [0, 0, 1, 1, 4],
            'p90_concurrency': [0, 0, 8, 6, 27],
            'max_concurrency': [10, 9, 15, 25, 63],
            'chosen_capacity_baseline': [1, 1, 3, 3, 3],
        })
    rows = []
    for _, row in cap_df.iterrows():
        pool = row['resource_pool']
        baseline_capacity = int(row.get('chosen_capacity_baseline', row.get('capacity', 1)))
        representative_overlap = float(np.clip(np.nanmean([row.get('median_concurrency', 0), row.get('p90_concurrency', 0)]), 0, None))
        candidate_caps = sorted({int(baseline_capacity), 1, 2, 3})
        for cap in candidate_caps:
            util_proxy = representative_overlap / max(cap, 1)
            rows.append({
                'resource_pool': pool,
                'baseline_capacity': baseline_capacity,
                'tested_capacity': cap,
                'representative_overlap': representative_overlap,
                'utilization_proxy': util_proxy,
                'relative_to_baseline': util_proxy / max(representative_overlap / max(baseline_capacity, 1), 1e-6),
            })
    return pd.DataFrame(rows)


def summarize_robustness(arrival_cv_df, duration_cv_df, capacity_sensitivity_df):
    summary = []
    if not arrival_cv_df.empty:
        pooled = arrival_cv_df[arrival_cv_df['model'].eq('pooled_mean')]
        tree = arrival_cv_df[arrival_cv_df['model'].eq('time_aware_tree')]
        summary.append({
            'analysis': 'arrival_model_robustness',
            'pooled_rmse_mean': pooled['rmse'].mean(),
            'tree_rmse_mean': tree['rmse'].mean(),
            'tree_rmse_improvement_pct': 100.0 * (pooled['rmse'].mean() - tree['rmse'].mean()) / pooled['rmse'].mean(),
            'pooled_r2_mean': pooled['r2'].mean(),
            'tree_r2_mean': tree['r2'].mean(),
        })
    if not duration_cv_df.empty:
        summary.append({
            'analysis': 'duration_model_robustness',
            'best_overall_activity_count': int(duration_cv_df['activity'].nunique()),
            'best_tree_r2_mean': float(duration_cv_df['tree_r2_mean'].mean()),
            'best_tree_r2_min': float(duration_cv_df['tree_r2_mean'].min()),
            'best_tree_r2_max': float(duration_cv_df['tree_r2_mean'].max()),
        })
    if not capacity_sensitivity_df.empty:
        summary.append({
            'analysis': 'capacity_sensitivity',
            'resource_count': int(capacity_sensitivity_df['resource_pool'].nunique()),
            'mean_utilization_proxy': float(capacity_sensitivity_df['utilization_proxy'].mean()),
            'max_relative_to_baseline': float(capacity_sensitivity_df['relative_to_baseline'].max()),
        })
    return pd.DataFrame(summary)


def main():
    parser = argparse.ArgumentParser(description='Robustness analysis on the held-out training log.')
    add_input_arg(parser)
    args = parser.parse_args()

    event_csv = resolve_event_csv(args.input_csv)
    suffix = paramset_suffix_for(event_csv)
    df = pd.read_csv(event_csv)
    case_arrivals = build_arrival_slots(df)

    arrival_cv_df, arrival_agg = arrival_cv_summary(case_arrivals)
    service_df = build_service_time_df(df)
    service_df = make_feature_frame(service_df)
    duration_cv_df = evaluate_duration_cv(service_df)
    capacity_sensitivity_df = resource_capacity_sensitivity_summary(suffix)
    robustness_summary_df = summarize_robustness(arrival_cv_df, duration_cv_df, capacity_sensitivity_df)

    discovery_dir = resolve_discovery_dir(suffix)
    arrival_cv_path = os.path.join(discovery_dir, 'arrival_cv_summary.csv')
    arrival_hyperparam_path = os.path.join(discovery_dir, 'arrival_hyperparameter_sensitivity.csv')
    duration_cv_path = os.path.join(discovery_dir, 'duration_cv_summary.csv')
    capacity_sensitivity_path = os.path.join(discovery_dir, 'capacity_sensitivity_summary.csv')
    robustness_path = os.path.join(discovery_dir, 'robustness_summary.csv')

    arrival_cv_df.to_csv(arrival_cv_path, index=False)
    arrival_agg.to_csv(arrival_hyperparam_path, index=False)
    duration_cv_df.to_csv(duration_cv_path, index=False)
    capacity_sensitivity_df.to_csv(capacity_sensitivity_path, index=False)
    robustness_summary_df.to_csv(robustness_path, index=False)

    print('Robustness analysis complete.')
    print(f'Wrote: {arrival_cv_path}')
    print(f'Wrote: {arrival_hyperparam_path}')
    print(f'Wrote: {duration_cv_path}')
    print(f'Wrote: {capacity_sensitivity_path}')
    print(f'Wrote: {robustness_path}')
    print('\nArrival robustness summary:')
    print(arrival_agg.sort_values(['rmse_mean', 'r2_mean']).to_string(index=False))
    print('\nDuration robustness summary:')
    print(duration_cv_df.sort_values('tree_rmse_mean').head(10).to_string(index=False))
    print('\nCapacity sensitivity summary:')
    print(capacity_sensitivity_df.head(20).to_string(index=False))


if __name__ == '__main__':
    main()
