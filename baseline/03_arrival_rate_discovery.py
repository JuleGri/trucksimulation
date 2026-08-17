"""
Purpose:
- benchmark a pooled arrival-rate assumption against a time-aware arrival model;
- quantify whether day/night structure materially improves predictive accuracy;
- provide the empirical evidence used to justify data-aware arrival discovery.

Inputs:
- the processed CTB event log in data/processed/CTB/;
- case-level arrival timestamps derived from the event log.

Outputs:
- arrival-rate benchmark CSVs under scenarios/baseline/discovery_params/<paramset>/;
- a day/night summary file showing the observed temporal differences in arrival behavior.

Methodological note:
- the pooled baseline represents a single global arrival rate;
- the time-aware model conditions on hour, weekday, and weekend indicators to approximate temporal dependence.
"""

import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from _eventlog_source import add_input_arg, paramset_suffix_for, resolve_event_csv


def build_case_arrivals(df):
    ts_col = 'enabled:timestamp' if 'enabled:timestamp' in df.columns else 'start:timestamp'
    df = df[["case:concept:name", ts_col]].dropna().copy()
    df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce')
    df = df.dropna(subset=[ts_col])
    case_arrivals = (
        df.groupby('case:concept:name', as_index=False)[ts_col]
        .min()
        .rename(columns={ts_col: 'arrival_ts'})
        .sort_values('arrival_ts')
        .reset_index(drop=True)
    )
    case_arrivals['arrival_hour'] = case_arrivals['arrival_ts'].dt.hour
    case_arrivals['weekday'] = case_arrivals['arrival_ts'].dt.dayofweek
    case_arrivals['is_weekend'] = case_arrivals['weekday'].ge(5).astype(int)
    case_arrivals['inter_arrival_min'] = case_arrivals['arrival_ts'].diff().dt.total_seconds().div(60.0)
    return case_arrivals


def summarize_day_night(case_arrivals):
    case_arrivals = case_arrivals.copy()
    case_arrivals['is_night'] = ((case_arrivals['arrival_hour'] < 6) | (case_arrivals['arrival_hour'] >= 22)).astype(int)
    day = case_arrivals[~case_arrivals['is_night'].astype(bool)]
    night = case_arrivals[case_arrivals['is_night'].astype(bool)]
    summary = []
    for name, subset in [('all', case_arrivals), ('day', day), ('night', night)]:
        if subset.empty:
            continue
        mean_iat = subset['inter_arrival_min'].dropna().mean()
        rate_per_hour = 60.0 / mean_iat if mean_iat > 0 else np.nan
        summary.append({
            'period': name,
            'mean_inter_arrival_min': float(mean_iat),
            'mean_arrivals_per_hour': float(rate_per_hour),
            'n_cases': int(len(subset)),
        })
    return pd.DataFrame(summary)


def train_context_tree(bucket_df):
    X = bucket_df[['hour', 'day_of_week', 'is_weekend']]
    y = bucket_df['arrivals']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

    best_model = None
    best_metrics = None
    for max_depth in [2, 3, 5, 8, None]:
        for min_leaf in [5, 10, 20, 50]:
            model = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_leaf, random_state=42)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            metrics = {
                'mae': mean_absolute_error(y_test, pred),
                'rmse': np.sqrt(mean_squared_error(y_test, pred)),
                'r2': r2_score(y_test, pred),
                'max_depth': max_depth,
                'min_samples_leaf': min_leaf,
            }
            if best_metrics is None or metrics['rmse'] < best_metrics['rmse']:
                best_metrics = metrics
                best_model = model
    return best_model, best_metrics, X_test, y_test


def main():
    parser = argparse.ArgumentParser(description='Arrival-rate discovery on the held-out training log.')
    add_input_arg(parser)
    args = parser.parse_args()

    event_csv = resolve_event_csv(args.input_csv)
    df = pd.read_csv(event_csv)
    case_arrivals = build_case_arrivals(df)
    summary = summarize_day_night(case_arrivals)

    # Build 30-minute arrival counts; this is the most interpretable signal for a time-aware model.
    arrivals_by_slot = (
        case_arrivals.set_index('arrival_ts')
        .resample('30min')
        .size()
        .rename('arrivals')
        .to_frame()
    )
    arrivals_by_slot['hour'] = arrivals_by_slot.index.hour
    arrivals_by_slot['day_of_week'] = arrivals_by_slot.index.dayofweek
    arrivals_by_slot['is_weekend'] = arrivals_by_slot['day_of_week'].ge(5).astype(int)

    baseline_mean = float(arrivals_by_slot['arrivals'].mean())
    X = arrivals_by_slot[['hour', 'day_of_week', 'is_weekend']]
    y = arrivals_by_slot['arrivals']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    baseline_pred = np.repeat(baseline_mean, len(y_test))

    tree_model, tree_metrics, _, _ = train_context_tree(arrivals_by_slot)
    tree_pred = tree_model.predict(X_test)

    model_rows = [
        {
            'model': 'pooled_exponential_rate',
            'mean_arrivals_per_slot': baseline_mean,
            'mae': mean_absolute_error(y_test, baseline_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, baseline_pred)),
            'r2': r2_score(y_test, baseline_pred),
            'notes': 'single arrival rate across all periods',
            'source_eventlog': os.path.basename(event_csv),
        },
        {
            'model': 'time_aware_decision_tree',
            'mean_arrivals_per_slot': float(np.mean(y_test)),
            'mae': mean_absolute_error(y_test, tree_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, tree_pred)),
            'r2': r2_score(y_test, tree_pred),
            'notes': 'tree conditioned on hour, weekday, weekend indicator',
            'tree_max_depth': tree_model.get_depth(),
            'tree_min_samples_leaf': tree_model.min_samples_leaf,
            'source_eventlog': os.path.basename(event_csv),
        },
    ]
    model_df = pd.DataFrame(model_rows)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    from _eventlog_source import resolve_paramset_dir
    param_dir = resolve_paramset_dir(script_dir, paramset_suffix_for(event_csv))
    summary_path = os.path.join(param_dir, 'arrival_rate_analysis.csv')
    model_df.to_csv(summary_path, index=False)
    summary.to_csv(os.path.join(param_dir, 'arrival_rate_by_period.csv'), index=False)

    print('Arrival-log analysis complete.')
    print(f'Wrote:{summary_path}')
    print(model_df.to_string(index=False))
    print('\nDay/night summary:')
    print(summary.to_string(index=False))


if __name__ == '__main__':
    main()
