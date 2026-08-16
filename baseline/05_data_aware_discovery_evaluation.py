"""
Purpose:
- evaluate whether contextual features improve over a pooled baseline for activity duration discovery;
- compare a global mean model against a time/context-aware decision tree per activity;
- quantify feature importance for the data-aware model specification.

Inputs:
- the processed CTB event log;
- activity-level service times and contextual attributes such as demand, utilization, hour, and weekday.

Outputs:
- data_aware_model_summary.csv for each activity and evaluated tree configuration;
- tree_feature_importance.csv capturing the most relevant context variables;
- a structured basis for discussing where data-aware simulation is actually useful.

Methodological note:
- this script is intentionally conservative: it compares contextual models against pooled baselines and does not claim universal superiority.
"""

import os
from itertools import product

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


def resolve_event_csv():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(script_dir, '..', '..', 'data', 'processed', 'CTB', 's6_eventlog_target_rank_features.csv'),
        os.path.join(script_dir, '..', '..', 'data', 'processed', 'CTB', 's6_eventlog_target_rank_features_v1.csv'),
        os.path.join(script_dir, 's6_eventlog_target_rank_features.csv'),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError('Could not find the target event log CSV.')


def build_service_time(df):

    out = df.copy()

    out["enabled:timestamp"] = pd.to_datetime(
        out["enabled:timestamp"],
        errors="coerce"
    )

    out["start:timestamp"] = pd.to_datetime(
        out["start:timestamp"],
        errors="coerce"
    )

    out["time:timestamp"] = pd.to_datetime(
        out["time:timestamp"],
        errors="coerce"
    )

    print(
        out[
            [
                "enabled:timestamp",
                "start:timestamp",
                "time:timestamp"
            ]
        ].dtypes
    )

    out["waiting_time_min"] = (
        out["start:timestamp"]
        -
        out["enabled:timestamp"]
    ).dt.total_seconds() / 60

    out["service_time_min"] = (
        out["time:timestamp"]
        -
        out["start:timestamp"]
    ).dt.total_seconds() / 60

    for col in ['enabled:timestamp', 'start:timestamp', 'time:timestamp']:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors='coerce')
    if 'time:timestamp' not in out.columns or 'start:timestamp' not in out.columns:
        raise KeyError('The event log must contain start:timestamp and time:timestamp for service-time discovery.')
    out['service_time_min'] = (out['time:timestamp'] - out['start:timestamp']).dt.total_seconds().div(60.0)
    if 'enabled:timestamp' in out.columns:
        out['waiting_time_min'] = (out['start:timestamp'] - out['enabled:timestamp']).dt.total_seconds().div(60.0)
        out['waiting_time_min'] = out['waiting_time_min'].clip(lower=0)
    else:
        out['waiting_time_min'] = np.nan
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


def evaluate_activity_tree(df, activity_name, hyperparams):
    subset = df[df['concept:name'].eq(activity_name)].copy()
    if subset.empty:
        return []
    feature_cols = ['target_demand', 'target_utilization', 'target_rank', 'visit_complexity', 'n_receives', 'n_deliveries', 'hour', 'weekday', 'is_weekend']
    X = subset[feature_cols]
    y = subset['service_time_min']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    baseline_pred = np.repeat(float(y_train.mean()), len(y_test))

    rows = []
    for max_depth, min_samples_leaf in hyperparams:
        depth_value = int(max_depth) if max_depth is not None else None
        model = DecisionTreeRegressor(max_depth=depth_value, min_samples_leaf=min_samples_leaf, random_state=42)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        rows.append({
            'activity': activity_name,
            'max_depth': depth_value,
            'min_samples_leaf': int(min_samples_leaf),
            'baseline_mae': mean_absolute_error(y_test, baseline_pred),
            'baseline_rmse': np.sqrt(mean_squared_error(y_test, baseline_pred)),
            'tree_mae': mean_absolute_error(y_test, pred),
            'tree_rmse': np.sqrt(mean_squared_error(y_test, pred)),
            'tree_r2': r2_score(y_test, pred),
            'n_test': int(len(y_test)),
            'n_train': int(len(y_train)),
        })
    return rows


def main():
    event_csv = resolve_event_csv()
    df = pd.read_csv(event_csv)
    df = build_service_time(df)
    df = make_feature_frame(df)

    hyperparams = list(product([2, 3, 5, 8, None], [5, 10, 20, 50]))
    rows = []
    for activity in sorted(df['concept:name'].dropna().unique()):
        rows.extend(evaluate_activity_tree(df, activity, hyperparams))

    summary = pd.DataFrame(rows)
    # Identify the best tree configuration per activity, and then aggregate feature importance for explanation.
    best_rows = []
    for activity, act_df in summary.groupby('activity'):
        best = act_df.sort_values(['tree_rmse', 'tree_mae']).iloc[0].to_dict()
        best_rows.append(best)
    best_df = pd.DataFrame(best_rows)

    importance_rows = []
    for activity in sorted(df['concept:name'].dropna().unique()):
        subset = df[df['concept:name'].eq(activity)].copy()
        if subset.empty:
            continue
        feature_cols = ['target_demand', 'target_utilization', 'target_rank', 'visit_complexity', 'n_receives', 'n_deliveries', 'hour', 'weekday', 'is_weekend']
        X = subset[feature_cols]
        y = subset['service_time_min']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
        best_params = best_df[best_df['activity'] == activity].iloc[0]
        max_depth_value = best_params['max_depth']
        if pd.isna(max_depth_value):
            max_depth_value = None
        else:
            max_depth_value = int(max_depth_value)
        model = DecisionTreeRegressor(max_depth=max_depth_value, min_samples_leaf=int(best_params['min_samples_leaf']), random_state=42)
        model.fit(X_train, y_train)
        importances = model.feature_importances_
        for feat, importance in zip(feature_cols, importances):
            importance_rows.append({
                'activity': activity,
                'feature': feat,
                'importance': float(importance),
            })

    script_dir = os.path.dirname(os.path.abspath(__file__))
    param_root = os.path.join(script_dir, 'discovery_params')
    summary_path = os.path.join(script_dir, 'data_aware_model_summary.csv')
    importance_path = os.path.join(script_dir, 'tree_feature_importance.csv')
    if os.path.exists(param_root):
        param_dirs = [p for p in os.listdir(param_root) if os.path.isdir(os.path.join(param_root, p))]
        if param_dirs:
            param_dir = os.path.join(param_root, sorted(param_dirs)[-1])
            summary_path = os.path.join(param_dir, 'data_aware_model_summary.csv')
            importance_path = os.path.join(param_dir, 'tree_feature_importance.csv')

    summary.to_csv(summary_path, index=False)
    pd.DataFrame(importance_rows).to_csv(importance_path, index=False)

    print('Data-aware discovery evaluation complete.')
    print(f'Wrote: {summary_path}')
    print(f'Wrote: {importance_path}')
    print(best_df[['activity', 'max_depth', 'min_samples_leaf', 'tree_mae', 'tree_rmse', 'tree_r2']].sort_values('tree_rmse').to_string(index=False))


if __name__ == '__main__':
    main()
