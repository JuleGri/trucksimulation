"""Smoke test: verify ProSiT discovery works without enabled:timestamp in input."""
import sys
sys.path.insert(0, 'baseline')
import pandas as pd
import pm4py
from prosit import SimulatorParameters

# Load a tiny slice of the training log
df = pd.read_csv('data/processed/CTB/s6_train.csv', nrows=500)
print(f"Columns: {sorted(df.columns.tolist())}")
print(f"Has enabled:timestamp: {'enabled:timestamp' in df.columns}")
assert 'enabled:timestamp' not in df.columns, "enabled:timestamp should NOT be in input"

# Parse timestamps
df['start:timestamp'] = pd.to_datetime(df['start:timestamp'], errors='coerce')
df['time:timestamp'] = pd.to_datetime(df['time:timestamp'], errors='coerce')

# Convert to pm4py log
formatted = pm4py.format_dataframe(
    df.copy(),
    case_id='case:concept:name',
    activity_key='concept:name',
    timestamp_key='time:timestamp',
)
log = pm4py.convert_to_event_log(formatted)
print(f"pm4py log: {len(log)} traces")

# Discover Petri net
tree = pm4py.discover_process_tree_inductive(log)
net, im, fm = pm4py.convert_to_petri_net(tree)
print(f"Net: {len(net.places)} places, {len(net.transitions)} transitions")

# ProSiT discovery (no-rules for speed)
params = SimulatorParameters(net, im, fm)
params.discover_from_eventlog(
    log,
    max_depth_tree=0,
    attribute_mode='empirical',
    random_state=42,
    verbose=False,
)
print("ProSiT discovery completed successfully WITHOUT enabled:timestamp in input!")

# Quick simulation
import random
import numpy as np
from prosit import SimulatorEngine
engine = SimulatorEngine(params)
_orig = random.choice
def _safe(seq):
    if isinstance(seq, np.ndarray):
        return seq[np.random.randint(len(seq))]
    return _orig(seq)
random.choice = _safe
try:
    sim_log = engine.apply(n_traces=50)
finally:
    random.choice = _orig

print(f"Simulated {len(sim_log)} events, cols: {sorted(sim_log.columns.tolist())}")
print(f"Sim log has enabled:timestamp: {'enabled:timestamp' in sim_log.columns}")
print("\nSMOKE TEST PASSED - ProSiT works without enabled:timestamp in input")
