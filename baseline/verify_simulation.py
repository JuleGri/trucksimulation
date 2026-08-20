"""
Purpose:
- verify that a valid baseline simulator bundle exists;
- summarize the discovered resources, activity families, and routing metadata;
- provide a lightweight template for running a Prosit simulation in a compatible environment.

Inputs:
- the most recent discovery_params/<paramset>/simulator_parameters_bundle.json;
- the baseline event log and project directory structure.

Outputs:
- a compact verification summary for the generated simulation bundle;
- a minimal Prosit execution template that can be adapted to the local environment.

Methodological note:
- this helper is intended for validation and reproducibility, not for generating new discoveries.
"""

import os
import json
from pprint import pprint

script_dir = os.path.dirname(os.path.abspath(__file__))
param_root = os.path.join(script_dir, 'discovery_params')
# pick the paramset used (first found)
paramset = None
if os.path.exists(param_root):
    candidates = [p for p in os.listdir(param_root) if os.path.isdir(os.path.join(param_root, p))]
    if len(candidates) > 0:
        paramset = candidates[0]

if paramset is None:
    raise SystemExit('No paramset found in discovery_params/. Run generate_simulation_parameters.py first.')

param_dir = os.path.join(param_root, paramset)
bundle_path = os.path.join(param_dir, 'simulator_parameters_bundle.json')
if not os.path.exists(bundle_path):
    # also try archiv location
    bundle_path = os.path.join(param_dir, 'archiv', 'simulator_parameters_bundle.json')
    if not os.path.exists(bundle_path):
        raise SystemExit('simulator_parameters_bundle.json not found in paramset folder. Run generate_simulation_parameters.py')

with open(bundle_path, 'r') as f:
    bundle = json.load(f)

print('Simulator bundle found at:', bundle_path)
print('\nBundle summary:')
print('Paramset:', bundle.get('paramset'))
print('Eventlog source:', bundle.get('eventlog_source'))
print('Created by:', bundle.get('created_by'))
print('Timestamp:', bundle.get('timestamp'))
print('\nResources:')
for r,info in bundle.get('resources', {}).items():
    print(f" - {r}: capacity={info.get('capacity')} candidates={info.get('capacity_candidates')}")

print('\nSample activities (first 10):')
for i,(act,vals) in enumerate(bundle.get('activities', {}).items()):
    if i>=10:
        break
    print(f" - {act}: service={vals['service']['family']} waiting={vals['waiting']['family']}")

print('\nRouting table (relative to param dir):', bundle.get('routing_table'))

print('\nVerification template:')
print('The following code sketch reuses the validated sequential ProSiT parameter bundle. Do not rediscover an unconstrained Petri net from completion-time order:')

print('''
# Template (do not run unless prosit and pm4py are available in your environment)
import pickle
from prosit import SimulatorEngine

# Reuse the fitted parameters; they contain the domain-constrained sequential
# control-flow net and the learned cross-case resource multitasking capacities.
with open('prosit_discovery_workload_sequential/prosit_params.pkl', 'rb') as fh:
    params = pickle.load(fh)

sim = SimulatorEngine(params)
sim_log = sim.apply(n_traces=1000)
print(sim_log.head())
''')

print('\nIf you want, I can try to run a short Prosit simulation here (in this environment). Reply to confirm and provide the kernel/conda environment to use. Otherwise run the template locally in your Prosit environment.')
