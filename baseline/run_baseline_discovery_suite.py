"""
Purpose:
- execute the core baseline discovery pipeline as a single reproducible workflow;
- mirror the methodological sequence used in the thesis for arrival, resource, and data-aware discovery;
- generate the evidence files used for the baseline results and robustness discussion.

Inputs:
- the current event log in data/processed/CTB/;
- the active baseline analysis scripts in scenarios/baseline/;
- the current discovery_params directory for parameter-set management.

Outputs:
- arrival benchmark CSVs;
- resource-capacity discovery CSVs;
- data-aware model-evaluation summaries;
- robustness-analysis outputs saved under discovery_params/<paramset>/.
"""

import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description='Run the baseline discovery suite on the held-out training log.')
    parser.add_argument('--input', dest='input_csv', default=None,
                        help='Event log CSV forwarded to every discovery script. Defaults to s6_train.csv when present.')
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    scripts = [
        '03_arrival_rate_discovery.py',
        '04_resource_concurrency_discovery.py',
        '05_data_aware_discovery_evaluation.py',
        '06_robustness_analysis.py',
    ]
    forwarded = ['--input', args.input_csv] if args.input_csv else []
    for script in scripts:
        path = os.path.join(base_dir, script)
        print(f'Running {script} ...')
        subprocess.run([sys.executable, path, *forwarded], cwd=base_dir, check=True)
    print('\nAll baseline discovery analyses completed successfully.')


if __name__ == '__main__':
    main()
