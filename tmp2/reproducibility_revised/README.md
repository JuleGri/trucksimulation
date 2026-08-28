# CTB ProSiT reproduction package

This directory can be uploaded to Google Drive on its own. It contains the saved Petri net, ProSiT parameter bundles, exact what-if models, frozen per-seed validation evidence, and a minimal notebook.

## Run it

- Local Jupyter: open `CTB_minimal_prosit_load_and_run.ipynb` and choose **Run All**.
- Google Colab: upload this complete directory to `MyDrive/reproducibility`, open `CTB_colab_prosit_load_and_run.ipynb`, and choose **Run All**.

The notebook installs the pinned requirements, verifies SHA-256 hashes before unpickling, loads the models, demonstrates ProSiT's JSON interface, reconstructs the reported validation statistics, reruns the three workload-aware scenario models, and compares all regenerated scenario tables with the frozen thesis output. The complete rerun contains 10 matched seeds × 3 models × 17,892 cases and took about 30 minutes on the author's computer. Set `RUN_FULL_SCENARIOS = False` only for a mechanics-only smoke run.

Python 3.11 or 3.12 is recommended. The first run needs internet access to install packages.

## What is reproduced

The package supports two levels of reproduction, stated explicitly:

1. The saved workload-aware baseline, T22-removal model, and 20% demand model are rerun exactly. Fresh replication, contract, KPI, and paired-delta tables must match the published frozen tables.
2. Historical hold-out validation, state ablation, temporal transfer, structural repair, bottleneck, and capacity-pressure claims are recomputed or inspected from frozen non-confidential per-seed and derived evidence.

The raw CTB log cannot be distributed. Consequently, the second group is an exact arithmetic and provenance reproduction, not a fresh event-level rerun from confidential records. `THESIS_RESULT_MAP.md` maps each manuscript claim to its evidence.

## Why the JSON is needed—and why it is not sufficient

ProSiT documents `SimulatorParameters.to_json()` and `from_json()` as its portable save/load interface. JSON is useful because it is human-readable and makes the fitted components inspectable. PNML separately stores the portable Petri-net control flow.

For this CTB bundle, ProSiT 1.0.3 does not preserve each rule leaf's empirical `sampled` array in JSON, and its loader cannot reliably parse empirical attribute-tuple keys containing `nan`. These samples are used by the calibrated runtime, especially the arrival model. Therefore the package contains:

- PNML for portable control flow;
- JSON for inspection and API demonstration; and
- verified pickle files as the authoritative executable state.

Pickle may execute code during loading. Use only the files in this trusted package and keep the integrity check enabled.

## Model families

The package includes the three historical state configurations (`no_rules`, `rules_only`, and `rules_workload`), the precision-oriented gate-bypass repair, and two complete cap-three scenario families. The default exact rerun uses the workload-aware family reported in the main scenario table. The workload-blind family and its ten-seed results are included for the state-ablation robustness check.

The resource intervention is a statistical pool reallocation, not a physical block-closure model. The demand intervention modifies the arrival process but does not add container locations, crane trajectories, physical transit, or explicit queues. Executability therefore does not establish physical causal validity.

## Directory guide

- `models/`: PNML, readable JSON, and executable parameter pickles.
- `thesis_results/`: frozen per-seed and derived evidence for manuscript claims.
- `expected_results/`: canonical tables used by the exact scenario comparison.
- `runtime/`: frozen scenario metric and contract code.
- `reviewer_runner.py`: compact loading, checking, reconstruction, and rerun API.
- `model_manifest.json`: file hashes and experiment provenance.

Notebook outputs are written to `outputs/` and are intentionally not part of the package.
