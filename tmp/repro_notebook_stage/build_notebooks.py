import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def markdown(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip().splitlines(True),
    }


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip().splitlines(True),
    }


INTRO = """
# CTB ProSiT reproduction

This notebook loads the saved CTB Petri net and ProSiT parameter bundles and reproduces the numerical results reported in the thesis.

The confidential terminal event log is not included. The saved workload-aware baseline and the two what-if models are simulated again from the PKL bundles. Historical hold-out, state-ablation, drift, structural-repair, bottleneck, and capacity results are recalculated from the included per-seed or derived evidence. They cannot be regenerated from the confidential raw events.
"""


FILES_TEXT = """
## Saved model files

The package follows ProSiT's documented save/load approach:

- PNML stores the Petri-net control flow.
- JSON is the readable parameter export produced by `SimulatorParameters.to_json()`.
- PKL stores the exact calibrated Python object used for the thesis simulations.

The PKL is required for exact reproduction because ProSiT 1.0.3 does not restore the empirical sample arrays stored in the calibrated CTB rule leaves from JSON. The notebook demonstrates the JSON API and reports this difference. Only load the verified PKL files supplied in this folder.
"""


REVIEW_CODE = r'''
from pathlib import Path
import pickle

import pandas as pd
import pm4py
from prosit import SimulatorParameters, SimulatorEngine

import reviewer_runner as rr

integrity = rr.verify_package_files()
print(f"Verified frozen files: {len(integrity)}")
print(rr.package_versions().to_string(index=False))

net, initial_marking, final_marking = pm4py.read_pnml(
    "models/ctb_inductive_miner.pnml"
)
print(
    f"Petri net: {len(net.places)} places, "
    f"{len(net.transitions)} transitions, {len(net.arcs)} arcs"
)

with open("models/params_baseline_rmg_max_concurrency_3.pkl", "rb") as handle:
    baseline = pickle.load(handle)

engine = SimulatorEngine(baseline)
print(f"Loaded executable baseline: {type(engine).__name__}")

models = rr.load_models()
print("\nSaved model configurations")
print(rr.model_summary(models).to_string(index=False))

print("\nModel contract checks")
print(rr.assert_model_contracts(models).to_string(index=False))

json_report = rr.export_and_reload_official_json(
    baseline, Path("outputs/json_api_demo.json")
)
print("\nProSiT JSON export/import check")
print(pd.Series(json_report).to_string())
'''


EVIDENCE_CODE = r'''
from pathlib import Path
import json

import pandas as pd
import reviewer_runner as rr

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

historical_summary, historical_contrasts = rr.reconstruct_historical_ablation()
historical_summary.to_csv(output_dir / "historical_ablation_summary.csv", index=False)
historical_contrasts.to_csv(output_dir / "historical_ablation_contrasts.csv", index=False)

print("Historical three-state ablation")
headline = historical_summary[historical_summary["metric"].isin([
    "case_turnaround_emd_min",
    "case_turnaround_sim_mean",
    "case_turnaround_sim_p90",
    "yard_service_time_emd_frequency_weighted_min",
    "yard_activity_rate_l1_error",
    "gate_only_cases",
])]
print(headline.to_string(index=False))

print("\nPaired state contrasts")
print(historical_contrasts.to_string(index=False))

evidence = rr.load_claim_evidence()

temporal = evidence["temporal_transfer"]
print("\nTemporal transfer")
print(pd.Series({
    "train_mean_turnaround_min": temporal["turnaround"]["train_mean_min"],
    "test_mean_turnaround_min": temporal["turnaround"]["test_mean_min"],
    "mean_shift_min": temporal["turnaround"]["mean_shift"]["difference_test_minus_train"],
    "mean_shift_ci95_lo": temporal["turnaround"]["mean_shift"]["ci95_lo"],
    "mean_shift_ci95_hi": temporal["turnaround"]["mean_shift"]["ci95_hi"],
    "p90_shift_min": temporal["turnaround"]["p90_shift"]["difference_test_minus_train"],
    "yard_service_weighted_emd_min": temporal["yard_service_frequency_weighted_wasserstein_min"],
}).to_string())

repair = evidence["structural_repair"]
print("\nStructural repair")
repair_table = pd.DataFrame([
    {"model": "discovered", **repair["before"]["test"]},
    {"model": "gate_only_restricted", **repair["after"]["test"]},
])
print(repair_table[[
    "model", "fitness", "precision", "generalization", "simplicity"
]].to_string(index=False))

print("\nBottleneck ranking")
print(evidence["bottleneck_ranking"].head(10).to_string(index=False))

capacity = evidence["capacity_pressure"]
print("\nRMG capacity pressure")
print(pd.Series({
    "highest_pressure_block": capacity["highest_pressure_block"],
    "baseline_nominal_utilization": capacity["highest_baseline_nominal_utilization"],
    "demand_plus_20_nominal_utilization": capacity["highest_scenario_nominal_utilization"],
    "multiplier_to_mean_saturation": capacity["smallest_multiplier_to_mean_saturation"],
    "blocks_with_minutes_above_capacity": capacity["blocks_with_observed_minutes_above_capacity"],
}).to_string())

print("\nScenario state-ablation contrasts")
state_ablation = evidence["scenario_state_ablation"]
state_ablation = state_ablation[state_ablation["metric"].isin([
    "mean_turnaround_min",
    "mean_rmg_service_min",
    "mean_rmg_pre_service_min",
    "mean_rmg_receive_service_min",
    "mean_rmg_delivery_service_min",
])]
print(state_ablation.to_string(index=False))
'''


SCENARIO_CODE = r'''
import reviewer_runner as rr

RUN_FULL_SCENARIOS = True
mode = "full" if RUN_FULL_SCENARIOS else "smoke"

output_dir = rr.run_saved_models(mode=mode)
print(f"Fresh scenario outputs: {output_dir}")

if RUN_FULL_SCENARIOS:
    comparison = rr.compare_with_frozen_results(output_dir)
    comparison.to_csv(
        output_dir / "comparison_with_thesis_results.csv", index=False
    )
    print(comparison.to_string(index=False))
    assert comparison["values_match"].all()
    print("Full reproduction passed: all scenario tables match the thesis results.")
else:
    print("Smoke test passed. It verifies execution but not the thesis values.")
'''


def local_notebook():
    cells = [
        markdown(INTRO),
        markdown("""
## 1. Install the frozen environment

Open this notebook from the reproduction directory with a Python 3.11 kernel and choose **Run All**.
"""),
        code("%pip install -r requirements.txt --disable-pip-version-check -q"),
        markdown(FILES_TEXT),
        markdown("## 2. Verify and load the saved models"),
        code(REVIEW_CODE),
        markdown("""
## 3. Reconstruct the remaining thesis results

These tables are recomputed from the included ten-seed validation outputs and non-confidential derived evidence.
"""),
        code(EVIDENCE_CODE),
        markdown("""
## 4. Rerun the saved baseline and what-if models

The default run uses 10 matched seeds, 3 saved models, and 17,892 cases per model. It can take about 30 minutes. Changing `RUN_FULL_SCENARIOS` to `False` runs a short mechanics test only.
"""),
        code(SCENARIO_CODE),
        markdown("""
## Interpretation

The rerun establishes that the saved model and its two interventions are executable and reproducible. It does not establish physical causal effects at the terminal. The model does not contain explicit container locations, crane trajectories, physical transit, or queue states.
"""),
    ]
    return make_notebook(cells, colab=False)


def colab_notebook():
    setup = r'''
from google.colab import drive
drive.mount("/content/drive")

import os
import shutil
import subprocess
import sys
from pathlib import Path

if sys.version_info[:2] != (3, 11):
    raise RuntimeError(
        "This reproduction uses Python 3.11. In Colab select "
        "Runtime > Change runtime type > Runtime Version > 2025.07, "
        "then run the notebook again."
    )

# Change this line only if the uploaded folder has another name or location.
REPRO_DIR = Path("/content/drive/MyDrive/reproducibility")
if not (REPRO_DIR / "reviewer_runner.py").is_file():
    raise FileNotFoundError(f"Reproduction folder not found: {REPRO_DIR}")

os.chdir(REPRO_DIR)

VENV = Path("/content/ctb_prosit_env")
PYTHON = VENV / "bin" / "python"

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "virtualenv"],
    check=True,
)

if not PYTHON.is_file():
    subprocess.run(
        [sys.executable, "-m", "virtualenv", str(VENV)],
        check=True,
    )

subprocess.run(
    [str(PYTHON), "-m", "pip", "install", "-q", "-r", "requirements.txt"],
    check=True,
)

ENV = os.environ.copy()
ENV["MPLBACKEND"] = "Agg"
ENV["PYTHONUNBUFFERED"] = "1"

def run_in_reproduction_environment(source):
    subprocess.run(
        [str(PYTHON), "-c", source],
        cwd=REPRO_DIR,
        env=ENV,
        check=True,
    )

print("Reproduction environment:", PYTHON)
'''
    cells = [
        markdown(INTRO),
        markdown("""
## 1. Connect Google Drive and create an isolated environment

Upload this complete folder to `MyDrive/reproducibility`. The isolated environment prevents Colab's preinstalled NumPy and pandas versions from being used by the reproduction.
"""),
        code(setup),
        markdown(FILES_TEXT),
        markdown("## 2. Verify and load the saved models"),
        code(f"run_in_reproduction_environment({REVIEW_CODE!r})"),
        markdown("""
## 3. Reconstruct the remaining thesis results

These tables are recomputed from the included ten-seed validation outputs and non-confidential derived evidence.
"""),
        code(f"run_in_reproduction_environment({EVIDENCE_CODE!r})"),
        markdown("""
## 4. Rerun the saved baseline and what-if models

The default run uses 10 matched seeds, 3 saved models, and 17,892 cases per model. It can take about 30 minutes. For a short mechanics test, change `RUN_FULL_SCENARIOS = True` to `False` in the code string below.
"""),
        code(f"run_in_reproduction_environment({SCENARIO_CODE!r})"),
        markdown("""
## Interpretation

The rerun establishes that the saved model and its two interventions are executable and reproducible. It does not establish physical causal effects at the terminal. The model does not contain explicit container locations, crane trajectories, physical transit, or queue states.
"""),
    ]
    return make_notebook(cells, colab=True)


def make_notebook(cells, colab):
    metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11"},
    }
    if colab:
        metadata["colab"] = {
            "provenance": [],
            "runtime_attributes": {"runtime_version": "2025.07"},
        }
    return {
        "cells": cells,
        "metadata": metadata,
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


notebooks = {
    "CTB_local_prosit_load_and_run.ipynb": local_notebook(),
    "CTB_colab_prosit_load_and_run.ipynb": colab_notebook(),
}

for name, notebook in notebooks.items():
    (ROOT / name).write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False),
        encoding="utf-8",
    )

readme = """# CTB ProSiT reproduction

Upload or copy this directory as one folder. It contains two notebooks:

- `CTB_local_prosit_load_and_run.ipynb` for a local Python 3.11 Jupyter environment.
- `CTB_colab_prosit_load_and_run.ipynb` for Google Colab. Upload the folder to `MyDrive/reproducibility` and select the Colab runtime version stated in the notebook.

Choose **Run All**. Both notebooks verify the saved files, load the Petri net and ProSiT parameters, demonstrate the documented JSON interface, reconstruct the non-confidential thesis evidence, and rerun the saved workload-aware baseline and the two what-if models. The full scenario run uses 10 matched seeds and 17,892 cases for each of three models and can take about 30 minutes.

The raw CTB event log cannot be distributed. Results that require the real hold-out log are therefore recalculated from the included per-seed validation outputs. The saved baseline and what-if models are simulated again from the verified PKL files and compared with the frozen thesis tables.

PNML stores the control flow. JSON is included for readable inspection and the ProSiT `to_json()` / `from_json()` interface. The verified PKL files are the authoritative executable state because the ProSiT 1.0.3 JSON representation does not preserve all empirical sampled arrays used by the CTB calibration.

Generated files are written to `outputs/`.
"""
(ROOT / "README.md").write_text(readme, encoding="utf-8")

manifest_path = ROOT / "model_manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
manifest["files"].pop("CTB_minimal_prosit_load_and_run.ipynb", None)

for name in (*notebooks, "README.md", "reviewer_runner.py"):
    path = ROOT / name
    manifest["files"][name] = {
        "role": (
            "minimal reviewer-facing Run All notebook"
            if name.endswith(".ipynb")
            else "reproduction package documentation or execution support"
        ),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }

manifest_path.write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print("Created:")
for name in notebooks:
    print(" ", name)
