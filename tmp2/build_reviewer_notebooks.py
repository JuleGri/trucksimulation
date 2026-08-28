import json
from pathlib import Path

ROOT = Path(__file__).parent / "reproducibility_revised"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(True)}


def notebook(colab=False):
    cells = [md("""# CTB ProSiT — minimal reproduction

This notebook loads the saved Petri net and ProSiT parameters, verifies every file, runs the published what-if models, and reconstructs the numerical thesis claims from the shipped per-seed evidence.

The confidential terminal event log is not included. Therefore the historical hold-out comparison is **arithmetically reproduced from frozen per-seed validation outputs**, while the saved-model what-if experiment is rerun exactly. This boundary is deliberate and auditable.
""")]
    if colab:
        cells.append(code("""from google.colab import drive
drive.mount('/content/drive')

# Upload this complete directory to MyDrive as `reproducibility`.
%cd /content/drive/MyDrive/reproducibility
"""))
    cells.extend([
        md("""## 1. Install the frozen environment

Run All is sufficient. Python 3.11 or 3.12 is recommended.
"""),
        code("%pip install -r requirements.txt --disable-pip-version-check -q\n"),
        md("## 2. Verify and load the saved models\n"),
        code("""import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, ".")
import reviewer_runner as rr

integrity = rr.verify_package_files()
print(f"Integrity PASS: {len(integrity)} frozen files verified")
models = rr.load_analysis_models()
display(rr.model_summary(models))
"""),
        md("""## Why PNML, JSON, and pickle are all provided

- **PNML** contains the portable Petri-net control flow.
- **JSON** makes the parameters human-readable and follows ProSiT's documented `to_json()` / `from_json()` interface.
- **Verified pickle** is authoritative for exact execution. In ProSiT 1.0.3, JSON does not preserve the empirical samples stored in decision-rule leaves and the CTB tuple keys containing `nan` cannot be read back reliably. JSON alone would lose runtime state.
"""),
        code("""scenario_models = rr.load_models()
json_report = rr.export_and_reload_official_json(
    scenario_models["baseline"], Path("outputs/json_api_demo.json")
)
display(pd.Series(json_report))
"""),
        md("## 3. Verify the original and what-if model contracts\n"),
        code("""display(rr.assert_model_contracts(scenario_models))
display(rr.parameter_component_inventory(scenario_models["baseline"]).head(20))
"""),
        md("""## 4. Reconstruct the historical three-state ablation

This recomputes means, 95% t-intervals, and paired seed contrasts from the ten shipped replication rows for `no_rules`, `rules_only`, and `rules_workload`.
"""),
        code("""historical_summary, historical_contrasts = rr.reconstruct_historical_ablation()
display(historical_summary)
display(historical_contrasts)
"""),
        md("## 5. Inspect the remaining claim evidence\n"),
        code("""evidence = rr.load_claim_evidence()
print("Available evidence:", ", ".join(evidence))
display(evidence["bottleneck_ranking"].head(8))
display(evidence["capacity_by_block"].head(8))
display(evidence["scenario_state_ablation"].query(
    "metric in ['mean_turnaround_min', 'mean_rmg_service_min', 'mean_rmg_pre_service_min']"
))
"""),
        md("""## 6. Exact saved-model scenario reproduction

The default reruns 10 matched seeds × 3 saved models × 17,892 cases and compares every regenerated table with the frozen thesis output. It took about 30 minutes on the author's computer. Set `RUN_FULL_SCENARIOS = False` only for a short mechanics check; a smoke run is not a numerical reproduction.
"""),
        code("""RUN_FULL_SCENARIOS = True

mode = "full" if RUN_FULL_SCENARIOS else "smoke"
output_dir = rr.run_saved_models(mode=mode)
print(f"Fresh outputs: {output_dir}")

if RUN_FULL_SCENARIOS:
    comparison = rr.compare_with_frozen_results(output_dir)
    display(comparison)
    assert comparison["values_match"].all()
    print("FULL REPRODUCTION PASS — all scenario tables match.")
else:
    print("SMOKE PASS — execution works; frozen thesis numbers were not rerun.")
"""),
        md("""## Interpretation boundary

The saved model is reproducible and its interventions are executable. This does not make the scenario effects causal estimates for the physical terminal. The current model lacks explicit container locations, crane trajectories, physical transit, and queue states; the notebook exposes those limits alongside the successful checks.
"""),
    ])
    return {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.11"}}, "nbformat": 4, "nbformat_minor": 5}


for filename, is_colab in (("CTB_minimal_prosit_load_and_run.ipynb", False), ("CTB_colab_prosit_load_and_run.ipynb", True)):
    with (ROOT / filename).open("w", encoding="utf-8") as handle:
        json.dump(notebook(is_colab), handle, indent=1, ensure_ascii=False)
