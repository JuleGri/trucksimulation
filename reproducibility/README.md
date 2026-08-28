# CTB ProSiT reproduction

Upload or copy this directory as one folder. It contains two notebooks:

- `CTB_local_prosit_load_and_run.ipynb` for a local Python 3.11 Jupyter environment.
- `CTB_colab_prosit_load_and_run.ipynb` for Google Colab. Upload the folder to `MyDrive/reproducibility` and select the Colab runtime version stated in the notebook.

Choose **Run All**. Both notebooks verify the saved files, load the Petri net and ProSiT parameters, demonstrate the documented JSON interface, reconstruct the non-confidential thesis evidence, and rerun the saved workload-aware baseline and the two what-if models. The full scenario run uses 10 matched seeds and 17,892 cases for each of three models and can take about 30 minutes.

The raw CTB event log cannot be distributed. Results that require the real hold-out log are therefore recalculated from the included per-seed validation outputs. The saved baseline and what-if models are simulated again from the verified PKL files and compared with the frozen thesis tables.

PNML stores the control flow. JSON is included for readable inspection and the ProSiT `to_json()` / `from_json()` interface. The verified PKL files are the authoritative executable state because the ProSiT 1.0.3 JSON representation does not preserve all empirical sampled arrays used by the CTB calibration.

Generated files are written to `outputs/`.
