# CTB ProSiT minimal reviewer handover

Open `CTB_ProSiT_minimal_loader.ipynb` from this folder and choose **Run All**.
The notebook uses ProSiT's documented loading pattern:

```python
params = SimulatorParameters(net, initial_marking, final_marking)
params.from_json(JSON_FILE)
simulator = SimulatorEngine(params)
```

`models/ctb_inductive_miner.pnml` is the shared Petri net. The three JSON files
are the final baseline, T22-pool reallocation, and +20% demand models.

## Why JSON is included

PNML provides only the process structure. JSON stores the learned ProSiT
parameters needed to inspect how the model routes cases, selects resources,
applies calendars/capacities, and samples arrival, service and waiting times.
Together they allow a reviewer to reconstruct and inspect each ProSiT model
using the upstream API.

The JSON files are intentionally **inspection files**, not a claim of bitwise
numerical replication. Standard ProSiT JSON omits cached stochastic `sampled`
arrays. CTB's empirical attribute tuples also contain missing values; these
are encoded as JSON `null` so `from_json()` can load the files. Exact reruns of
the archived thesis tables need the frozen binary bundles and controlled
runner retained in the main project.

## Google Drive

Upload this complete folder, then download it locally and open the notebook in
Jupyter. If using Google Colab, mount Google Drive, set `ROOT` in the second
cell to this folder's Drive path, and run all cells. The first cell uses the
same `prosit-pm==1.0.3` installation pattern as the ProSiT documentation.
