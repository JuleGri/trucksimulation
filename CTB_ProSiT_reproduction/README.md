# CTB ProSiT reproduction

This folder contains the three final simulation models used in the thesis:

- the calibrated baseline;
- the T22-closure scenario; and
- the 20% demand-increase scenario.

Open either `CTB_ProSiT_Colab.ipynb` or `CTB_ProSiT_Local.ipynb` and select
**Run All**. The full run simulates 17,892 cases for each model with the ten
matched seeds 42--51. It takes approximately 30 minutes.

## What is loaded

The PNML file contains the common Petri-net control flow. The JSON files are
readable ProSiT parameter exports and are loaded with the documented
`SimulatorParameters.from_json()` method. The PKL files preserve the exact
calibrated Python objects used for the thesis runs, including empirical sample
arrays that are not restored completely by ProSiT 1.0.3 JSON import. Therefore
JSON is used for inspection and PKL for exact numerical reproduction. Only
load the PKL files supplied in this folder.

## What the notebook does

1. installs the pinned environment;
2. loads the PNML and all three JSON exports;
3. loads the exact PKL models;
4. checks the two intervention definitions;
5. runs 10 matched seeds for all three models;
6. calculates the five reported result tables; and
7. compares every fresh table with `expected_results/`.

Successful completion ends with five `values_match = True` rows and a maximum
absolute numerical difference of `0.0`. New files are written only to
`outputs/full/` (or `outputs/smoke/` when the short test is selected).

## Scope

The notebook reproduces the final simulations from frozen models. It does not
repeat discovery or event-level validation because the confidential CTB event
log is not distributed.
