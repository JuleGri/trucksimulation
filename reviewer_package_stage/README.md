# CTB ProSiT reproduction

This folder contains the three saved simulation models used in the thesis:

- the calibrated baseline;
- the T22-closure scenario; and
- the 20% demand-increase scenario.

The demand-saturation diagnostic does not need another model file. Its six
arrival-intensity levels are derived in memory from the saved baseline so that
the Petri net, rules, resources, and capacities remain fixed.

Open either `CTB_ProSiT_Colab.ipynb` or `CTB_ProSiT_Local.ipynb` and select
**Run All**. The first run simulates 17,892 cases for each saved model with the
ten matched seeds 42--51. The second run uses the same case count and seeds at
six demand levels. Allow approximately 45--75 minutes, depending on the
computer or Colab runtime.

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
5. runs 10 matched seeds for all three saved models;
6. reproduces the paired policy-scenario tables;
7. derives and runs the six-level demand-saturation diagnostic; and
8. compares every fresh table with `expected_results/`.

Successful completion ends with two comparison tables whose
`values_match` entries are all `True` and whose maximum absolute numerical
difference is `0.0`. New files are written only to `outputs/full/` and
`outputs/saturation_full/` (or the corresponding smoke folders when the short
test is selected).

## Scope

The notebook reproduces the final simulations from frozen models. It does not
repeat discovery or event-level validation because the confidential CTB event
log is not distributed.

## Expected comparison

The T22 and 20% demand policy runs do not resolve an overall mean-turnaround
change. The demand ladder explains the latter result. A realised elapsed-rate
ratio of 1.159 remains unresolved; at 1.378 the first small response appears,
and at 2.367 mean turnaround rises by 4.644 minutes while mean RMG pre-service
rises by 5.878 minutes. The high-load service-time change remains unresolved,
so the response is carried by model-derived pre-service delay. This is a
result about the saved simulation model, not a physical forecast for CTB.
