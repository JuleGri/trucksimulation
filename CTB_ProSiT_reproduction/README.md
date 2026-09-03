# CTB ProSiT reproduction

This folder contains the final cap-three `visit-only` baseline and the two
saved what-if models reported in the thesis: T22 closure and 20% higher
working-time arrival intensity.

Open either `CTB_ProSiT_Local.ipynb` or `CTB_ProSiT_Colab.ipynb` and select
**Run All**. The notebook installs the pinned requirements, loads the common
PNML, loads all three readable JSON exports through ProSiT, loads the exact PKL
objects, checks the intervention definitions, simulates 17,892 cases for each
model with matched seeds 42--51, and compares every fresh table with
`expected_results/`.

The JSON files follow ProSiT's documented save-and-load interface and expose
the fitted parameters for inspection. The PKL files are used for exact
numerical reproduction because they preserve the empirical sample arrays used
by the CTB calibration. Only load the PKL files supplied in this trusted
folder.

A successful full run ends with `FULL REPRODUCTION PASSED`. Fresh files are
written only to `outputs/full/`. The notebook reproduces the frozen simulations
but does not repeat discovery or event-level validation because the
confidential event log is not distributed.

Historical capacity and saturation scripts remain as supplementary diagnostics
and are not executed by the primary notebooks. They use an earlier
fully-contextualised bundle and must not be interpreted as the final
`visit-only` reference.
