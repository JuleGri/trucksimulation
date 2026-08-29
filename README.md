# trucksimulation
Masterthesis on data-aware process simulation of truck processes

## Frozen thesis evidence (29 August 2026)

The authoritative result set is indexed by
`validation/results/final_deterministic_20260829_provenance.json`. All central
tables use `_prosit_ctb_calibration.py`, ten matched seeds (42--51), 17,892
cases per run, the same start time, and minute timestamp resolution. The final
Inductive-Miner scenarios are stored in the matching `final_deterministic_*`
folders. Cap three is the expert-constrained reference; the uncapped folder is
a paired sensitivity, not an alternative final baseline.

## Clone an existing local `trucksimulation` working directory into this repository

If your project already exists locally (for example at `~/trucksimulation`), copy its files into this repository root, then commit and push:

```bash
cd /home/runner/work/trucksimulation/trucksimulation
rsync -av --exclude '.git' ~/trucksimulation/ .
git add .
git commit -m "Import local trucksimulation working directory"
```
