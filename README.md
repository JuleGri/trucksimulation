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

## Demand-intensity experiments

`calendar_preserving_experiment.py` evaluates demand scaling while retaining
the ProSiT arrival calendar. It divides working-minute inter-arrivals by the
demand factor and admits `round(base_cases * factor)` cases, keeping the
expected working-time horizon stable. It audits Sunday/closed-slot arrivals,
case counts, structural contracts, and completion. For example:

```powershell
python calendar_preserving_experiment.py --base-cases 2000 --n-seeds 10 `
  --output-dir experimental_results/calendar_preserving_10seeds_base2000_final
```

The existing `CTB_ProSiT_reproduction/saturation_experiment.py` is the
fixed-case arrival-horizon-compression experiment. Its ten-seed, 2,000-case
comparison is in `experimental_results/saturation_full/`. The paired
comparison is in `experimental_results/paired_comparison_10seeds_base2000/`.

## Clone an existing local `trucksimulation` working directory into this repository

If your project already exists locally (for example at `~/trucksimulation`), copy its files into this repository root, then commit and push:

```bash
cd /home/runner/work/trucksimulation/trucksimulation
rsync -av --exclude '.git' ~/trucksimulation/ .
git add .
git commit -m "Import local trucksimulation working directory"
```
