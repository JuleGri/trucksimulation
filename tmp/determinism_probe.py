from __future__ import annotations

import hashlib
import pickle
import sys
from datetime import datetime
from pathlib import Path


PROJECT = Path(r"C:\Users\Jule\Documents\Master\Masterthesis\trucksimulation")
MODEL = (
    PROJECT
    / "reproducibility"
    / "models"
    / "params_baseline_rmg_max_concurrency_3.pkl"
)
sys.path.insert(0, str(PROJECT))

from _prosit_ctb_calibration import simulate_ctb  # noqa: E402


with MODEL.open("rb") as handle:
    parameters = pickle.load(handle)

simulation = simulate_ctb(
    parameters,
    n_traces=250,
    t_start=datetime.fromisoformat("2026-04-20T18:17:00"),
    seed=42,
    timestamp_resolution="min",
)
payload = simulation.to_csv(index=False, lineterminator="\n").encode("utf-8")
print(hashlib.sha256(payload).hexdigest())
