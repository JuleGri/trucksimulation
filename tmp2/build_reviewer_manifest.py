import hashlib
import json
from pathlib import Path

root = Path(__file__).parent / "reproducibility_revised"
old = json.loads((root / "model_manifest.json").read_text(encoding="utf-8"))


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def role(relative):
    if relative.startswith("models/"):
        return "frozen model, control-flow, or model-provenance artifact"
    if relative.startswith("expected_results/"):
        return "canonical table for exact saved-model scenario comparison"
    if relative.startswith("thesis_results/"):
        return "non-confidential per-seed or derived manuscript evidence"
    if relative.startswith("runtime/"):
        return "frozen scenario execution, metric, and contract code"
    if relative.endswith(".ipynb"):
        return "minimal reviewer-facing Run All notebook"
    return "reproduction package documentation or execution support"


files = {}
for path in sorted(root.rglob("*")):
    if not path.is_file():
        continue
    relative = path.relative_to(root).as_posix()
    if relative == "model_manifest.json" or relative.startswith("outputs/") or relative.startswith(".reviewer_env/") or "__pycache__" in relative:
        continue
    files[relative] = {"role": role(relative), "sha256": digest(path), "size_bytes": path.stat().st_size}

old["created"] = "2026-08-27"
old["scope"] = {
    "raw_log_included": False,
    "exact_saved_model_rerun": "workload-aware baseline plus two what-if scenarios",
    "frozen_evidence_reconstruction": "historical hold-out, three-state ablation, drift, structural repair, bottleneck, state robustness, and capacity pressure",
}
old["files"] = files
(root / "model_manifest.json").write_text(json.dumps(old, indent=2), encoding="utf-8")
print(f"manifested {len(files)} files")
