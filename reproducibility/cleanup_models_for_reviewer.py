from pathlib import Path
import json
import shutil

base = Path(r"c:\Users\Jule\Documents\Master\Masterthesis\trucksimulation\reproducibility")
models = base / "models"
jsons_dir = base / "jsons_for_inspection"
jsons_dir.mkdir(exist_ok=True)

required_models = {
    "ctb_inductive_miner.pnml",
    "params_baseline_rmg_max_concurrency_3.pkl",
    "params_t22_closed.pkl",
    "params_demand_plus_20pct.pkl",
    "params_discovered_rules_workload.pkl",
    "params_no_rules.pkl",
    "params_rules_only_revised.pkl",
}

# Remove all non-required artifacts from models/
for path in list(models.iterdir()):
    if path.name not in required_models:
        if path.is_file() or path.is_symlink():
            path.unlink()
        elif path.is_dir():
            for child in sorted(path.rglob("*"), reverse=True):
                if child.is_file() or child.is_symlink():
                    child.unlink()
                elif child.is_dir():
                    child.rmdir()
            path.rmdir()

# Move inspection JSONs out of models/
inspection_jsons = [
    "params_no_rules.json",
    "params_rules_only_revised.json",
    "params_rules_workload.json",
]
for name in inspection_jsons:
    src = models / name
    dst = jsons_dir / name
    if src.exists():
        if dst.exists():
            dst.unlink()
        shutil.copy2(src, dst)
        src.unlink()

# Keep manifest aligned to final package contents
manifest_path = base / "model_manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
required_manifest_entries = {
    *{f"models/{name}" for name in required_models},
    *{f"jsons_for_inspection/{name}" for name in inspection_jsons},
}
new_files = {}
for rel, meta in manifest.get("files", {}).items():
    if rel.startswith("models/") and rel in required_manifest_entries:
        new_files[rel] = meta
    elif rel.startswith("jsons_for_inspection/") and rel in required_manifest_entries:
        new_files[rel] = meta
    elif not rel.startswith("models/") and not rel.startswith("jsons_for_inspection/"):
        new_files[rel] = meta
manifest["files"] = new_files
manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

print("models:")
for path in sorted(models.iterdir(), key=lambda p: p.name):
    print(" -", path.name)
print("inspection jsons:")
for path in sorted(jsons_dir.iterdir(), key=lambda p: p.name):
    print(" -", path.name)
