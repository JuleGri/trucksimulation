"""Chain: regenerate clean XES then run ProSiT discovery."""
import subprocess, sys, os

os.chdir(r"c:\Users\Jule\Documents\Master\Masterthesis\trucksimulation")
py = sys.executable

print("=" * 60)
print("STEP 1: Regenerate clean XES files")
print("=" * 60)
r = subprocess.run([py, "_regen_xes.py"], capture_output=False)
if r.returncode != 0:
    print(f"XES regen failed with code {r.returncode}")
    sys.exit(1)

print("\n" + "=" * 60)
print("STEP 2: Run ProSiT discovery from XES (Vinci approach)")
print("=" * 60)
r = subprocess.run([
    py, "baseline/07_run_prosit_discovery.py",
    "--xes", "data/processed/CTB/xes_files/s6_train.xes",
    "--test-xes", "data/processed/CTB/xes_files/s6_test.xes",
    "--skip-figures",
], capture_output=False)
sys.exit(r.returncode)
