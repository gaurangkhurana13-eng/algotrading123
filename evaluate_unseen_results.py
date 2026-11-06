import subprocess, sys
from pathlib import Path

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
eval_script = ROOT_DIR / "strategies" / "strategy_evaluate_gate_v2.py"

folders = [
    ("V34 - 2026 Unseen 52 Midcaps", ROOT_DIR / "strategies" / "V34" / "unseen_2026"),
    ("V34 - 2025 Q4 (Oct-Dec 2025)", ROOT_DIR / "strategies" / "V34" / "unseen_2025_q4"),
    ("V34.2 - 2026 Unseen 52 Midcaps", ROOT_DIR / "strategies" / "V34.2" / "unseen_2026"),
    ("V34.2 - 2025 Q4 (Oct-Dec 2025)", ROOT_DIR / "strategies" / "V34.2" / "unseen_2025_q4"),
]

print("=" * 85)
print("GATE V2 AUDIT: BHAVISHYA V34 VS V34.2 ON UNSEEN DATASETS")
print("=" * 85)

all_passed = True
for label, folder in folders:
    print(f"\n=======================================================")
    print(f"AUDITING: {label}")
    print(f"Folder: {folder}")
    print(f"=======================================================")
    
    cmd = [sys.executable, str(eval_script), "--strategy-dir", str(folder)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    
    print(res.stdout)
    if res.stderr:
        print("STDERR:", res.stderr)
        
    if res.returncode != 0:
        print(f"[FAIL] {label} FAILED GATE V2 AUDIT! Exit code: {res.returncode}")
        all_passed = False
    else:
        print(f"[PASS] {label} PASSED ALL GATE V2 CHECKS! Exit code: 0")

print("\n" + "=" * 85)
if all_passed:
    print("ALL 4 UNSEEN FOLDS PASSED GATE V2 RIGOROUS ZERO-MODIFICATION AUDIT! [EXIT CODE 0]")
else:
    print("SOME RUNS FAILED GATE V2 AUDIT. CHECK LOGS ABOVE.")
print("=" * 85)
