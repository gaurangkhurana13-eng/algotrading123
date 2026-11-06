"""
Agent 5: Independent Auditor (auditor.py)
Executes the pristine, uncompromised strategy_evaluate_gate.py on any version folder.
Outputs audit_report.txt and enforces PASS/FAIL decision.
"""

import subprocess
import sys
from pathlib import Path

GATE_SCRIPT = Path(__file__).parent.parent / "strategy_evaluate_gate.py"
DATA_DIR = Path("C:/Users/admin/.gemini/antigravity/scratch/new_trade/data")
ROOT_DIR = Path(__file__).parent.parent

def run_independent_audit(version_tag: str, v_dir: Path) -> tuple[bool, str]:
    python_exe = sys.executable
    cmd = [
        python_exe, str(GATE_SCRIPT), version_tag,
        "--root", str(ROOT_DIR),
        "--data-dir", str(DATA_DIR),
        "--no-color"
    ]
    
    res = subprocess.run(cmd, capture_output=True, text=True)
    out = res.stdout + "\n" + res.stderr
    
    # Save audit_report.txt
    (v_dir / "audit_report.txt").write_text(out, encoding="utf-8")
    
    is_passed = (res.returncode == 0) and ("VERDICT: [FAIL]" not in out)
    return is_passed, out

if __name__ == "__main__":
    passed, log = run_independent_audit("V6", ROOT_DIR / "V6")
    print(f"Audit Result for V6: {'PASSED' if passed else 'FAILED'}")
