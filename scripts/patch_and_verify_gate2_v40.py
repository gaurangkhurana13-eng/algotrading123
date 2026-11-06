import os, sys, subprocess
from pathlib import Path

ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
GATE2 = ROOT / "strategies/verify_simulation_integrity.py"

folds_def = [
    (0, ROOT / 'data/2021-22'),
    (1, ROOT / 'data'),
    (2, ROOT / 'data/2022-23'),
    (3, ROOT / 'data/2023-24'),
    (4, ROOT / 'data/2023-24'),
    (5, ROOT / 'data/2023-24'),
    (6, ROOT / 'data/2023-24'),
    (7, ROOT / 'data/2024-25'),
    (8, ROOT / 'data/2024-25'),
    (9, ROOT / 'data/2025-26'),
    (10, ROOT / 'data/2025-26'),
    (11, ROOT / 'data/2025-26'),
    (12, ROOT / 'data/virgin_20_bars'),
    (13, ROOT / 'data'),
    (14, ROOT / 'data/2020_FULL'),
]

gate2_results = {}

for fid, data_dir in folds_def:
    f_dir = ROOT / f"strategies/V40_FOLD_{fid}"
    run_file = f_dir / f"run_V40_FOLD_{fid}.py"
    if run_file.exists():
        content = run_file.read_text(encoding="utf-8")
        if "pool_spec = 'C:\\" in content:
            content = content.replace("pool_spec = 'C:\\", "pool_spec = r'C:\\")
            run_file.write_text(content, encoding="utf-8")
        elif "pool_spec = 'C:/" in content:
            content = content.replace("pool_spec = 'C:/", "pool_spec = r'C:/")
            run_file.write_text(content, encoding="utf-8")
            
    p = subprocess.run([sys.executable, str(GATE2), str(f_dir), "--data-dir", str(data_dir)], capture_output=True, text=True, cwd=str(ROOT))
    status = "PASS" if p.returncode == 0 else "FAIL"
    gate2_results[fid] = status
    print(f"V40_FOLD_{fid}: Gate 2 -> {status}")

print("\nSummary of Gate 2 across all 15 folds:")
for fid, status in gate2_results.items():
    print(f"Fold {fid}: {status}")
