"""
Agent 7 & 8: Risk Analyst & Release Documenter (releaser.py)
Validates risk limits, generates run_manifest.json with cryptographic hashes,
and updates the Master README.md ledger.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
import pandas as pd

def get_hash(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()[:16]

def check_risk_and_costs(v_dir: Path) -> tuple[bool, str]:
    intra_csv = v_dir / "trades_intraday.csv"
    if not intra_csv.exists():
        return False, "trades_intraday.csv missing"
    df = pd.read_csv(intra_csv)
    if len(df) == 0:
        return False, "Zero trades"
        
    worst_loss = df["net_pnl"].min()
    if worst_loss < -15.50:
        return False, f"Worst loss {worst_loss} breached Rs 15.50 cap"
        
    tot_charges = df["charges"].sum()
    tot_gross = df[df["gross_pnl"] > 0]["gross_pnl"].sum()
    drag = (tot_charges / tot_gross) if tot_gross > 0 else 1.0
    if drag > 0.40:
        return False, f"Fee drag {drag:.1%} exceeds 40% threshold"
        
    return True, "Risk and cost limits verified."

def create_run_manifest(version_tag: str, v_dir: Path, status="APPROVED"):
    manifest = {
        "version": version_tag,
        "run_id": f"RUN_{version_tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "timestamp": datetime.now().isoformat(),
        "status": status,
        "files": {}
    }
    for f in v_dir.glob("*.*"):
        if f.is_file():
            manifest["files"][f.name] = {
                "sha256": get_hash(f),
                "size_bytes": f.stat().st_size
            }
    (v_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest

def update_master_readme(version_tag: str, v_dir: Path, hypothesis: dict):
    readme_path = v_dir.parent / "README.md"
    res_path = v_dir / "results.txt"
    if not readme_path.exists() or not res_path.exists():
        return
        
    txt = res_path.read_text(encoding="utf-8")
    # Parse metrics from results.txt
    intra_lines = txt.split("INTRADAY")[1].split("SWING")[0]
    trades = [l.split(":")[1].strip() for l in intra_lines.splitlines() if "Trades:" in l][0]
    wr = [l.split(":")[1].strip() for l in intra_lines.splitlines() if "Win Rate:" in l][0]
    pnl = [l.split(":")[1].strip() for l in intra_lines.splitlines() if "Net Profit:" in l][0]
    pf = [l.split(":")[1].strip() for l in intra_lines.splitlines() if "Profit Factor:" in l][0]
    
    swing_lines = txt.split("SWING")[1].split("=")[0]
    sw_trades = [l.split(":")[1].strip() for l in swing_lines.splitlines() if "Trades:" in l][0]
    sw_wr = [l.split(":")[1].strip() for l in swing_lines.splitlines() if "Win Rate:" in l][0]
    sw_pnl = [l.split(":")[1].strip() for l in swing_lines.splitlines() if "Net Profit:" in l][0]
    sw_pf = [l.split(":")[1].strip() for l in swing_lines.splitlines() if "Profit Factor:" in l][0]
    
    new_row = f"| **[{version_tag}](file:///{v_dir.as_posix()}/results.txt)** | {hypothesis['name']} | {trades} | **{wr}** | **{pf}** | **{pnl}** | {sw_trades} | {sw_wr} | {sw_pf} | {sw_pnl} | 🏆 **APPROVED & VERIFIED** |\n"
    
    content = readme_path.read_text(encoding="utf-8")
    if version_tag not in content:
        # Insert before section 2
        parts = content.split("---")
        if len(parts) >= 2:
            parts[1] = parts[1] + new_row
            content = "---".join(parts)
            readme_path.write_text(content, encoding="utf-8")
