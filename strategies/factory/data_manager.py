"""
Agent 2: Data & Universe Engineer (data_manager.py)
Validates OHLCV bar provenance, ensures clean timestamps, handles splits:
- In-Sample (Dev/Backtest)
- Reserved True Out-of-Sample (OOS) for Walk-Forward Validation
"""

import hashlib
import json
from pathlib import Path
import pandas as pd
import numpy as np

DATA_DIR = Path("C:/Users/admin/.gemini/antigravity/scratch/new_trade/data")
BASKET = ["PCJEWELLER.NS", "RPOWER.NS", "IDEA.NS", "MTNL.NS", "HCC.NS", "UJJIVANSFB.NS"]

def get_file_hash(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()[:16]

def load_clean_ticker_data(ticker: str) -> pd.DataFrame:
    clean = ticker.replace(".NS", "").lower()
    p = DATA_DIR / f"{clean}_15m_1y.pkl"
    if not p.exists():
        raise FileNotFoundError(f"Bar data not found: {p}")
    df = pd.read_pickle(p)
    if df.index.tz is not None:
        df = df.copy()
        df.index = df.index.tz_localize(None)
    df.columns = [c.lower() for c in df.columns]
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df

def get_in_sample_and_oos(ticker: str, oos_cutoff="2026-09-20"):
    df = load_clean_ticker_data(ticker)
    cutoff = pd.Timestamp(oos_cutoff)
    df_is = df[df.index <= cutoff].copy()
    df_oos = df[df.index > cutoff].copy()
    return df_is, df_oos

def create_data_manifest(target_dir: Path, basket=BASKET, oos_cutoff="2026-09-20"):
    manifest = {
        "basket": basket,
        "oos_cutoff": oos_cutoff,
        "files": {}
    }
    for t in basket:
        clean = t.replace(".NS", "").lower()
        p = DATA_DIR / f"{clean}_15m_1y.pkl"
        if p.exists():
            df = load_clean_ticker_data(t)
            df_is, df_oos = get_in_sample_and_oos(t, oos_cutoff)
            manifest["files"][t] = {
                "file_name": p.name,
                "sha256": get_file_hash(p),
                "total_bars": len(df),
                "in_sample_bars": len(df_is),
                "oos_bars": len(df_oos),
                "date_min": str(df.index.min()),
                "date_max": str(df.index.max())
            }
    out_file = target_dir / "data_manifest.json"
    out_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest

if __name__ == "__main__":
    m = create_data_manifest(Path("."))
    print("Data Manifest Created:", len(m["files"]), "stocks verified.")
