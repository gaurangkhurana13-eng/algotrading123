from pathlib import Path

ROOT_DIR = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
unseen_dir = ROOT_DIR / "data" / "2025-26"
files = [f.stem.replace("_15m_1y", "").upper() for f in unseen_dir.glob("*.pkl")]
print(f"Total symbols in data/2025-26: {len(files)}")
print("Symbols:", sorted(files))
