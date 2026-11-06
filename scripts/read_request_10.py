import json
from pathlib import Path

p = Path(r"C:\Users\admin\.gemini\antigravity\brain\412f3c06-0a3e-451b-ac6b-7c58ad299e9f\.system_generated\logs\transcript_full.jsonl")
if not p.exists():
    p = Path(r"C:\Users\admin\.gemini\antigravity\brain\412f3c06-0a3e-451b-ac6b-7c58ad299e9f\.system_generated\logs\transcript.jsonl")

with open(p, "r", encoding="utf-8", errors="ignore") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            if d.get("type") == "USER_INPUT" and "Sizing-Denominator Trap" in d.get("content", ""):
                print("=== FOUND REQUEST 10 ===")
                with open("request_10_full.txt", "w", encoding="utf-8") as out:
                    out.write(d["content"])
                print("Written to request_10_full.txt, length:", len(d["content"]))
                break
        except Exception as e:
            continue
