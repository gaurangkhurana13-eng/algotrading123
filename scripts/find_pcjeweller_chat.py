import os
from pathlib import Path
import json

BRAIN_DIR = Path(r"C:\Users\admin\.gemini\antigravity\brain")

for conv_dir in BRAIN_DIR.iterdir():
    if not conv_dir.is_dir():
        continue
    # Check transcript
    t_file = conv_dir / ".system_generated/logs/transcript.jsonl"
    if t_file.exists():
        try:
            with open(t_file, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
                if "pcjeweller" in content.lower() and ("swing" in content.lower() or "southbank" in content.lower()):
                    print(f"Matched conversation: {conv_dir.name}")
                    # Print first user input
                    for line in content.splitlines()[:10]:
                        try:
                            obj = json.loads(line)
                            if obj.get("type") == "USER_INPUT":
                                print(f"  First user input: {obj.get('content', '')[:100]}")
                                break
                        except:
                            pass
        except Exception as e:
            pass
