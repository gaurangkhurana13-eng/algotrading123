import json

log_path = r"C:\Users\admin\.gemini\antigravity\brain\412f3c06-0a3e-451b-ac6b-7c58ad299e9f\.system_generated\logs\transcript.jsonl"
with open(log_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            content = data.get("content", "")
            if "Fold 13" in content and "V34.3" in content and "|" in content:
                print(f"Step: {data.get('step_index')}")
                print(content[:3000])
                print("..." if len(content) > 3000 else "")
        except:
            pass
