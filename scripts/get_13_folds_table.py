import json

log_path = r"C:\Users\admin\.gemini\antigravity\brain\412f3c06-0a3e-451b-ac6b-7c58ad299e9f\.system_generated\logs\transcript.jsonl"
with open(log_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            step = data.get("step_index")
            if step in [15654, 15655, 15656, 15657]:
                content = data.get("content", "")
                if "Fold" in content:
                    print(f"=== STEP {step} ===")
                    print(content[:4000])
        except Exception as e:
            pass
