import json

transcript_path = r"C:\Users\admin\.gemini\antigravity\brain\412f3c06-0a3e-451b-ac6b-7c58ad299e9f\.system_generated\logs\transcript_full.jsonl"
with open(transcript_path, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        try:
            data = json.loads(line)
            if data.get("step_index") == 12136:
                print("Keys:", data.keys())
                for k, v in data.items():
                    print(f"Key: {k}, type: {type(v)}, length: {len(str(v))}")
                    if k in ['content', 'text']:
                        print(f"Sample: {str(v)[:300]}")
        except Exception as e:
            pass
