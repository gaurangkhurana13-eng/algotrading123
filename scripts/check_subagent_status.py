import json, os

subagents = [
    ('14caeb9d-9b06-488d-b768-84aae247fbad', 'Exit Optimizer'),
    ('631c4cec-ff12-43e9-94fd-d50f8a7c55c6', 'Capital Allocator'),
    ('c5233b5a-86c7-46cc-9937-ed59616a37c6', 'Pattern Architect')
]

for sub_id, name in subagents:
    path = f"C:/Users/admin/.gemini/antigravity/brain/{sub_id}/.system_generated/logs/transcript.jsonl"
    print(f"================ {name} ================")
    if not os.path.exists(path):
        print("Transcript does not exist yet.")
        continue
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    print(f"Total lines: {len(lines)}")
    for l in lines[-4:]:
        try:
            d = json.loads(l)
            t = d.get("type", "")
            c = d.get("content", "") or ""
            print(f"[{t}] {c[:250].strip()}")
        except Exception as e:
            print("Error parsing line:", e)
