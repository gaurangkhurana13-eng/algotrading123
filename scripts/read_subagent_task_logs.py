import os

task_logs = [
    ("Pattern Architect task-104", r"C:\Users\admin\.gemini\antigravity\brain\c5233b5a-86c7-46cc-9937-ed59616a37c6\.system_generated\tasks\task-104.log"),
    ("Exit Optimizer task-111", r"C:\Users\admin\.gemini\antigravity\brain\14caeb9d-9b06-488d-b768-84aae247fbad\.system_generated\tasks\task-111.log"),
    ("Capital Allocator task-98", r"C:\Users\admin\.gemini\antigravity\brain\631c4cec-ff12-43e9-94fd-d50f8a7c55c6\.system_generated\tasks\task-98.log"),
]

for label, log_path in task_logs:
    print(f"================ {label} ================")
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        print(f"Size: {len(content)} chars")
        lines = content.strip().splitlines()
        print(f"Total lines: {len(lines)}")
        if len(lines) > 40:
            print("--- First 15 lines ---")
            print("\n".join(lines[:15]))
            print("--- Last 25 lines ---")
            print("\n".join(lines[-25:]))
        else:
            print("\n".join(lines))
    else:
        print("File not found.")
