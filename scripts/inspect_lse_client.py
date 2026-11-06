with open("lse_bundle.js", "r", encoding="utf-8") as f:
    js = f.read()

target = "71f880e1d2ef471664f3b6c04c6dc1e618f94e51f68c87522bc6dcbc0ca173a5"
idx = js.find(target)
if idx != -1:
    print("Found target string!")
    print("\n=== CONTEXT AROUND API CONFIG & CLIENT ===")
    print(js[max(0, idx - 400): min(len(js), idx + 2500)])
else:
    print("Target not found")
