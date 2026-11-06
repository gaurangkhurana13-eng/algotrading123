#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re

with open("dhan_calc_engine.js", "r", encoding="utf-8") as f:
    text = f.read()

matches = [m.start() for m in re.finditer(r'brokerage', text, re.I)]

with open("dhan_snippets.txt", "w", encoding="utf-8") as out:
    out.write(f"Brokerage mentions: {len(matches)}\n\n")
    for idx in matches:
        start = max(0, idx - 200)
        end = min(len(text), idx + 350)
        snippet = text[start:end]
        if any(w in snippet for w in ["stt", "sebi", "stamp", "gst", "0.03", "0.1", "turnover"]):
            out.write(f"=== Match at {idx} ===\n")
            out.write(snippet + "\n")
            out.write("="*70 + "\n\n")

print("Saved relevant snippets to dhan_snippets.txt")
