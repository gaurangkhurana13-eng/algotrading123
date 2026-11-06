#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re

with open("dhan_calc_engine.js", "r", encoding="utf-8") as f:
    text = f.read()

# Search for variables or calculations involving brokerage
# Find parts with .delivery or .intraday or rates
print("Script size:", len(text))
matches = [m.start() for m in re.finditer(r'brokerage', text, re.I)]
print("Brokerage mentions:", len(matches))

for idx in matches[:10]:
    start = max(0, idx - 150)
    end = min(len(text), idx + 250)
    print("\n--- Snippet around idx", idx, "---")
    print(text[start:end])
