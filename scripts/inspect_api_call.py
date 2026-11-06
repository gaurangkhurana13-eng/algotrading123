#!/usr/bin/env python3
# -*- coding: utf-8 -*-
with open("dhan_calc_engine.js", "r", encoding="utf-8") as f:
    text = f.read()

idx = 20562
start = max(0, idx - 600)
end = min(len(text), idx + 200)
print(text[start:end])
