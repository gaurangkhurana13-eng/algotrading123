#!/usr/bin/env python3
# -*- coding: utf-8 -*-
with open("dhan_calc_engine.js", "r", encoding="utf-8") as f:
    text = f.read()

import re
matches = [m.start() for m in re.finditer(r'brokerage.*api|/api/|calc.*api|https://', text, re.I)]
for idx in matches[:10]:
    print("Match at", idx)
    print(text[max(0, idx-100):min(len(text), idx+200)])
    print("="*60)
