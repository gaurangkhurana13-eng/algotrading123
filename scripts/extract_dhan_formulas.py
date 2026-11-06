#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
extract_dhan_formulas.py
Downloads the Next.js chunk for Dhan's brokerage calculator and extracts the exact formulas.
"""

import urllib.request
import re

chunk_url = "https://dhan.co/_next/static/chunks/pages/calculators/brokerage-calculator-7991cc9c23d4c75f.js"
req = urllib.request.Request(chunk_url, headers={"User-Agent": "Mozilla/5.0"})

try:
    with urllib.request.urlopen(req, timeout=10) as response:
        js = response.read().decode("utf-8", errors="ignore")
        print("JS chunk fetched. Length:", len(js))
        with open("dhan_calc_engine.js", "w", encoding="utf-8") as f:
            f.write(js)
            
        # Search for key calculations
        # Looking for functions calculating brokerage, stt, delivery, intraday
        matches = re.findall(r'(\b[a-zA-Z0-9_$]+\s*=\s*function\s*\([^)]*\)\s*\{[^}]*\b(?:brokerage|stt|turnover|sebi|exchange)[^}]*\})', js, re.IGNORECASE)
        print("Function matches found:", len(matches))
        
        # Search for specific rates
        print("\n--- Searching for fee constants in JS chunk ---")
        for m in re.finditer(r'([a-zA-Z0-9_$]+)?\s*[:=]\s*(?:0\.00[0-9]+|0\.0[0-9]+|0\.18|20|14\.75|12\.5|1500)', js):
            context = js[max(0, m.start()-50):min(len(js), m.end()+50)]
            if any(k in context.lower() for k in ["stt", "brokerage", "delivery", "intraday", "gst", "stamp", "sebi", "exchange", "dp"]):
                print("Context:", context.replace("\n", " "))
                
except Exception as e:
    print("Error:", e)
