#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reverse_engineer_dhan.py
Fetches Dhan Brokerage Calculator page and inspects exact formulas for:
1. Equity Intraday
2. Equity Delivery
"""

import urllib.request
import re
import json

url = "https://dhan.co/calculators/brokerage-calculator/"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})

try:
    with urllib.request.urlopen(req, timeout=10) as response:
        html = response.read().decode("utf-8", errors="ignore")
        print("HTML fetched successfully. Length:", len(html))
        
        # Save HTML locally for forensic examination
        with open("dhan_calc.html", "w", encoding="utf-8") as f:
            f.write(html)
            
        # Search for inline scripts or calculators logic
        scripts = re.findall(r'<script[^>]*src=["\']([^"\']+)["\']', html)
        print("Total external scripts found:", len(scripts))
        for s in scripts:
            if any(k in s.lower() for k in ["brokerage", "calc", "main", "chunk", "app"]):
                print("Candidate script:", s)
                
        # Also check inline script tags
        inline_scripts = re.findall(r'<script(?![^>]*src)[^>]*>(.*?)</script>', html, re.DOTALL)
        print("Inline script blocks:", len(inline_scripts))
        for idx, sc in enumerate(inline_scripts):
            if "brokerage" in sc.lower() or "stt" in sc.lower() or "turnover" in sc.lower():
                print(f"Inline script #{idx} contains fee keywords (length: {len(sc)})")
                with open(f"inline_script_{idx}.js", "w", encoding="utf-8") as f:
                    f.write(sc)
                    
except Exception as e:
    print("Fetch error:", e)
