import urllib.request
import re
import json

url = 'https://dhan.co/calculators/brokerage-calculator/'
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'})

try:
    with urllib.request.urlopen(req) as resp:
        html = resp.read().decode('utf-8', errors='ignore')
        print(f"HTML length: {len(html)}")
        
        # Save HTML for inspection
        with open("dhan_calc.html", "w", encoding="utf-8") as f:
            f.write(html)
        print("Saved dhan_calc.html")
        
        # Find script tags
        scripts = re.findall(r'<script[^>]+src=["\']([^"\']+)["\']', html)
        print("Found scripts:", len(scripts))
        calc_scripts = [s for s in scripts if 'calc' in s.lower() or 'brokerage' in s.lower() or 'main' in s.lower() or 'app' in s.lower() or '_app' in s.lower() or 'chunk' in s.lower()]
        for s in calc_scripts:
            print("Calc script:", s)
            
        # Also check inline scripts
        inline_scripts = re.findall(r'<script(?![^>]*src)[^>]*>(.*?)</script>', html, re.DOTALL)
        print("Found inline scripts:", len(inline_scripts))
        for i, s in enumerate(inline_scripts):
            if any(k in s.lower() for k in ['brokerage', 'stt', 'stamp', '0.03', '20']):
                print(f"Inline script {i} relevant (len {len(s)})")
                with open(f"inline_script_{i}.js", "w", encoding="utf-8") as f:
                    f.write(s)
except Exception as e:
    print("Error:", e)
