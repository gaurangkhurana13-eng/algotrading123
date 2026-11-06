import os
import re

for fname in os.listdir("."):
    if fname.endswith(".js"):
        with open(fname, "r", encoding="utf-8", errors="ignore") as f:
            c = f.read()
            if "69006:" in c or "69006 :" in c:
                print(f"Found 69006 in {fname}")
            if "BT:" in c or "BT :" in c or "calculateBrokerage" in c:
                print(f"Found BT/calc in {fname}")
