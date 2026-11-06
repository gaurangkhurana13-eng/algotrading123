with open(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade\strategies\V36\run_v36.py", "r") as f:
    code = f.read()

# Replace V36 with V37
code = code.replace("BHAVISHYA V36", "BHAVISHYA V37")
code = code.replace("Bhavishya V36", "Bhavishya V37")
code = code.replace("V36", "V37")
code = code.replace("v36", "v37")

# Update t1_b RVOL threshold from 1.30 to 1.50
code = code.replace("(er10 >= 0.40) & (rvol >= 1.30) &", "(er10 >= 0.40) & (rvol >= 1.50) &")

with open(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade\strategies\V37\run_v37.py", "w") as f:
    f.write(code)

print("Created strategies/V37/run_v37.py successfully.")
