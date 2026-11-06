with open("brokerage-calculator-84e252e92a7f7dbf.js", "r", encoding="utf-8") as f:
    code = f.read()

idx = code.find("function eq_cal")
# let's look at the 2000 chars before function eq_cal
print(code[max(0, idx-2000):idx])
