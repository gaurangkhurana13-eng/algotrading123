with open("brokerage-calculator-84e252e92a7f7dbf.js", "r", encoding="utf-8") as f:
    code = f.read()

idx = code.find("function eq_cal")
print(code[idx:idx+2500])
