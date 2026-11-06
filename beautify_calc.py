import re
import jsbeautifier

with open("brokerage-calculator-84e252e92a7f7dbf.js", "r", encoding="utf-8") as f:
    code = f.read()

opts = jsbeautifier.default_options()
opts.indent_size = 2
try:
    beautified = jsbeautifier.beautify(code, opts)
    with open("dhan_calc_beautified.js", "w", encoding="utf-8") as f:
        f.write(beautified)
    print("Beautified successfully!")
except Exception as e:
    print("Jsbeautifier error or not installed:", e)
    # If jsbeautifier not installed, format with simple formatting
    formatted = code.replace(";", ";\n").replace("{", "{\n").replace("}", "\n}\n")
    with open("dhan_calc_beautified.js", "w", encoding="utf-8") as f:
        f.write(formatted)
    print("Formatted with simple replacements.")
