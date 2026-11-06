import sys
import pandas as pd
import numpy as np
print(f"Python: {sys.version.split()[0]}")
print(f"Pandas: {pd.__version__}")
print(f"NumPy: {np.__version__}")
try:
    from tvDatafeed import TvDatafeed, Interval
    print("TvDatafeed: Available")
except Exception as e:
    print(f"TvDatafeed: Not available ({e})")
