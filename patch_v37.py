import pickle
from pathlib import Path
ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")

with open(ROOT / "strategies" / "V37" / "run_v37.py", "r") as f:
    code = f.read()

# 1. Modify function definition
code = code.replace(
    'max_slots: int = 2\n) -> Tuple[float, List[dict]]:',
    'max_slots: int = 2,\n    cooldown_sessions: int = 2\n) -> Tuple[float, List[dict]]:'
)

# 2. Add state
code = code.replace(
    '    slippage = 0.0005\n    alpha_streak = 0.082',
    '    slippage = 0.0005\n    symbol_sl_lockout = {}\n    unique_dates = sorted(list(set(ts.date() for ts in all_ts)))\n    date_to_idx = {d: i for i, d in enumerate(unique_dates)}\n    alpha_streak = 0.082'
)

# 3. Add cur_d_idx
code = code.replace(
    '        cur_date = ts.date()\n        t_time = ts.time()',
    '        cur_date = ts.date()\n        cur_d_idx = date_to_idx[cur_date]\n        t_time = ts.time()'
)

# 4. In finalize_trade
code = code.replace(
    '                last_exit_times.add(ts)\n                del positions[ticker]',
    '                last_exit_times.add(ts)\n                if reason == "HARD_SL" and cooldown_sessions > 0:\n                    symbol_sl_lockout[ticker] = cur_d_idx + cooldown_sessions\n                del positions[ticker]'
)

# 5. In candidate loop
code = code.replace(
    '        for ticker in data_map.keys():\n            if ticker in positions or ticker in daily_traded[cur_date]:\n                continue',
    '        for ticker in data_map.keys():\n            if ticker in positions or ticker in daily_traded[cur_date]:\n                continue\n            if ticker in symbol_sl_lockout and cur_d_idx < symbol_sl_lockout[ticker]:\n                continue'
)

with open(ROOT / "test_v37_plus_cooldown.py", "w") as f:
    f.write(code)

print("test_v37_plus_cooldown.py written!")
