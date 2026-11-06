from pathlib import Path
import shutil

ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
v39_file = ROOT / "strategies" / "V39" / "run_v39.py"
v39_code = v39_file.read_text(encoding="utf-8")

# Header update
v39_1_code = v39_code.replace(
    "BHAVISHYA V39: THE QUANTUM HYBRID QUANT ENGINE",
    "BHAVISHYA V39.1: THE QUANTUM HYBRID QUANT ENGINE (WITH OPENING GAP-UP FILTER)"
)

# 1. Update data prep to include prev_close and open_gap_pct
old_prep = '        day_open = df_full.groupby("_date")["open"].transform("first")\n        df_full["day_ret"] = (df_full["close"] / day_open) - 1.0'
new_prep = '''        day_open = df_full.groupby("_date")["open"].transform("first")
        daily_close = df_full.groupby("_date")["close"].last()
        prev_close_map = daily_close.shift(1).to_dict()
        df_full["prev_close"] = df_full["_date"].map(prev_close_map)
        df_full["open_gap_pct"] = (day_open - df_full["prev_close"]) / (df_full["prev_close"] + 1e-9) * 100.0
        df_full["day_ret"] = (df_full["close"] / day_open) - 1.0'''

# 2. Add open_gap_pct to data_map dict
old_dict = '            "c_300": c_300, "day_open": day_open,'
new_dict = '            "c_300": c_300, "day_open": day_open, "open_gap_pct": df_full["open_gap_pct"],'

# 3. Add max_gap_up to run_simulation signature
old_sig = '    min_tm_odds: float = 0.40,\n    skip_midday_deadzone: bool = True\n) -> Tuple[float, List[dict]]:'
new_sig = '    min_tm_odds: float = 0.40,\n    skip_midday_deadzone: bool = True,\n    max_gap_up: float = 1.50\n) -> Tuple[float, List[dict]]:'

# 4. Add gap check in candidate loop
old_cand = '            # Day Trader Skill: Time Machine Odds Check\n            if min_tm_odds > 0:\n                odds = t_data["tm_odds"].iloc[idx]\n                if not np.isnan(odds) and odds < min_tm_odds:\n                    continue'
new_cand = '''            # Day Trader Skill: Time Machine Odds Check
            if min_tm_odds > 0:
                odds = t_data["tm_odds"].iloc[idx]
                if not np.isnan(odds) and odds < min_tm_odds:
                    continue

            # V39.1: Morning Opening Gap-Up Exhaustion Filter (max 1.5% gap-up)
            if max_gap_up is not None:
                gap_pct = t_data["open_gap_pct"].iloc[idx]
                if not np.isnan(gap_pct) and gap_pct > max_gap_up:
                    continue'''

v39_1_code = v39_1_code.replace(old_prep, new_prep).replace(old_dict, new_dict).replace(old_sig, new_sig).replace(old_cand, new_cand)

target_dir = ROOT / "strategies" / "V39.1"
target_dir.mkdir(parents=True, exist_ok=True)
(target_dir / "run_v39_1.py").write_text(v39_1_code, encoding="utf-8")
print("Wrote strategies/V39.1/run_v39_1.py successfully.")

# Copy Pine script and update title
v39_pine = ROOT / "strategies" / "V39" / "v39.pine"
pine_code = v39_pine.read_text(encoding="utf-8")
pine_code = pine_code.replace('strategy("Bhavishya V39', 'strategy("Bhavishya V39.1')
(target_dir / "v39_1.pine").write_text(pine_code, encoding="utf-8")
print("Wrote strategies/V39.1/v39_1.pine successfully.")
