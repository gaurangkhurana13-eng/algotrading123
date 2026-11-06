from pathlib import Path

ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
v39_1_file = ROOT / "strategies" / "V39.1" / "run_v39_1.py"
code = v39_1_file.read_text(encoding="utf-8")

# 1. Update header
code = code.replace(
    "BHAVISHYA V39.1: THE QUANTUM HYBRID QUANT ENGINE (WITH OPENING GAP-UP FILTER)",
    "BHAVISHYA V40: THE BULLETPROOF QUANTUM GENERATIONAL WEALTH ENGINE"
)

# 2. Add allow_dynamic_runner_slot parameter
code = code.replace(
    "    max_gap_up: float = 1.50\n) -> Tuple[float, List[dict]]:",
    "    max_gap_up: float = 1.50,\n    allow_dynamic_runner_slot: bool = True\n) -> Tuple[float, List[dict]]:"
)

# 3. Replace ratchet with V40 Elastic Runner Ratchet
old_ratchet = '''            # Hybrid Stage Ratchet
            if mfe_atr >= 3.00:
                if psi_climax:
                    stop_ladder = p["high_seen"] - (0.08 * p["entry_atr"])
                else:
                    prev_l = p.get("prev_l", cur_l)
                    stop_ladder = max(p["high_seen"] - (0.26 * p["entry_atr"]), prev_l - (0.08 * p["entry_atr"]))
            elif mfe_atr >= 2.00:
                stop_ladder = p["high_seen"] - (0.20 * p["entry_atr"])
            elif mfe_atr >= 1.15:
                stop_ladder = p["high_seen"] - (0.28 * p["entry_atr"])
            elif mfe_atr >= 0.85:
                stop_ladder = p["entry_p"] + (0.40 * p["entry_atr"])
            else:
                stop_ladder = p["entry_p"] - (0.90 * p["entry_atr"])

            if mfe_atr >= 0.85 and cur_c < cur_ema9 and cur_rsi < 55.0:
                stop_ladder += 0.05 * p["entry_atr"]'''

new_ratchet = '''            # V40 Elastic Trend Runner Ratchet
            prev_l = p.get("prev_l", cur_l)
            if mfe_atr >= 3.50:
                if psi_climax:
                    stop_ladder = p["high_seen"] - (0.12 * p["entry_atr"])
                else:
                    stop_ladder = max(p["high_seen"] - (0.35 * p["entry_atr"]), prev_l - (0.05 * p["entry_atr"]))
            elif mfe_atr >= 2.25:
                if psi_climax:
                    stop_ladder = p["high_seen"] - (0.18 * p["entry_atr"])
                else:
                    stop_ladder = max(p["high_seen"] - (0.40 * p["entry_atr"]), prev_l - (0.08 * p["entry_atr"]))
            elif mfe_atr >= 1.50:
                stop_ladder = max(p["entry_p"] + (0.75 * p["entry_atr"]), p["high_seen"] - (0.45 * p["entry_atr"]))
            elif mfe_atr >= 0.85:
                stop_ladder = p["entry_p"] + (0.18 * p["entry_atr"])
            else:
                stop_ladder = p["entry_p"] - (0.90 * p["entry_atr"])

            if mfe_atr >= 1.50 and cur_c < cur_ema9 and cur_rsi < 52.0:
                stop_ladder += 0.05 * p["entry_atr"]'''

code = code.replace(old_ratchet, new_ratchet)

# 4. Replace single slot check with Smart Concurrency
old_slot = '''        # Single Slot Discipline
        if len(positions) >= 1 or ts in last_exit_times or t_time < pd.Timestamp("09:20").time() or t_time > pd.Timestamp("14:15").time():
            continue'''

new_slot = '''        # Base time filters
        if ts in last_exit_times or t_time < pd.Timestamp("09:20").time() or t_time > pd.Timestamp("14:15").time():
            continue

        # Smart Concurrency Management:
        # Default max positions = 1.
        # Allow 2nd position ONLY IF 1st position has ALREADY locked in profit (active_sl >= entry_p).
        max_allowed_positions = 1
        if allow_dynamic_runner_slot and len(positions) == 1:
            p_first = list(positions.values())[0]
            if p_first.get("active_sl", 0.0) >= p_first["entry_p"]:
                max_allowed_positions = 2

        if len(positions) >= max_allowed_positions:
            continue'''

code = code.replace(old_slot, new_slot)

# 5. Replace gap check with Bull/Bear adaptive gap check
old_gap = '''            # V39.1: Morning Opening Gap-Up Exhaustion Filter (max 1.5% gap-up)
            if max_gap_up is not None:
                gap_pct = t_data["open_gap_pct"].iloc[idx]
                if not np.isnan(gap_pct) and gap_pct > max_gap_up:
                    continue'''

new_gap = '''            # V40: Opening Gap-Up Exhaustion & Continuation Filter
            if max_gap_up is not None:
                gap_pct = t_data["open_gap_pct"].iloc[idx]
                if not np.isnan(gap_pct):
                    b_c300_now = c300_breadth.loc[ts] if ts in c300_breadth.index else 0.50
                    b_day_now = bench_day.loc[ts] if ts in bench_day.index else 0.0
                    eff_cap = 2.50 if (b_c300_now >= 0.50 and b_day_now > 0.0 and t_data["rvol"].iloc[idx] >= 1.50) else max_gap_up
                    if gap_pct > eff_cap:
                        continue'''

code = code.replace(old_gap, new_gap)

# Save V40
target_dir = ROOT / "strategies" / "V40"
target_dir.mkdir(parents=True, exist_ok=True)
(target_dir / "run_v40.py").write_text(code, encoding="utf-8")
print("Successfully generated clean strategies/V40/run_v40.py")
