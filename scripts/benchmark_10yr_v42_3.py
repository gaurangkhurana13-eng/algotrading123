import sys, pickle
from pathlib import Path
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(r"C:\Users\admin\.gemini\antigravity\scratch\new_trade")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "strategies"))

from strategies.V42_3.run_v42_3 import run_simulation_v42_3
from scripts.test_base_risk_sensitivity import simulate_v42_risk_tuning

stocks_30_legacy = {
    'CANBK', 'PNB', 'BANKBARODA', 'SBIN', 'IOC',
    'BPCL', 'GAIL', 'COALINDIA', 'TATASTEEL', 'TATAPOWER',
    'TATAMOTORS', 'VEDL', 'HINDALCO', 'JINDALSTEL', 'JSWSTEEL',
    'BEL', 'ADANIPOWER', 'ADANIENT', 'RECLTD', 'PFC',
    'DLF', 'MOTHERSON', 'ITC', 'ICICIBANK', 'AXISBANK',
    'RELIANCE', 'INFY', 'SUNPHARMA', 'TECHM', 'CIPLA'
}
V29_30_UNIVERSE = {
    "J&KBANK", "UTKARSHBNK", "MANALIPETC", "IDEA", "PCJEWELLER",
    "NIACL", "MTNL", "IFCI", "TEXRAIL", "SWSOLAR",
    "TTML", "RENUKA", "RCF", "NBCC", "JYOTHYLAB",
    "RBLBANK", "ELECTCAST", "PNB", "GSFC", "TATASTEEL",
    "SUZLON", "CENTRALBK", "IOB", "IRCON", "NMDC",
    "RITES", "NCC", "ZEEL", "DELTACORP", "RPOWER"
}
core_27 = {
    'CANBK', 'DELTACORP', 'FEDERALBNK', 'GMRINFRA', 'GSFC', 'IDFCFIRSTB', 'IOC',
    'JYOTHYLAB', 'MANAPPURAM', 'MOTHERSON', 'NBCC', 'NCC', 'NATIONALUM', 'NMDC',
    'POONAWALLA', 'PNB', 'RBLBANK', 'RCF', 'SAIL', 'SUZLON', 'TATACHEM',
    'TATASTEEL', 'TATAPOWER', 'TRIDENT', 'UNIONBANK', 'YESBANK', 'ZEEL'
}
u_2021 = {'CENTRALBK', 'DELTACORP', 'GSFC', 'IDEA', 'IOB', 'JYOTHYLAB', 'NBCC', 'NCC', 'NIACL', 'NMDC', 'PNB', 'RBLBANK', 'RCF', 'RENUKA', 'RITES', 'SUZLON', 'SWSOLAR', 'TATASTEEL', 'TTML', 'ZEEL'}

# Load Pools
p_2016_h1 = pickle.load(open(ROOT / 'data/2016_H1_UNSEEN/universe_30_15m.pkl', 'rb'))
p_2016_h2 = pickle.load(open(ROOT / 'data/2016_H2_UNSEEN/universe_30_15m.pkl', 'rb'))
p_2017 = pickle.load(open(ROOT / 'data/2017_UNSEEN/universe_30_15m.pkl', 'rb'))
p_2018 = pickle.load(open(ROOT / 'data/2018_UNSEEN/universe_30_15m.pkl', 'rb'))
p_2019 = pickle.load(open(ROOT / 'data/2019_UNSEEN/universe_30_15m.pkl', 'rb'))
p_2020 = pickle.load(open(ROOT / 'data/2020_FULL/universe_30_15m.pkl', 'rb'))
p_2021 = pickle.load(open(ROOT / 'data/2021-22/universe_30_15m.pkl', 'rb'))
p_2022 = pickle.load(open(ROOT / 'data/2022-23/universe_30_15m.pkl', 'rb'))
p_2023 = pickle.load(open(ROOT / 'data/2023-24/universe_30_15m.pkl', 'rb'))
p_2024 = pickle.load(open(ROOT / 'data/2024-25/universe_14_15m.pkl', 'rb'))
p_2025 = pickle.load(open(ROOT / 'data/2025-26/universe_14_15m.pkl', 'rb'))
p_2026 = pickle.load(open(ROOT / 'data/universe_78_15m_1y.pkl', 'rb'))

timeline = [
    ("2016_H1", p_2016_h1, "2016-01-01", "2016-06-30", stocks_30_legacy),
    ("2016_H2", p_2016_h2, "2016-07-01", "2016-12-31", stocks_30_legacy),
    ("2017", p_2017, "2017-01-01", "2017-12-31", stocks_30_legacy),
    ("2018", p_2018, "2018-01-01", "2018-12-31", stocks_30_legacy),
    ("2019", p_2019, "2019-01-01", "2019-12-31", None),
    ("2020", p_2020, "2020-01-01", "2020-12-31", core_27),
    ("2021", p_2021, "2021-01-01", "2021-12-31", u_2021),
    ("2022", p_2022, "2022-01-01", "2022-12-31", V29_30_UNIVERSE),
    ("2023", p_2023, "2023-01-02", "2023-12-29", V29_30_UNIVERSE),
    ("2024", p_2024, "2024-01-01", "2024-12-31", V29_30_UNIVERSE),
    ("2025", p_2025, "2025-01-01", "2025-12-31", V29_30_UNIVERSE),
    ("2026", p_2026, "2026-01-01", "2026-10-29", V29_30_UNIVERSE),
]

def main():
    print("=" * 115)
    print("BHAVISHYA 10-YEAR HISTORICAL BENCHMARK: V42 TURBO vs V42.3 (INVINCIBLE PREDATOR)")
    print("Starting Capital: Rs 500.00 | Full 5x MIS Margin | Actual Dhan Brokerage + Slippage")
    print("=" * 115)
    print(f"{'Era / Fold':<10} | {'V42 Turbo Net':<14} | {'V42 Turbo DD':<12} | {'V42.3 Predator Net':<18} | {'V42.3 WR':<10} | {'V42.3 DD':<10} | {'Status'}")
    print("-" * 115)

    tot_v42_net = 0.0
    tot_v42_3_net = 0.0
    results_table = []

    for era_name, pool, s_dt, e_dt, u_filt in timeline:
        # 1. Run V42 Turbo Baseline
        _, net_v42, wr_v42, dd_v42, tr_v42 = simulate_v42_risk_tuning(pool, s_dt, e_dt, u_filt, base_risk=25.0)
        tot_v42_net += net_v42

        # 2. Run V42.3 Hybrid Predator
        cap_v42_3, trades_v42_3, wr_v42_3, dd_v42_3, pf_v42_3 = run_simulation_v42_3(
            pool, s_dt, e_dt, era_name, universe_filter=u_filt, base_risk=25.0, use_orb_filter=True
        )
        net_v42_3 = cap_v42_3 - 500.0
        tot_v42_3_net += net_v42_3

        status = "EXCELLENT" if net_v42_3 > 0 and dd_v42_3 < 15.0 else ("SURVIVED" if net_v42_3 >= 0 else "DEFENSE")
        print(f"{era_name:<10} | Rs {net_v42:+10.2f} | {dd_v42:8.1f}%   | Rs {net_v42_3:+14.2f}   | {wr_v42_3:6.1f}%   | {dd_v42_3:6.1f}%   | {status}")

        results_table.append({
            "era": era_name,
            "v42_net": net_v42,
            "v42_dd": dd_v42,
            "v42_3_net": net_v42_3,
            "v42_3_wr": wr_v42_3,
            "v42_3_dd": dd_v42_3,
            "v42_3_pf": pf_v42_3,
            "v42_3_trades": len(trades_v42_3)
        })

    print("=" * 115)
    print(f"10-YEAR GRAND TOTAL NET PnL:")
    print(f"  V42 Turbo Baseline       : Rs {tot_v42_net:+10.2f}")
    print(f"  V42.3 Invincible Predator: Rs {tot_v42_3_net:+10.2f}")
    print(f"  Difference               : Rs {tot_v42_3_net - tot_v42_net:+10.2f}")
    print("=" * 115)

    with open(ROOT / "data" / "v42_3_10yr_benchmark_results.pkl", "wb") as f:
        pickle.dump(results_table, f)
    print("Results saved to data/v42_3_10yr_benchmark_results.pkl")

if __name__ == "__main__":
    main()
