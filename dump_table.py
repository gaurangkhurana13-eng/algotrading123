import json, os

versions_meta = [
    {'v': 'V7.0', 'focus': 'Colab Benchmark (Compounding Sizing Flaw)', 'trades': 33, 'w': 19, 'l': 14, 'wr': 57.6, 'avg_w': 68.77, 'avg_l': -69.61, 'payoff': 0.99, 'pf': 1.34, 'pnl': 332.08, 'roi': 66.42, 'ruin': 25.15, 'dd': 22.8},
    {'v': 'V7.1', 'focus': 'Asymmetric Tight SL 1.0 ATR (Noise Choke)', 'trades': 62, 'w': 10, 'l': 52, 'wr': 16.13, 'avg_w': 31.76, 'avg_l': -10.79, 'payoff': 2.94, 'pf': 0.57, 'pnl': -243.55, 'roi': -48.71, 'ruin': 58.43, 'dd': 51.2},
    {'v': 'V8', 'focus': 'Macro Nifty Filter (Noise Buffer SL 1.5 ATR)', 'trades': 37, 'w': 13, 'l': 24, 'wr': 35.14, 'avg_w': 17.67, 'avg_l': -17.72, 'payoff': 1.00, 'pf': 0.54, 'pnl': -195.47, 'roi': -39.09, 'ruin': 37.15, 'dd': 41.8},
    {'v': 'V9', 'focus': 'Dual-Horizon Scale-Out (50% @ 1.2, 50% @ 2.2 ATR)', 'trades': 72, 'w': 44, 'l': 28, 'wr': 61.11, 'avg_w': 22.21, 'avg_l': -32.21, 'payoff': 0.69, 'pf': 1.08, 'pnl': 75.49, 'roi': 15.10, 'ruin': 23.17, 'dd': 24.3},
    {'v': 'V10', 'focus': 'Premature Breakeven Trail (+1.0 ATR Scratch Flaw)', 'trades': 52, 'w': 10, 'l': 42, 'wr': 19.23, 'avg_w': 16.94, 'avg_l': -9.16, 'payoff': 1.85, 'pf': 0.44, 'pnl': -215.37, 'roi': -43.07, 'ruin': 38.67, 'dd': 45.1}
]

base_dir = r'C:\Users\admin\.gemini\antigravity\scratch\new_trade\trading_system'

v_focus = {
    'V11': 'Fixed ₹2,500 Position Sizing + Pre-Trade MC (EV > 0)',
    'V12': 'Dynamic Volatility Risk Budgeting (₹20 Floor) + Fat-Tail MC',
    'V13': 'ADX Trend Gate (ADX >= 25.0) + Directional Filter',
    'V14': 'Tiered Milestone Capital Scaling (Ratchet Compounding)',
    'V15': 'Dual-Target Scale-Out (1.6 & 2.6 ATR, BE Scratch Flaw)',
    'V16': 'Institutional RVOL Gate (RVOL >= 1.80 Climax Trap)',
    'V17': 'Time-of-Day Microstructure Window (09:45-14:15 IST)',
    'V18': 'Sector Dispersion & Alpha Leader Selection (Max 1/Sector)',
    'V19': 'Half-Kelly Conviction-Weighted Risk Budgeting',
    'V20': 'Quant Apex Master System (Full Institutional Synthesis)'
}

rows = list(versions_meta)

for i in range(11, 21):
    v = f'V{i}'
    mf = os.path.join(base_dir, v, 'manifest.json')
    if os.path.exists(mf):
        with open(mf, 'r') as f:
            m = json.load(f)
            met = m['metrics']
            rows.append({
                'v': v,
                'focus': v_focus.get(v, ''),
                'trades': met.get('trades'),
                'w': met.get('wins'),
                'l': met.get('losses'),
                'wr': met.get('win_rate'),
                'avg_w': met.get('avg_win'),
                'avg_l': met.get('avg_loss'),
                'payoff': met.get('payoff_ratio'),
                'pf': met.get('profit_factor'),
                'pnl': met.get('net_pnl'),
                'roi': met.get('roi_pct'),
                'ruin': met.get('ruin_prob_50pct'),
                'dd': met.get('expected_max_dd')
            })

for r in rows:
    print(json.dumps(r))
