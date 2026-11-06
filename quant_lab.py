import sys
import os
import pandas as pd
import numpy as np

# Ensure root is in sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_fetcher import DataFetcher
from src.backtester import BacktestEngine
from src.strategy import calculate_atr, calculate_rsi

def test_sniper_pullback(df, ema_regime=100, rsi_period=7, rsi_buy_thresh=38, tp_mult=2.5, sl_mult=1.5):
    d = df.copy()
    d['ema_regime'] = d['close'].ewm(span=ema_regime, adjust=False).mean()
    d['ema_fast'] = d['close'].ewm(span=9, adjust=False).mean()
    d['rsi'] = calculate_rsi(d['close'], period=rsi_period)
    d['atr'] = calculate_atr(d, period=14)
    
    # Bullish regime: close > ema_regime
    bull_regime = d['close'] > d['ema_regime']
    
    # Dip condition: RSI dropped below threshold recently (within last 3 bars)
    rsi_dip = (d['rsi'] < rsi_buy_thresh) | (d['rsi'].shift(1) < rsi_buy_thresh) | (d['rsi'].shift(2) < rsi_buy_thresh)
    
    # Reversal trigger: close crosses above ema_fast and bullish candle
    reversal = (d['close'] > d['ema_fast']) & (d['close'].shift(1) <= d['ema_fast'].shift(1)) & (d['close'] > d['open'])
    
    raw_buy = bull_regime & rsi_dip & reversal
    
    # State debounce
    buy_signals = np.zeros(len(d), dtype=bool)
    sell_signals = np.zeros(len(d), dtype=bool)
    in_pos = False
    
    for i in range(len(d)):
        if raw_buy.iloc[i] and not in_pos:
            buy_signals[i] = True
            in_pos = True
        elif in_pos and d['close'].iloc[i] < d['ema_regime'].iloc[i]:
            sell_signals[i] = True
            in_pos = False
            
    d['buy_signal'] = buy_signals
    d['sell_signal'] = sell_signals
    d['long_sl'] = d['close'] - (d['atr'] * sl_mult)
    d['long_tp'] = d['close'] + (d['atr'] * tp_mult)
    
    engine = BacktestEngine(initial_capital=100000, position_pct=100, commission_pct=0.03)
    res = engine.run(d, use_tp_sl=True, trade_direction='Long Only')
    return res['metrics']

def main():
    fetcher = DataFetcher(data_dir="data")
    symbols = ["SUZLON", "PCJEWELLER"]
    timeframes = ["15m", "30m", "1h", "1d"]
    
    print("=" * 85)
    print("  QUANT LAB: SNIPER PULLBACK ENGINE EVALUATION")
    print("=" * 85)
    print(f"{'Symbol':<12} | {'TF':<5} | {'Trades':<8} | {'Win Rate':<10} | {'Net Return':<12} | {'Profit Factor':<14} | {'Max DD':<8}")
    print("-" * 85)
    
    for sym in symbols:
        for tf in timeframes:
            data = fetcher.get_data(sym, tf)
            if data is not None and len(data) > 100:
                m = test_sniper_pullback(data)
                print(f"{sym:<12} | {tf:<5} | {m['total_trades']:<8} | {m['win_rate']:>6.1f}%   | {m['net_profit_pct']:>9.2f}% | {m['profit_factor']:>12.2f}  | {m['max_drawdown_pct']:>6.2f}%")

if __name__ == "__main__":
    main()
