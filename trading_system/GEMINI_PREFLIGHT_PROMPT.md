# BHAVISHYA — PRE-FLIGHT PROMPT FOR GEMINI
# Yeh prompt paste karo Gemini ko koi bhi naya V banane se PEHLE.

---

## MANDATORY RULES BEFORE BUILDING ANY NEW VERSION

You are building a quantitative trading system version (VXX). Before writing a single line of code or claiming any result, you MUST follow ALL of these rules without exception:

### STRICT RULES (Non-Negotiable)

1. **No look-ahead bias** — Features must only use data available at bar close time `t`. No shift(-1), no future prices, no forward-fill from future bars. VWAP must use shift(1).

2. **No data leakage** — Training folds must NEVER contain data from test fold dates. Enforce 2-day embargo between end of train and start of test in all walk-forward splits.

3. **No hyperparameter tuning on the final test set** — If you use Optuna or any optimizer, it MUST run on a DIFFERENT date range than the final backtest. Example: tune on Fold 3, test on Fold 4. Never tune and test on the same OOS dates.

4. **No fake trades, no fabricated results** — Every trade must come from real bar data in `universe_78_15m.pkl`. No synthetic prices, no manually crafted PnL.

5. **No curve fitting** — Do not adjust parameters, thresholds, or architecture based on what you see in the backtest results. Decide parameters BEFORE running the backtest, not after.

6. **Realistic execution** — Apply full Dhan MIS brokerage (0.03% capped at Rs 20/leg), STT (0.025%), exchange fees, SEBI, stamp duty, GST. No zero-cost trades.

7. **No cherry-picking** — Report ALL trades that occurred. Do not filter or exclude losing trades from the reported results.

8. **Sanity check before claiming results** — If Win Rate > 85% or Profit Factor > 15 on fewer than 25 trades, treat it as a red flag and explicitly audit why before reporting.

### MANDATORY FINAL STEP — Run Integrity Gate

After building and running VXX, ALWAYS run this before reporting results:

```
python C:/Users/admin/.gemini/antigravity/scratch/new_trade/trading_system/integrity_gate.py --version VXX
```

If the gate reports ANY FAIL, fix the issue first. Do NOT report results until the gate shows 0 FAIL.
If the gate shows WARN, explicitly disclose the warning in the results summary.

### WHAT TO REPORT

Report ONLY what the code actually outputs. No rounding up. No paraphrasing in a positive direction.
State clearly:
- Total trades, Wins, Losses
- Win Rate, Average Win, Average Loss
- Payoff Ratio, Profit Factor
- Net PnL, Net ROI on Rs 500 capital
- Total Dhan charges paid
- 10,000-sim ruin probability
- Integrity Gate result: PASS / WARN / FAIL

**If any number looks too good — investigate it honestly and report what you find, even if it is bad news.**

---
