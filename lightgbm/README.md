# 🚀 BHAVISHYA LightGBM — Model Summary & Results

## 📊 Overview
* **Model Type:** Microsoft LightGBM Multi-Class Classifier (3 Classes: 0=DOWN, 1=FLAT, 2=UP)
* **Optimization:** 20 Optuna Hyperparameter Trials
* **Evaluation Framework:** 3-Fold Walk-Forward Cross-Validation (30 days train / 10 days test)
* **Data:** NSE 5-minute candles (60 days), 44 liquid stocks + Nifty Index
* **Trading Capital:** ₹500 (with 5x Groww Margin = ₹2,500 purchasing power)

---

## 🏆 Best Optuna Hyperparameters (Trial 18)
```json
{
  "objective": "multiclass",
  "num_class": 3,
  "n_estimators": 570,
  "learning_rate": 0.08930274585407648,
  "max_depth": 10,
  "num_leaves": 111,
  "min_child_samples": 63,
  "subsample": 0.6862387779526721,
  "colsample_bytree": 0.9989300688203024,
  "reg_alpha": 0.09250145386280544,
  "reg_lambda": 4.498682692251446,
  "class_weight": "balanced"
}
```

---

## 📈 Performance Metrics

| Metric | Result |
|---|---|
| **Raw 3-Class Validation Score** | 26.68% (unfiltered multi-class) |
| **Walk-Forward Fold 1 Hit Rate** | 59.2% (7,188 signals) |
| **Walk-Forward Fold 2 Hit Rate** | 59.5% (9,730 signals) |
| **Walk-Forward Fold 3 Hit Rate** | 58.4% (12,511 signals) |
| **Average Directional Accuracy** | **59.0%** |
| **Total Executed Trades** | 60 (Avg 2.0 trades/day) |
| **Win Rate** | **55.0%** (33 Wins / 27 Losses) |
| **Total Net P&L** | **+₹193.61** |
| **Total Groww Charges Paid** | ₹223.26 |
| **Net Account Growth** | **+38.72%** (₹500 → ₹693.61) |
| **Profit Factor** | **1.68** |

---

## 📁 Files Included
* `bhavishya_lgbm.txt` — Trained LightGBM model text representation
* `bhavishya_features.json` — 37 engineered features list
* `bhavishya_trades.csv` — Full trade ledger of all 60 executed trades
* `bhavishya_results.png` — Equity growth curve and feature importance plot
* `colab_training_script.py` — Complete Google Colab runnable script
* `parameters.json` — Exact configuration and tuned parameters
