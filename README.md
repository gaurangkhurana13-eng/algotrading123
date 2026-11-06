# Bhavishya V43 Titan — Autonomous Live Cloud Forward-Tester 🏛️

Autonomous institutional quantitative trading scanner running on **GitHub Actions Cloud Runner**.
Zero laptop running required. 100% Free real-time NSE intraday streaming and instant Telegram alerts.

---

## ⚡ Core Strategy: Bhavishya V43 Titan (15-Minute Institutional)

* **Timeframe:** 15-Minute
* **Universe:** Certified 30 Stocks (`CENTRALBK`, `DELTACORP`, `ELECTCAST`, `GSFC`, `IDEA`, `IFCI`, `IOB`, `IRCON`, `J&KBANK`, `JYOTHYLAB`, `MANALIPETC`, `MTNL`, `NBCC`, `NCC`, `NIACL`, `NMDC`, `PCJEWELLER`, `PNB`, `RBLBANK`, `RCF`, `RENUKA`, `RITES`, `RPOWER`, `SUZLON`, `SWSOLAR`, `TATASTEEL`, `TEXRAIL`, `TTML`, `UTKARSHBNK`, `ZEEL`).
* **Macro Filter:** `Close > EMA50` & `Close > EMA110` & `Close > VWAP` (Ensures trading strictly with macro institutional flow).
* **Dual Alpha Triggers:**
  1. **Institutional MA20 Reclaim:** `Low[1] < EMA20[1]` & `Close[1] <= EMA20[1]` & `Close > EMA20` & `RVOL >= 1.25` & `RSI7 > 50` & `MACD > Signal`.
  2. **Core MACD Momentum Cross:** `MACD Cross Up > 0` & `Close > EMA20` & `RVOL >= 1.30` & `RSI7 > 50`.
* **Asymmetric Ratchet Engine (Fortress Safety):**
  * **Initial Hard SL:** Entry Price `-0.90 ATR`
  * **Breakeven Lock Target:** At `+0.45 ATR MFE`, Stop Loss automatically ratchets to `Entry + 0.05 ATR` (Zero Risk Trade).
  * **Runner Trailing:** At `+0.85 ATR MFE`, Stop Loss ratchets to `Entry + 0.40 ATR` (Guaranteed Profit).
  * **Mandatory EOD Cutoff:** All positions closed at `15:15 IST` before market close.

---

## 🛠️ GitHub Cloud Runner Configuration

### 1. Repository Secrets
Under **Settings** -> **Secrets and variables** -> **Actions**:

| Secret Name | Description |
|---|---|
| `TELEGRAM_BOT_TOKEN` | Telegram Bot Token (`9129905672:AAGVMEPQ...`) |
| `TELEGRAM_CHAT_ID` | Telegram User Chat ID (`7022758119`) |

### 2. Autonomous Daily Schedule
* **Cron:** `45 3 * * 1-5` (03:45 UTC = 09:15 AM IST, Monday through Friday)
* **Automatic Runtime:** Automatically starts at 09:15 IST and scans every 55 seconds until 15:30 IST market close.
* **Manual Trigger:** Can also be triggered anytime via the **Actions** tab -> **Run workflow**.

---

## 📂 Repository Structure

```
├── .github/workflows/
│   └── live_scanner.yml               # Cloud Runner Workflow
├── scripts/
│   ├── live_telegram_forward_tester.py # Live V43 Cloud Dispatcher
│   └── multi_slot_sim_engine.py       # Core Multi-Slot Simulation & Universe
├── strategies/
│   └── V43/
│       ├── run_v43.py                 # Master Backtest Engine (10-Year Verified)
│       ├── v43_tradingview_ready.pine # Pine Script v5 Strategy
│       ├── v43.pine                   # Core Pine Script Logic
│       ├── trades_intraday.csv        # Certified Trade Log
│       └── results.txt                # Audit Verification Metrics
├── framework.py                       # Quant Calculation Engine & Dhan Fees
├── requirements.txt                   # Minimal Cloud Dependencies
├── .gitignore                         # Clean lightweight Git exclusions
└── README.md                          # Production Documentation
```
