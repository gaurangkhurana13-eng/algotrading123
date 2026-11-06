# Tri-Engine Autonomous Live Cloud Forward-Tester 🔱

Autonomous multi-strategy quantitative trading scanner running on **GitHub Actions Cloud Runner**.  
Zero laptop running required. 100% Free real-time NSE intraday streaming and instant Telegram alerts.

---

## 🔱 Tri-Engine Parallel Architecture (Strictly Compartmentalized)

Three independent engines operate in strict isolation. State dictionaries, position tracking, timeframes, and trigger conditions never overlap or collide:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       TRI-ENGINE CLOUD RUNNER LOOP                          │
├───────────────────────────────┬─────────────────────────────┬───────────────┤
│ 🏛️ Bhavishya V43 Titan (15M) │ 🟢 Predator V3 Bull (1M)    │ 🔴 Bear (1M)  │
│ 30 Certified Core Stocks      │ 5 High-Beta Long Stocks     │ 3 Short Stocks│
│ Institutional MA20 / MACD     │ 30m ORB High Breakout       │ 30m ORB Low   │
│ Asymmetric Ratchet Trailing   │ 1.2% SL | 2.52% TP (2.1R)   │ 1.2% SL | 2.5%│
└───────────────────────────────┴─────────────────────────────┴───────────────┘
```

---

### 1️⃣ 🏛️ Engine 1: Bhavishya V43 Titan (15-Minute Institutional)
* **Timeframe:** 15-Minute
* **Universe:** Certified 30 Stocks (`CENTRALBK`, `DELTACORP`, `ELECTCAST`, `GSFC`, `IDEA`, `IFCI`, `IOB`, `IRCON`, `J&KBANK`, `JYOTHYLAB`, `MANALIPETC`, `MTNL`, `NBCC`, `NCC`, `NIACL`, `NMDC`, `PCJEWELLER`, `PNB`, `RBLBANK`, `RCF`, `RENUKA`, `RITES`, `RPOWER`, `SUZLON`, `SWSOLAR`, `TATASTEEL`, `TEXRAIL`, `TTML`, `UTKARSHBNK`, `ZEEL`).
* **Macro Filter:** `Close > EMA50` & `Close > EMA110` & `Close > VWAP`.
* **Dual Alpha Triggers:**
  1. **Institutional MA20 Reclaim:** `Low[1] < EMA20[1]` & `Close[1] <= EMA20[1]` & `Close > EMA20` & `RVOL >= 1.25` & `RSI7 > 50` & `MACD > Signal`.
  2. **Core MACD Momentum Cross:** `MACD Cross Up > 0` & `Close > EMA20` & `RVOL >= 1.30` & `RSI7 > 50`.
* **Asymmetric Ratchet Engine (Fortress Safety):**
  * **Initial Hard SL:** Entry Price `-0.90 ATR`
  * **Breakeven Lock:** At `+0.45 ATR MFE`, SL ratchets to `Entry + 0.05 ATR` (Zero Risk).
  * **Runner Trailing:** At `+0.85 ATR MFE`, SL ratchets to `Entry + 0.40 ATR` (Guaranteed Profit).
  * **Mandatory EOD Cutoff:** `15:15 IST`.

---

### 2️⃣ 🟢 Engine 2: Predator V3 Bull Squad (1-Minute Long Sniper)
* **Timeframe:** 1-Minute
* **Universe:** 5 High-Beta Rocket Stocks:
  * `PCJEWELLER`, `TTML`, `SOUTHBANK`, `WEBELSOLAR`, `SJVN`
* **Filters:**
  * 30-Min Opening Window (09:15 - 09:45 IST) defines `ORB High`.
  * `Close > VWAP` & `Close > Daily 20 EMA` (Previous Day's Daily EMA20).
  * Volume Spike: `Volume > 1.05 * 20 SMA(Volume)`.
* **Dual Entry Setups:**
  1. **Morning ORB High Breakout:** `CrossOver(Close, ORB High)` after 09:45 IST.
  2. **Afternoon Momentum Push:** Breakout of recent 15-minute consolidation high.
* **Risk & Profit Engine:**
  * **Hard SL:** `Entry - 1.2%`
  * **Take Profit:** `Entry + 2.52%` (2.1R Reward)
  * **Trailing Lock:** Once gain reaches `+1.2%`, trailing stop activates with `0.6%` offset behind peak.
* **Official Pine Script:** [`strategies/V3/v3_long_bull_squad.pine`](strategies/V3/v3_long_bull_squad.pine)

---

### 3️⃣ 🔴 Engine 3: Predator V3 Bear Squad (1-Minute Short Sniper)
* **Timeframe:** 1-Minute
* **Universe:** 3 Heavy Distribution Stocks:
  * `NMDC`, `IDFCFIRSTB`, `SJVN`
* **Filters:**
  * 30-Min Opening Window (09:15 - 09:45 IST) defines `ORB Low`.
  * `Close < VWAP` & `Close < Daily 20 EMA` (Previous Day's Daily EMA20).
  * Volume Spike: `Volume > 1.05 * 20 SMA(Volume)`.
* **Dual Short Setups:**
  1. **Morning ORB Low Breakdown:** `CrossUnder(Close, ORB Low)` after 09:45 IST.
  2. **Afternoon Breakdown:** Breakdown of recent 15-minute support low.
* **Risk & Profit Engine:**
  * **Hard SL:** `Entry + 1.2%` (Above Entry)
  * **Take Profit:** `Entry - 2.52%` Drop (2.1R Reward)
  * **Trailing Lock:** Once drop reaches `-1.2%`, trailing stop activates with `0.6%` offset above trough.
* **Official Pine Script:** [`strategies/V3/v3_short_bear_squad.pine`](strategies/V3/v3_short_bear_squad.pine)

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
* **Automatic Runtime:** Automatically starts at 09:15 IST and scans every 50 seconds until 15:30 IST market close.
* **Manual Trigger:** Can also be triggered anytime via the **Actions** tab -> **Run workflow**.

---

## 📂 Repository Structure

```
├── .github/workflows/
│   └── live_scanner.yml               # Tri-Engine Cloud Runner Workflow
├── scripts/
│   ├── live_telegram_forward_tester.py # Tri-Engine Live Cloud Dispatcher
│   └── multi_slot_sim_engine.py       # Multi-Slot Backtest Simulator
├── strategies/
│   ├── V3/
│   │   ├── v3_long_bull_squad.pine    # Official V3 Long Pine Script (100% Untampered)
│   │   └── v3_short_bear_squad.pine   # Official V3 Short Pine Script (100% Untampered)
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
