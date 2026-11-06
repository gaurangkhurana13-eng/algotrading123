# Bhavishya Autonomous Live Cloud Forward-Tester 🚀

Autonomous real-time quantitative trading scanner running on **GitHub Actions Cloud Runner**. 
Zero laptop running required. 100% Free real-time NSE intraday streaming and instant Telegram alerts.

---

## ⚡ Core Strategy Engines Monitored

1. **PC Jeweller 1m Live Predator V3 (Micro Sniper)**
   - **Timeframe:** 1-Minute
   - **Logic:** 30-Minute Opening Range Breakout (ORB) High break + VWAP confirmation + Volume surge.
   - **Risk Management:** 1.2% Stop Loss, 2.1R Fixed Target, 5x Intraday MIS leverage.

2. **Bhavishya V43 Titan (15m Institutional Fortress)**
   - **Timeframe:** 15-Minute
   - **Logic:** MA20 Mean-Reversion Reclaim + EMA50 Macro Trend + MACD Signal Confirmation.
   - **Risk Management:** 0.90 ATR Initial Stop Loss + Asymmetric Ratchet (+0.45 ATR MFE locks Breakeven, +0.85 ATR locks +0.40 ATR profit).

---

## 🛠️ GitHub Cloud Runner Configuration

### 1. Add Repository Secrets
Navigate to **Settings** -> **Secrets and variables** -> **Actions** -> **New repository secret**:

| Secret Name | Value |
|---|---|
| `TELEGRAM_BOT_TOKEN` | Your Telegram Bot Token (`9129905672:AAGVMEPQ...`) |
| `TELEGRAM_CHAT_ID` | Your Telegram User ID (`7022758119`) |

### 2. Autonomous Daily Schedule
The workflow runs automatically via GitHub Actions Cron:
- **Cron:** `45 3 * * 1-5` (03:45 UTC = 09:15 AM IST, Monday to Friday)
- **Runtime:** Scans continuously until 15:30 IST market close.
- **Manual Trigger:** Can also be manually dispatched anytime via the **Actions** tab -> **Run workflow**.

---

## 💻 Local Testing

To test connection to Telegram:
```bash
python scripts/live_telegram_forward_tester.py --test
```

To run locally in live mode:
```bash
python scripts/live_telegram_forward_tester.py --live
```
