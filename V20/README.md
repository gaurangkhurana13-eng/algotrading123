# Bhavishya Strategy V20: Full Institutional Synthesis

Bhavishya V20 represents the production synthesis of early loss mitigation, asymmetric 4-phase hyper-trailing, and dynamic equity-tiered risk rules on Indian Equities (NSE 15-Minute Bars).

---

## 1. Directory Structure

```text
V20/
├── backtests/
│   └── run_v20.py              # Main walk-forward backtest execution script
├── config/
│   └── config.json             # Immutable parameters and capital risk schedule
├── data_manifest/
│   └── manifest.json           # SHA-256 hashes & sizes of real historical pickle data
├── logs/
│   └── gate_v2_audit.log       # Full execution log from Gate V2 auditor
├── monte_carlo/
│   └── mc_results.json         # 10,000-iteration bootstrap simulation statistics
├── reports/
│   └── V20_MASTER_AUDIT_REPORT.md  # Forensic comparison report against V19
├── src/
│   ├── engine.py               # Complete standalone execution engine
│   └── v20.pine                # TradingView Pine Script v6 strategy implementation
├── tests/
│   └── test_v20.py             # Pytest automated test suite (5 passing unit tests)
├── trades/
│   ├── v20_trades.csv          # Reconciled trade ledger (167 audited trades)
│   └── v20_trades_swing.csv    # Swing ledger (zero trades, 100% intraday)
└── README.md                   # System documentation & reproduction commands
```

---

## 2. Core Architecture

1. **Universe:** Top 10 NSE Alpha Leaders (`PCJEWELLER`, `RPOWER`, `DELTACORP`, `RITES`, `IDEA`, `MTNL`, `ZEEL`, `UTKARSHBNK`, `IFCI`, `NCC`).
2. **Preserved V19 Entry Signal:**
   $$\text{Signal} = \text{MACD} > \text{Signal} > 0 \land \text{RSI}(7) > 50 \land \text{ADX}(10) > 18 \land +\text{DI} > -\text{DI} \land \text{Close} > \text{EMA}(50) > \text{EMA}(110) \land \text{Close} > \text{VWAP}(t-1)$$
3. **Early Loss Cut Pre-emption:**
   - **Bar-1 Micro-Compression:** If $\text{bars\_held} == 1 \land \text{MFE} < 0.30\times\text{ATR} \land \text{Close} \le \text{Entry} \land \text{Volume} < 0.70\times\text{Entry Volume}$, tighten stop to $\text{Entry} - 0.35\times\text{ATR}$.
   - **Bar-2 Hard Stall Abort:** If $\text{bars\_held} == 2 \land \max(C_1, C_2) \le \text{Entry} \land \text{MFE} < 0.40\times\text{ATR} \land (\text{CLV} < 0.35 \lor \text{Volume} < 0.65\times\text{Entry Volume})$, exit immediately at realistic market price.
   - **Bar-3 Legacy Scratch:** If $\text{bars\_held} \ge 3 \land \text{MFE} < 0.40\times\text{ATR} \land \text{Close} < \text{Entry} \land \text{RSI}(7) < 46.0$, exit.
4. **4-Phase Asymmetric Hyper-Trailing:**
   - **Phase 1 (Ratchet):** $\text{MFE} \ge 0.85\times\text{ATR} \implies \text{Stop} = \max(\text{Prev}, \text{Entry} + 0.22\times\text{ATR})$.
   - **Phase 2 (Structural):** $1.65 \le \text{MFE}/\text{ATR} < 2.50 \implies \text{Stop} = \max(\text{Prev}, \text{Entry} + 0.80\times\text{ATR}, \text{High} - 0.65\times\text{ATR})$.
   - **Phase 3 (Hyper-Trail):** $\text{MFE} \ge 2.50\times\text{ATR} \implies \text{Stop} = \max(\text{Prev}, \text{Entry} + 1.65\times\text{ATR}, \min(L_1, L_2, L_3) - 0.15\times\text{ATR}, \text{High} - 1.10\times\text{ATR})$.
   - **Phase 4 (Climax Override):** $\text{MFE} \ge 3.80\times\text{ATR} \lor (\text{MFE} \ge 2.80\times\text{ATR} \land \text{RSI} \ge 82 \land \text{CLV} < 0.40) \implies \text{Stop} = \max(\text{Prev}, \text{High} - 0.38\times\text{ATR})$.
5. **Capital-Based Risk Sizing:**
   - ₹500–₹1,000: Max loss ₹40, Max 1 slot.
   - ₹1,000–₹2,000: Max loss ₹110, Max 1 slot.
   - Above ₹2,000: Max loss 3% of realized equity, Max 2 slots (46% buying power per slot, 8% buffer).

---

## 3. Local Reproduction Commands

### Run Strategy Backtest:
```powershell
& "C:\Users\admin\.gemini\antigravity\scratch\new_trade\.venv\Scripts\python.exe" "C:\Users\admin\.gemini\antigravity\scratch\new_trade\V20\backtests\run_v20.py"
```

### Run Gate V2 Forensic Institutional Audit:
```powershell
& "C:\Users\admin\.gemini\antigravity\scratch\new_trade\.venv\Scripts\python.exe" "C:\Users\admin\.gemini\antigravity\scratch\new_trade\strategies\strategy_evaluate_gate_v2.py" V20
```

### Run Automated Unit Test Suite:
```powershell
& "C:\Users\admin\.gemini\antigravity\scratch\new_trade\.venv\Scripts\python.exe" -m pytest "C:\Users\admin\.gemini\antigravity\scratch\new_trade\V20\tests\test_v20.py"
```
