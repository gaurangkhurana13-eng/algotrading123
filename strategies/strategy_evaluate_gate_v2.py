#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
strategy_evaluate_gate.py  --  Independent Forensic Quantitative Auditor
=========================================================================

Purpose
-------
A standalone, trust-nothing audit gate for rule-based intraday strategies on
Indian equities (NSE, 15-minute bars, Dhan broker, MIS 5x).  It re-derives
everything from the raw artefacts in a strategy version folder and never
imports or trusts the strategy's own code, reports or helper modules.

Usage
-----
    python strategy_evaluate_gate.py V5                 # audit one version
    python strategy_evaluate_gate.py V5 --strict        # WARN also fails the gate
    python strategy_evaluate_gate.py --all              # audit every V<n> folder
    python strategy_evaluate_gate.py --all path/to/strategies
    python strategy_evaluate_gate.py V5 --root scratch/new_trade/strategies \
                                        --data-dir scratch/new_trade/data --json out.json

Expected folder layout (names are matched case-insensitively):
    strategies/V5/v5.pine
    strategies/V5/results.txt           (has an "Intraday" and a "Swing" section)
    strategies/V5/trades_intraday.csv
    strategies/V5/trades_swing.csv
    strategies/V5/run_v5.py
    data/<symbol>_15m_1y.pkl            (raw OHLCV; DataFrame with datetime index/column)

The 10 checks
-------------
 1 File existence & CSV schema            6 100% intraday square-off
 2 Raw-bar grounding & price realism      7 CSV vs results.txt reconciliation
 3 Lookahead & same-bar execution audit   8 Pine Script audit
 4 Dhan charges + net-PnL recalculation   9 R-multiple plausibility / compounding alarms
 5 Dynamic margin & risk cap R(E_k)      10 Visual report + exit code

Exit code: 0 only if no [FAIL] (and, with --strict, no [WARN]); otherwise 1.
The auditor FAILS CLOSED: an exception inside any check is reported as [FAIL].

Design notes / assumptions (read these - they decide what is flagged)
---------------------------------------------------------------------
* Win  = net_pnl > 0 (after charges).  Loss = net_pnl < 0.  Break-even is neither.
* Trade direction is taken from a side/direction column if present, otherwise it
  is inferred per trade from which direction reproduces net_pnl from the prices.
* Two price conventions are tried when reconciling net_pnl:
    A) prices in the CSV already include the 0.05% slippage  (default)
    B) prices in the CSV are raw; slippage is applied by the backtester on top
  The convention that reproduces the CSV best is used (and reported).
* Statutory costs follow the schedule in the task brief.  Swing (delivery) uses
  STT 0.1% on buy+sell; stamp duty is kept at 0.003% on the buy leg per brief.
* pickle files can execute code on load - only point this at your own data.

Dynamic-equity rules (v2 - compounding-aware, still reality-only)
------------------------------------------------------------------
* Trades of BOTH books are merged chronologically into one shared account.
  E_k = CAPITAL + sum(net_pnl of trades whose exit is strictly before entry_k).
  Same-day realized MIS profit is therefore usable immediately; open positions'
  unrealized P&L is ignored (closed trades only).
* Risk budget   R(E_k) = min(3.1% * E_k, Rs 250), halved while E_k is >15% below
  its high-water mark.  Loss > R+0.50 is FAIL, (R, R+0.50] is WARN.
* Margin (shared pool): margin in use at entry_k + the new trade's margin must be
  <= 95% of E_k.  Intraday margin = notional / min(5, symbol leverage); delivery
  (swing) locks 100% cash, and its sale proceeds stay locked T+1 business day.
* One trade = one row.  Sliced orders, repeated parent ids and overlapping
  positions in one symbol are FAIL.  No slicing is ever needed because R scales.
* Participation: shares <= 5% of the fill bar's volume (needs a volume column).
* Timestamps: trade times are bar-START labels (use --ts-convention end for
  bar-end labels).  The bar pickle's own label convention is auto-detected.
* Check 9 works on R-multiples (net_pnl / planned stop risk, or / R(E_k) when the
  CSV has no stop column) and on growth per trade, never on raw Rs.
* The CLI can only tighten the constitution; loosening any value is a FAIL.
* Optional: --leverage-table symbols.csv (columns symbol,leverage) enforces
  Dhan's real per-stock MIS leverage; a symbol missing from the table is FAIL.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import sys
import tokenize
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# =============================================================================
# 0. CONFIGURATION  (edit here, not inside the checks)
# =============================================================================
# ---- THE CONSTITUTION: dynamic, equity-derived limits (reality-first) --------
# Every limit below is evaluated per trade k against the REALIZED equity E_k that
# the gate itself reconstructs from verified net_pnl of trades CLOSED BEFORE
# trade k's entry (strictly earlier timestamp).  Nothing here is read from the
# strategy's own equity column.  The CLI may only TIGHTEN these values; loosening
# any of them from the command line is a [FAIL].  Loosen only by editing this
# block (the gate prints its own SHA-256 so an orchestrator can pin it).
CAPITAL = 500.0                    # starting equity E_0 (Rs)
LEVERAGE = 5.0                     # max MIS leverage; per-symbol table may only LOWER it
MARGIN_BUFFER = 0.95               # total margin in use (incl. new trade) <= 95% of E_k
RISK_PCT = 0.090                   # Phase 1 Nano-Capital risk cap = 9.0% of E_k (Dhan 5x MIS natural limit)
RISK_CEILING = 260.0               # Phase 2 Institutional hard ceiling = Rs 260
DD_THROTTLE_TRIGGER = 0.15         # if realized equity is >15% below its high-water mark ...
DD_THROTTLE_FACTOR = 0.50          # ... the risk budget is halved
RISK_HARD_EXTRA = 0.50             # loss in (R_k, R_k+0.50] = WARN, > R_k+0.50 = FAIL
MAX_PARTICIPATION = 0.05           # shares <= 5% of the fill bar's traded volume
SWING_SETTLE_BDAYS = 1             # delivery sale proceeds free up T+N business days after exit
TS_CONVENTION = "start"            # trade timestamps are bar-START labels ("end" = bar-end labels)
CONSTITUTION = dict(CAPITAL=CAPITAL, LEVERAGE=LEVERAGE, MARGIN_BUFFER=MARGIN_BUFFER, RISK_PCT=RISK_PCT,
                    RISK_CEILING=RISK_CEILING, DD_THROTTLE_TRIGGER=DD_THROTTLE_TRIGGER,
                    DD_THROTTLE_FACTOR=DD_THROTTLE_FACTOR, RISK_HARD_EXTRA=RISK_HARD_EXTRA,
                    MAX_PARTICIPATION=MAX_PARTICIPATION, SWING_SETTLE_BDAYS=SWING_SETTLE_BDAYS,
                    TS_CONVENTION=TS_CONVENTION)
# direction in which a value is "looser" (+1: bigger is looser, -1: smaller is looser)
LOOSER_IF = dict(LEVERAGE=+1, MARGIN_BUFFER=+1, RISK_PCT=+1, RISK_CEILING=+1, DD_THROTTLE_TRIGGER=+1,
                 DD_THROTTLE_FACTOR=+1, RISK_HARD_EXTRA=+1, MAX_PARTICIPATION=+1, SWING_SETTLE_BDAYS=-1)

# Check 9 (R-multiple / compounding plausibility)
AVG_LOSS_MIN_R = 0.05              # average loser smaller than 0.05R = stops are not real stops
GEO_GROWTH_WARN = 0.03             # geometric mean growth per trade (net_pnl / E_k)
GEO_GROWTH_FAIL = 0.05
GROWTH_MULT_WARN = 10.0            # >10x total growth needs hidden-holdout evidence
TAIL_SHARE_WARN = 0.40             # top-3 trades' share of total log-growth
TOP3_R_WARN = 0.60                 # top-3 trades' share of gross positive R
SLIPPAGE = 0.0005                  # 0.05% per leg

SQUAREOFF_POLICY = (15, 10)        # strategy rule: flat by 15:10
SQUAREOFF_HARD = (15, 15)          # MIS auto-square-off cut-off: exit after this is a FAIL
SESSION_OPEN = (9, 15)
SESSION_CLOSE = (15, 30)

BROKERAGE_RATE = 0.0003            # 0.03% of turnover ...
BROKERAGE_CAP = 20.0               # ... capped at Rs 20 per order
STT_INTRADAY_SELL = 0.00025        # 0.025% on sell leg
STT_DELIVERY_BOTH = 0.001          # 0.1% on buy + sell
EXCHANGE_FEE = 0.0000345           # 0.00345% of turnover
SEBI_FEE = 10.0 / 1e7              # Rs 10 per crore
STAMP_DUTY_BUY = 0.00003           # 0.003% on buy leg
GST_RATE = 0.18                    # on brokerage + exchange + SEBI

CHARGE_TOL = 0.10                  # max per-trade |csv - recomputed| charges (Rs)
PNL_TOL = 0.10                     # max per-trade |csv net_pnl - price-derived net_pnl| (Rs)
CHARGE_RATIO_BAND = (0.0002, 0.006)  # sanity band for total charges / total turnover (size-independent)
PRICE_ENV_LO, PRICE_ENV_HI = 0.999, 1.001   # bar envelope for price grounding
NEXT_OPEN_BAND = 0.0015            # |price - next bar open| allowance

RESULT_TOL = {"trades": 0, "wins": 0, "losses": 0, "win_rate": 0.1,
              "net": 0.05, "pf": 0.02, "avg_win": 0.05, "avg_loss": 0.05}

MANDATORY_COLS = ["entry_time", "exit_time", "entry_price", "exit_price",
                  "shares", "net_pnl", "charges", "exit_reason"]
NUMERIC_COLS = ["entry_price", "exit_price", "shares", "net_pnl", "charges"]
SIDE_COLS = ["side", "direction", "dir", "position", "trade_side"]
SL_COLS = ["sl", "stop_loss", "stoploss", "sl_price", "stop_price", "stop"]
TP_COLS = ["tp", "target", "take_profit", "tp_price", "target_price"]
DEFAULT_SYMBOL = "pcjeweller"

CHECK_TITLES = {
    1: "File existence & schema",
    2: "Raw-bar grounding & price realism",
    3: "Lookahead & same-bar execution",
    4: "Dhan charges & PnL recalculation",
    5: "Dynamic margin & risk cap R(E_k)",
    6: "100% intraday square-off",
    7: "CSV vs results.txt reconciliation",
    8: "Pine Script audit",
    9: "Statistical plausibility",
}
RANK = {"PASS": 0, "WARN": 1, "FAIL": 2}


# =============================================================================
# 1. REPORT PLUMBING
# =============================================================================
class Report:
    """Collects findings (per check) and a forensic discrepancy log."""

    def __init__(self, version: str):
        self.version = version
        self.findings: List[Tuple[int, str, str]] = []
        self.discs: List[dict] = []
        self.meta: Dict[str, object] = {}
        self.files: Dict[str, str] = {}

    def add(self, check: int, status: str, msg: str) -> None:
        self.findings.append((check, status, msg))

    def disc(self, check: int, where: str, field: str, expected, actual, note: str = "") -> None:
        self.discs.append(dict(check=check, where=where, field=field,
                               expected=str(expected), actual=str(actual), note=note))

    def check_status(self, check: int) -> str:
        sts = [s for c, s, _ in self.findings if c == check]
        if not sts:
            return "FAIL"            # a check that produced nothing did not pass
        return max(sts, key=lambda s: RANK[s])

    def counts(self) -> Tuple[int, int]:
        fails = sum(1 for _, s, _ in self.findings if s == "FAIL")
        warns = sum(1 for _, s, _ in self.findings if s == "WARN")
        return fails, warns

    def verdict(self, strict: bool) -> str:
        fails, warns = self.counts()
        if fails or (strict and warns):
            return "FAIL"
        return "WARN" if warns else "PASS"


def log_mask(rep: Report, check: int, df: pd.DataFrame, mask, field: str,
             expected, actual, note: str = "", limit: int = 200) -> None:
    """Add one discrepancy-log line per flagged trade (bounded)."""
    mask = np.asarray(mask, dtype=bool)
    actual = np.asarray(actual, dtype=object)
    exp_is_arr = isinstance(expected, (np.ndarray, list, pd.Series))
    for i in np.where(mask)[0][:limit]:
        rep.disc(check, f"row {int(df['_row'].iloc[i])}", field,
                 expected[i] if exp_is_arr else expected, actual[i], note)


# =============================================================================
# 2. LOW-LEVEL HELPERS
# =============================================================================
def sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def read_text(path: Optional[Path]) -> str:
    if not path:
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def find_file(d: Path, *names: str) -> Optional[Path]:
    lower = {p.name.lower(): p for p in d.iterdir() if p.is_file()}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    return None


def parse_times(s: pd.Series) -> pd.Series:
    """Parse to naive Asia/Kolkata wall-clock timestamps."""
    try:
        out = pd.to_datetime(s, errors="coerce")
    except (ValueError, TypeError):
        out = pd.to_datetime(s, errors="coerce", utc=True)
    if not pd.api.types.is_datetime64_any_dtype(out):
        out = pd.to_datetime(s, errors="coerce", utc=True)
    if getattr(out.dt, "tz", None) is not None:
        out = out.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
    return out.astype("datetime64[ns]")


def mins(ts: pd.Series) -> np.ndarray:
    return (ts.dt.hour * 60 + ts.dt.minute).to_numpy()


def hm(t: Tuple[int, int]) -> int:
    return t[0] * 60 + t[1]


def fmt(x: float, nd: int = 2) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "nan"
    if isinstance(x, float) and math.isinf(x):
        return "inf"
    return f"{x:,.{nd}f}"


def strip_py_comments(src: str) -> str:
    """Remove '#' comments using the tokenizer (keeps strings intact)."""
    try:
        cuts = {}
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.COMMENT:
                cuts[tok.start[0]] = tok.start[1]
        lines = src.splitlines()
        return "\n".join(l[:cuts[i + 1]] if (i + 1) in cuts else l for i, l in enumerate(lines))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return "\n".join(re.sub(r"#.*$", "", l) for l in src.splitlines())


def pick_col(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    for n in names:
        if n in df.columns:
            return n
    return None


# =============================================================================
# 3. DHAN CHARGES (independent implementation)
# =============================================================================
def dhan_charges(buy_turn: np.ndarray, sell_turn: np.ndarray, product: str) -> np.ndarray:
    """Per-trade statutory + brokerage charges.  Turnover = shares * executed price."""
    total = buy_turn + sell_turn
    brokerage = (np.minimum(BROKERAGE_CAP, BROKERAGE_RATE * buy_turn)
                 + np.minimum(BROKERAGE_CAP, BROKERAGE_RATE * sell_turn))
    stt = STT_INTRADAY_SELL * sell_turn if product == "intraday" else STT_DELIVERY_BOTH * total
    exch = EXCHANGE_FEE * total
    sebi = SEBI_FEE * total
    stamp = STAMP_DUTY_BUY * buy_turn
    gst = GST_RATE * (brokerage + exch + sebi)
    return brokerage + stt + exch + sebi + stamp + gst


def eff_prices(e, x, d, mode: str):
    """Executed prices under convention A (already slipped) or B (raw + slippage)."""
    if mode == "A":
        return e, x
    return e * (1 + SLIPPAGE * d), x * (1 - SLIPPAGE * d)


def metrics(df: pd.DataFrame) -> dict:
    pnl = df["net_pnl"].to_numpy(float)
    n = len(pnl)
    wins = int((pnl > 0).sum())
    losses = int((pnl < 0).sum())
    gp = float(pnl[pnl > 0].sum())
    gl = float(-pnl[pnl < 0].sum())
    pf = gp / gl if gl > 0 else (math.inf if gp > 0 else 0.0)
    return dict(n=n, wins=wins, losses=losses,
                win_rate=100.0 * wins / n if n else 0.0,
                net=float(pnl.sum()), gross_profit=gp, gross_loss=gl, pf=pf,
                avg_win=gp / wins if wins else 0.0,
                avg_loss=-gl / losses if losses else 0.0)


def equity_curve(pnl: np.ndarray) -> np.ndarray:
    return CAPITAL + np.cumsum(pnl)


# =============================================================================
# 4. CONTEXT: loading files, preparing trades, grounding to bars
# =============================================================================
class Ctx:
    def __init__(self, v_dir: Path, args, root: Path):
        self.dir = v_dir
        self.name = v_dir.name
        self.tag = v_dir.name.lower()
        self.args = args
        self.root = root
        self.rep = Report(self.name)
        self.paths: Dict[str, Optional[Path]] = {}
        self.books: Dict[str, pd.DataFrame] = {}
        self.ground: Dict[str, pd.DataFrame] = {}
        self.bars_cache: Dict[str, Optional[pd.DataFrame]] = {}
        self.bar_notes: List[str] = []
        self.pine = ""
        self.run = ""
        self.results = ""
        self.ledger: Optional[pd.DataFrame] = None
        self.lev_table: Optional[Dict[str, float]] = None
        self.lev_err = ""
        tbl = getattr(args, "leverage_table", None)
        if tbl:
            try:
                self.lev_table = load_lev_table(Path(tbl))
            except Exception as exc:                                    # noqa: BLE001
                self.lev_err = f"{type(exc).__name__}: {exc}"

    def lev_for(self, sym: str) -> Optional[float]:
        """MIS leverage usable for a symbol: min(LEVERAGE, table value); None = not tradable on MIS."""
        if self.lev_table is None:
            return LEVERAGE
        v = self.lev_table.get(sym)
        return None if v is None or v <= 0 else min(LEVERAGE, float(v))

    # ---- raw bar data ------------------------------------------------------
    def data_dirs(self) -> List[Path]:
        c = []
        if self.args.data_dir:
            c.append(Path(self.args.data_dir))
        here = Path(__file__).resolve().parent
        c += [self.root.parent / "data", self.root.parent.parent / "data", self.root / "data",
              Path.cwd() / "data", here / "data", here.parent / "data", self.dir / "data"]
        seen, out = set(), []
        for p in c:
            if p not in seen and p.is_dir():
                seen.add(p)
                out.append(p)
        return out

    def default_symbol(self) -> str:
        m = re.search(r"([A-Za-z0-9_&\-]+?)_15m[_\w]*\.pkl", self.run)
        return m.group(1).lower() if m else DEFAULT_SYMBOL

    def get_bars(self, sym: str) -> Optional[pd.DataFrame]:
        if sym in self.bars_cache:
            return self.bars_cache[sym]
        bars = None
        key = re.sub(r"[^a-z0-9]", "", sym.lower())
        for d in self.data_dirs():
            exact = [d / f"{sym}_15m_1y.pkl", d / f"{sym.lower()}_15m_1y.pkl"]
            cands = [p for p in exact if p.exists()]
            if not cands:
                cands = [p for p in sorted(d.glob("*.pkl"))
                         if key and key in re.sub(r"[^a-z0-9]", "", p.name.lower())]
            if cands:
                try:
                    bars = load_bars(cands[0], sym, self)
                    self.rep.files[f"bars:{sym}"] = f"{cands[0].name} ({sha(cands[0])})"
                    break
                except Exception as exc:                               # noqa: BLE001
                    self.bar_notes.append(f"{cands[0]}: {type(exc).__name__}: {exc}")
        self.bars_cache[sym] = bars
        return bars


def load_lev_table(path: Path) -> Dict[str, float]:
    """symbol -> MIS leverage.  CSV (columns symbol,leverage) or JSON object {symbol: leverage}."""
    def norm(s) -> str:
        return re.sub(r"\.ns$", "", re.sub(r"^nse:", "", str(s).strip().lower()))
    if path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {norm(k): float(v) for k, v in raw.items()}
    t = pd.read_csv(path)
    t.columns = [str(c).strip().lower() for c in t.columns]
    lc = pick_col(t, ["leverage", "lev", "mis_leverage", "multiplier"])
    sc = pick_col(t, ["symbol", "ticker", "stock", "name"])
    if lc is None or sc is None:
        raise ValueError("leverage table needs columns 'symbol' and 'leverage'")
    return {norm(s): float(v) for s, v in zip(t[sc], t[lc])}


def load_bars(path: Path, sym: str, ctx: Ctx) -> pd.DataFrame:
    obj = pd.read_pickle(path)
    if isinstance(obj, dict):
        keys = {re.sub(r"[^a-z0-9]", "", str(k).lower()): k for k in obj}
        k = keys.get(re.sub(r"[^a-z0-9]", "", sym.lower()))
        if k is None:
            raise ValueError(f"symbol '{sym}' not in pickle dict (keys: {list(obj)[:5]})")
        obj = obj[k]
    df = obj.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).strip().lower() for c in df.columns]
    if isinstance(df.index, pd.DatetimeIndex):
        ts = pd.Series(df.index, index=df.index)
    else:
        col = pick_col(df, ["datetime", "date", "timestamp", "time", "dt", "date_time"])
        if col is None:
            raise ValueError("no datetime index/column in bar data")
        ts = df[col]
    ts = parse_times(ts.reset_index(drop=True))
    df = df.reset_index(drop=True)
    for c in ("open", "high", "low", "close"):
        if c not in df.columns:
            raise ValueError(f"bar data missing column '{c}'")
    df = df[["open", "high", "low", "close"] + (["volume"] if "volume" in df.columns else [])]
    df.index = pd.DatetimeIndex(ts)
    df = df[~df.index.isna()].sort_index()
    df = df[~df.index.duplicated(keep="first")].astype(float)
    # UTC-vs-IST sanity: NSE bars must sit inside 09:15-15:30 IST.
    m = df.index.hour * 60 + df.index.minute
    frac = float(((m >= hm(SESSION_OPEN)) & (m < hm(SESSION_CLOSE))).mean())
    if frac < 0.5:
        shifted = df.index + pd.Timedelta(hours=5, minutes=30)
        m2 = shifted.hour * 60 + shifted.minute
        if float(((m2 >= hm(SESSION_OPEN)) & (m2 < hm(SESSION_CLOSE))).mean()) > 0.9:
            df.index = shifted
            ctx.bar_notes.append(f"{sym}: bar timestamps looked like UTC; shifted +05:30")
    # Bar-label convention.  NSE's first 15m bar spans 09:15-09:30 and the last 15:15-15:30.
    # Start-labelled data begins 09:15; end-labelled data begins 09:30 (and ends 15:30).
    # Everything downstream uses bar-START labels, so end-labelled data is shifted back 15 min.
    day = df.index.normalize()
    stamp = pd.Series(df.index, index=df.index)
    first = stamp.groupby(day).min()
    last = stamp.groupby(day).max()
    fmin = (first.dt.hour * 60 + first.dt.minute).mode()
    lmin = (last.dt.hour * 60 + last.dt.minute).mode()
    if len(fmin) and len(lmin) and int(fmin.iloc[0]) == 9 * 60 + 30 and int(lmin.iloc[0]) == 15 * 60 + 30:
        df.index = df.index - pd.Timedelta(minutes=15)
        ctx.bar_notes.append(f"{sym}: bars are END-labelled (first bar 09:30); normalised to bar-START labels")
    return df


def parse_side(col: pd.Series) -> np.ndarray:
    v = col.astype(str).str.strip().str.lower()
    short = v.str.contains(r"short|sell|^-1$|^s$", regex=True)
    long_ = v.str.contains(r"long|buy|^\+?1$|^b$", regex=True)
    return np.where(short, -1, np.where(long_, 1, 0))


def load_trades(ctx: Ctx, key: str, path: Path) -> Optional[pd.DataFrame]:
    r = ctx.rep
    try:
        df = pd.read_csv(path)
    except Exception as exc:                                            # noqa: BLE001
        r.add(1, "FAIL", f"{path.name}: unreadable CSV ({type(exc).__name__}: {exc})")
        return None
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    missing = [c for c in MANDATORY_COLS if c not in df.columns]
    if missing:
        r.add(1, "FAIL", f"{path.name}: missing mandatory columns {missing}")
        return None
    if df.empty:
        r.add(1, "FAIL" if key == "intraday" else "WARN", f"{path.name}: zero trades")
        return None
    df["_row"] = np.arange(len(df)) + 2                              # CSV line number
    for c in NUMERIC_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["_entry"] = parse_times(df["entry_time"])
    df["_exit"] = parse_times(df["exit_time"])
    if TS_CONVENTION == "end":                  # canonicalise to bar-START labels
        df["_entry"] = df["_entry"] - pd.Timedelta(minutes=15)
        df["_exit"] = df["_exit"] - pd.Timedelta(minutes=15)
    bad = df[NUMERIC_COLS].isna().any(axis=1) | df["_entry"].isna() | df["_exit"].isna()
    if bad.any():
        r.add(1, "FAIL", f"{path.name}: {int(bad.sum())} row(s) with NaN/unparseable mandatory values")
        log_mask(r, 1, df, bad.to_numpy(), "mandatory fields", "valid number/timestamp", ["NaN/invalid"] * len(df))
        return None
    problems = []
    if (df["shares"] <= 0).any() or (df["shares"] != df["shares"].round()).any():
        problems.append("shares must be positive integers")
    if (df["entry_price"] <= 0).any() or (df["exit_price"] <= 0).any():
        problems.append("non-positive prices")
    if df["exit_reason"].astype(str).str.strip().eq("").any() or df["exit_reason"].isna().any():
        problems.append("blank exit_reason")
    if problems:
        r.add(1, "FAIL", f"{path.name}: " + "; ".join(problems))
        return None
    r.add(1, "PASS", f"{path.name}: {len(df)} trades, all {len(MANDATORY_COLS)} mandatory columns valid")
    return df


def prepare_book(ctx: Ctx, key: str, df: pd.DataFrame) -> None:
    """Infer direction + price convention; derive effective prices and expected net PnL."""
    e, x = df["entry_price"].to_numpy(float), df["exit_price"].to_numpy(float)
    sh, pnl, ch = df["shares"].to_numpy(float), df["net_pnl"].to_numpy(float), df["charges"].to_numpy(float)
    scol = pick_col(df, SIDE_COLS)
    side = parse_side(df[scol]) if scol else np.zeros(len(df), dtype=int)
    best = None
    for mode in ("A", "B"):
        D = np.vstack([np.abs((np.subtract(*eff_prices(e, x, d, mode)[::-1])) * sh * d - ch - pnl)
                       for d in (1, -1)])
        D[1, side == 1] = np.inf          # side known -> forbid the other direction
        D[0, side == -1] = np.inf
        score = float(D.min(axis=0).sum())
        if best is None or score < best[0]:
            best = (score, mode, D)
    _, mode, D = best
    d = np.where(D[0] <= D[1], 1, -1)
    ee, xx = eff_prices(e, x, d, mode)
    df["_d"] = d
    df["_mode"] = mode
    df["_e"], df["_x"] = ee, xx
    df["_exp_net"] = (xx - ee) * sh * d - ch
    df["_pnl_diff"] = np.abs(df["_exp_net"] - pnl)
    # Position sizes grow with equity, so rounding the CSV prices to 2 dp (+-0.005 per leg) can move
    # net_pnl by up to ~0.0101*shares.  Allow that ONLY when the whole book is visibly 2-dp rounded;
    # full-precision books keep the strict tolerance.
    two_dp = (np.abs(e * 100 - np.round(e * 100)) < 1e-6) & (np.abs(x * 100 - np.round(x * 100)) < 1e-6)
    rounded_book = bool(two_dp.mean() > 0.8)
    df["_pnl_tol"] = PNL_TOL + (0.0101 * sh if rounded_book else 0.0)
    if "symbol" in df.columns:
        df["_sym"] = (df["symbol"].astype(str).str.strip().str.lower()
                      .str.replace(r"^nse:", "", regex=True).str.replace(r"\.ns$", "", regex=True))
    else:
        df["_sym"] = ctx.default_symbol()
    ctx.books[key] = df


def lookup(bars: pd.DataFrame, times: pd.DatetimeIndex):
    """Index of the bar for each timestamp: exact match, else floor to 15 min."""
    idx = bars.index.get_indexer(times)
    kind = np.where(idx >= 0, "exact", "none").astype(object)
    miss = np.where(idx < 0)[0]
    if len(miss):
        fl = pd.DatetimeIndex(times[miss]).floor("15min")
        idx2 = bars.index.get_indexer(fl)
        ok = idx2 >= 0
        idx[miss[ok]] = idx2[ok]
        kind[miss[ok]] = "floor"
    return idx, kind


def ground_book(ctx: Ctx, key: str) -> Optional[pd.DataFrame]:
    """Attach the raw bar (and next bar open) for every trade entry/exit timestamp."""
    df = ctx.books[key]
    n = len(df)
    names = ["idx", "open", "high", "low", "close", "nopen", "volume"]
    out = {f"{s}_{k}": np.full(n, np.nan) for s in ("e", "x") for k in names}
    kinds = {s: np.array(["nodata"] * n, dtype=object) for s in ("e", "x")}
    any_bars = False
    for sym in df["_sym"].unique():
        bars = ctx.get_bars(sym)
        if bars is None:
            continue
        any_bars = True
        mask = (df["_sym"] == sym).to_numpy()
        rows = np.where(mask)[0]
        o, h, l, c = (bars[k].to_numpy() for k in ("open", "high", "low", "close"))
        dates = bars.index.normalize().to_numpy()
        for s, col in (("e", "_entry"), ("x", "_exit")):
            times = pd.DatetimeIndex(df.loc[mask, col]).astype("datetime64[ns]")
            idx, kind = lookup(bars, times)
            valid = idx >= 0
            safe = np.where(valid, idx, 0)
            nidx = safe + 1
            nvalid = valid & (nidx < len(bars))
            nsafe = np.minimum(nidx, len(bars) - 1)
            nvalid &= dates[nsafe] == dates[safe]
            out[f"{s}_idx"][rows] = np.where(valid, idx, np.nan)
            for k, arr in (("open", o), ("high", h), ("low", l), ("close", c)):
                out[f"{s}_{k}"][rows] = np.where(valid, arr[safe], np.nan)
            out[f"{s}_nopen"][rows] = np.where(nvalid, o[nsafe], np.nan)
            if "volume" in bars.columns:
                out[f"{s}_volume"][rows] = np.where(valid, bars["volume"].to_numpy()[safe], np.nan)
            kinds[s][rows] = kind
    if not any_bars:
        return None
    g = pd.DataFrame(out, index=df.index)
    g["e_kind"], g["x_kind"] = kinds["e"], kinds["x"]
    ctx.ground[key] = g
    return g


# =============================================================================
# 5. THE CHECKS
# =============================================================================
# ---- Check 1 ----------------------------------------------------------------
def check1(ctx: Ctx) -> None:
    r, d, tag = ctx.rep, ctx.dir, ctx.tag
    if not d.is_dir():
        r.add(1, "FAIL", f"version folder not found: {d}")
        return
    spec = {"pine": [f"{tag}.pine"], "results": ["results.txt"],
            "intraday": ["trades_intraday.csv"], "swing": ["trades_swing.csv"],
            "run": [f"run_{tag}.py"]}
    for key, names in spec.items():
        p = find_file(d, *names)
        if p is None and key in ("pine", "run"):
            alt = sorted(d.glob("*.pine" if key == "pine" else "run_*.py"))
            if len(alt) == 1:
                p = alt[0]
                r.add(1, "WARN", f"{names[0]} not found; using the only candidate {p.name}")
        ctx.paths[key] = p
        if p is None:
            r.add(1, "FAIL", f"missing required file: {names[0]}")
        else:
            r.files[key] = f"{p.name} ({sha(p)})"
    ctx.pine, ctx.run, ctx.results = (read_text(ctx.paths.get(k)) for k in ("pine", "run", "results"))
    if ctx.paths.get("results") and not ctx.results.strip():
        r.add(1, "FAIL", "results.txt is empty")
    for key in ("intraday", "swing"):
        p = ctx.paths.get(key)
        if p:
            df = load_trades(ctx, key, p)
            if df is not None:
                prepare_book(ctx, key, df)
    # ---- One trade = one row.  Position size scales with equity through R(E_k), so there is no
    # ---- legitimate reason to slice a decision into child fills: any slicing is flagged.
    for key, df in ctx.books.items():
        tag_ = f"trades_{key}.csv"
        dup = df.duplicated(["entry_time", "exit_time", "entry_price", "exit_price"], keep=False).to_numpy()
        if dup.any():
            r.add(1, "FAIL", f"{int(dup.sum())} duplicated trade rows in {tag_} (inflated trade count)")
            log_mask(r, 1, df, dup, "row", "unique trade", ["duplicate"] * len(df), "identical entry/exit/prices")
        pid = pick_col(df, ["parent_id", "parent_order_id", "signal_id", "order_id", "parent"])
        if pid is not None:
            rep_id = df[pid].notna() & df.duplicated(pid, keep=False)
            if rep_id.any():
                r.add(1, "FAIL", f"{int(rep_id.sum())} rows in {tag_} share a '{pid}' - sliced orders; one trade must be one row")
                log_mask(r, 1, df, rep_id.to_numpy(), pid, "unique per trade", df[pid].astype(str).to_numpy(), "child fills of one order")
        ebar = df["_entry"].dt.floor("15min")
        same = df.assign(_ebar=ebar).duplicated(["_sym", "_ebar", "_d"], keep=False).to_numpy()
        if same.any():
            r.add(1, "FAIL", f"{int(same.sum())} rows in {tag_} share symbol+direction+entry bar - sliced order; one trade must be one row")
            log_mask(r, 1, df, same, "entry bar", "one row per symbol/direction/bar", df["entry_time"].astype(str).to_numpy(), "possible slicing")
        overlap = np.zeros(len(df), dtype=bool)
        for _, grp in df.sort_values(["_entry", "_row"]).groupby("_sym"):
            run_max = grp["_exit"].cummax().shift(1)
            ov = (grp["_entry"] < run_max).to_numpy()
            overlap[df.index.get_indexer(grp.index[ov])] = True
        if overlap.any() and not same.any():
            r.add(1, "FAIL", f"{int(overlap.sum())} rows in {tag_} open while another position in the SAME symbol is still live (pyramiding/stacking)")
            log_mask(r, 1, df, overlap, "entry_time", "flat before re-entry", df["entry_time"].astype(str).to_numpy(), "overlapping same-symbol position")
        if not (dup.any() or same.any() or overlap.any() or (pid is not None and rep_id.any())):
            r.add(1, "PASS", f"{tag_}: one trade = one row (no slicing, duplicates or same-symbol stacking)")


# ---- Check 2 ----------------------------------------------------------------
def check2(ctx: Ctx) -> None:
    r = ctx.rep
    for key, df in ctx.books.items():
        g = ground_book(ctx, key)
        if g is None:
            sev = "WARN" if ctx.args.allow_missing_data else "FAIL"
            notes = "; ".join(ctx.bar_notes) or "no pickle found"
            r.add(2, sev, f"[{key}] cannot ground trades - raw bar data not loadable for "
                          f"{sorted(df['_sym'].unique())} ({notes}). Use --data-dir.")
            continue
        for note in ctx.bar_notes:
            r.add(2, "PASS" if "normalised" in note else "WARN", f"[{key}] {note}")
        # 2a. timestamps correspond to real bars
        for s, label in (("e", "entry_time"), ("x", "exit_time")):
            none = g[f"{s}_kind"].isin(["none", "nodata"]).to_numpy()
            floor = (g[f"{s}_kind"] == "floor").to_numpy()
            if none.any():
                r.add(2, "FAIL", f"[{key}] {int(none.sum())} trade(s) with {label} matching NO historical bar")
                log_mask(r, 2, df, none, label, "timestamp present in raw bars",
                         df["entry_time" if s == "e" else "exit_time"].to_numpy(), "no such bar")
            if floor.any():
                r.add(2, "WARN", f"[{key}] {int(floor.sum())} {label} value(s) fall mid-bar (matched by flooring to 15m)")
        # 2b. prices inside the bar envelope (or next bar open)
        for s, label, pcol in (("e", "entry_price", "entry_price"), ("x", "exit_price", "exit_price")):
            p = df[pcol].to_numpy(float)
            lo, hi, nop = g[f"{s}_low"].to_numpy(), g[f"{s}_high"].to_numpy(), g[f"{s}_nopen"].to_numpy()
            have = ~np.isnan(lo)
            in_bar = (p >= lo * PRICE_ENV_LO) & (p <= hi * PRICE_ENV_HI)
            in_next = np.abs(p - nop) <= nop * NEXT_OPEN_BAND          # NaN -> False
            bad = have & ~(in_bar | in_next)
            if bad.any():
                r.add(2, "FAIL", f"[{key}] {int(bad.sum())} {label}(s) outside the real bar range "
                                 f"[Low*{PRICE_ENV_LO}, High*{PRICE_ENV_HI}] - prices that never traded")
                exp = [f"[{a * PRICE_ENV_LO:.2f}, {b * PRICE_ENV_HI:.2f}]" for a, b in zip(lo, hi)]
                log_mask(r, 2, df, bad, label, exp, p, "price not in bar range")
            elif have.any():
                r.add(2, "PASS", f"[{key}] all {int(have.sum())} {label} values lie within real bar ranges")
        # 2c. slippage evidence on entries (vs bar open / close)
        eo, ec = g["e_open"].to_numpy(), g["e_close"].to_numpy()
        have = ~np.isnan(eo)
        if have.sum() >= 10:
            e = df["entry_price"].to_numpy(float)
            d = df["_d"].to_numpy()
            dev_o, dev_c = (e / eo - 1), (e / ec - 1)
            adv = np.where(np.abs(dev_o) <= np.abs(dev_c), dev_o, dev_c) * d      # adverse = positive
            zero = float((np.abs(adv[have]) < 5e-5).mean())
            ok = float((np.abs(adv[have] - SLIPPAGE) < 2e-4).mean())
            if zero > 0.5:
                r.add(2, "WARN", f"[{key}] {zero:.0%} of entries fill at the exact bar open/close - "
                                 f"no sign of the mandatory 0.05% slippage")
            elif ok < 0.3 and df["_mode"].iloc[0] == "A":
                r.add(2, "WARN", f"[{key}] only {ok:.0%} of entries show ~0.05% adverse slippage vs bar open/close")
            else:
                r.add(2, "PASS", f"[{key}] entries show slippage evidence (~0.05% adverse on {ok:.0%} of trades)")
        # 2d. liquidity: the order must be a small fraction of what the fill bar really traded.
        #     Position sizes grow with equity, so this is what stops a compounding backtest from
        #     "buying" more shares than the market ever offered.  (Volume assumed in shares.)
        sh = df["shares"].to_numpy(float)
        vols = {s: g[f"{s}_volume"].to_numpy() for s in ("e", "x")}
        if all(np.isnan(v).all() for v in vols.values()):
            sev = "WARN" if ctx.args.allow_missing_data else "FAIL"
            r.add(2, sev, f"[{key}] raw bars carry no volume column - the {MAX_PARTICIPATION:.0%} participation cap cannot be verified")
        else:
            worst = 0.0
            for s, label in (("e", "entry"), ("x", "exit")):
                v = vols[s]
                have = ~np.isnan(v)
                part = np.full(len(df), np.nan)
                pos = have & (v > 0)
                part[pos] = sh[pos] / v[pos]
                bad = have & (~(v > 0) | (part > MAX_PARTICIPATION + 1e-12))
                worst = max(worst, float(np.nanmax(part)) if np.isfinite(part).any() else 0.0)
                if bad.any():
                    r.add(2, "FAIL", f"[{key}] {int(bad.sum())} {label} fill(s) exceed {MAX_PARTICIPATION:.0%} of the bar's volume "
                                     f"(worst {np.nanmax(np.where(bad, np.where(v > 0, sh / np.where(v > 0, v, 1), np.inf), np.nan)):.1%}) - more shares than the market traded")
                    exp_ = [f"<= {MAX_PARTICIPATION:.0%} of {x:,.0f}" if x == x else "n/a" for x in v]
                    act_ = [f"{a:.1%} ({int(b)} sh)" if a == a else f"zero-volume bar ({int(b)} sh)" for a, b in zip(part, sh)]
                    log_mask(r, 2, df, bad, f"{label} participation", exp_, act_, "order larger than liquidity")
            if worst > 0 and not any(c == 2 and s == "FAIL" and "bar's volume" in m and f"[{key}]" in m for c, s, m in r.findings):
                r.add(2, "PASS", f"[{key}] participation within {MAX_PARTICIPATION:.0%} of bar volume on every fill (max {worst:.2%})")


# ---- Check 3 ----------------------------------------------------------------
PY_RULES = [
    (r"\.shift\(\s*(?:periods\s*=\s*)?-\s*[\w.]+", "FAIL", "negative shift() reads FUTURE bars"),
    (r"\.(?:pct_change|diff)\(\s*(?:periods\s*=\s*)?-\s*\w+", "FAIL", "negative-period diff/pct_change reads FUTURE bars"),
    (r"np\.roll\([^)]*,\s*-\s*\w+", "FAIL", "np.roll with negative shift reads FUTURE bars"),
    (r"center\s*=\s*True", "FAIL", "centered rolling window uses FUTURE bars"),
    (r"merge_asof\([^)]*direction\s*=\s*['\"](?:forward|nearest)", "FAIL", "merge_asof forward/nearest leaks future data"),
    (r"\.bfill\(|method\s*=\s*['\"]bfill", "WARN", "backfill propagates FUTURE values backwards"),
    (r"\[\s*::\s*-1\s*\]", "WARN", "reversed series - verify rolling/cum ops do not run on reversed data"),
    (r"\.resample\(", "WARN", "resample(): higher-timeframe values must be shifted one HTF bar before use"),
]
FWD_INDEX = re.compile(r"\[\s*[A-Za-z_]\w*\s*\+\s*\d+\s*[\],]")
SIGNAL_WORDS = re.compile(r"(?i)\b(?:sig\w*|cond\w*|setup\w*|trigger\w*|long_\w*|short_\w*|buy\w*|sell\w*|enter\w*|entry_ok)\b")
FUTURE_NAMES = re.compile(r"(?i)\b(?:future|fwd|forward|lookahead|next_(?:close|high|low|ret\w*))\w*")


def line_of(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def check3(ctx: Ctx) -> None:
    r = ctx.rep
    # ---- 3a. static scan of the Python backtester ---------------------------
    if not ctx.run:
        r.add(3, "FAIL", "run script missing - cannot audit for lookahead bias")
    else:
        code = strip_py_comments(ctx.run)
        hard = 0
        for pat, sev, why in PY_RULES:
            for m in re.finditer(pat, code):
                ln = line_of(code, m.start())
                snippet = code.splitlines()[ln - 1].strip()[:90]
                r.add(3, sev, f"run script L{ln}: {why}  ->  {snippet}")
                hard += sev == "FAIL"
        for m in FWD_INDEX.finditer(code):
            ln = line_of(code, m.start())
            line = code.splitlines()[ln - 1]
            is_sig = bool(SIGNAL_WORDS.search(line)) and "open" not in line.lower()
            r.add(3, "FAIL" if is_sig else "WARN",
                  f"run script L{ln}: forward row index [i+k] "
                  f"{'in a SIGNAL expression (future bar used to decide)' if is_sig else '- legitimate only if it is the next-bar-open fill, verify'}"
                  f"  ->  {line.strip()[:80]}")
            hard += is_sig
        for m in list(FUTURE_NAMES.finditer(code))[:5]:
            ln = line_of(code, m.start())
            r.add(3, "WARN", f"run script L{ln}: suspicious 'future/forward' identifier '{m.group(0)}' (forward label leakage?)")
        lines = code.splitlines()
        vw = [i for i, l in enumerate(lines) if re.search(r"(?i)vwap", l)]
        if vw:
            if any(re.search(r"(?i)vwap[^\n]*shift\(\s*1\s*\)|shift\(\s*1\s*\)[^\n]*vwap", lines[i]) for i in vw) \
                    or re.search(r"(?is)vwap\w*\s*=.*?shift\(\s*1\s*\)", code):
                r.add(3, "PASS", "VWAP is shifted by 1 bar before signal generation")
            else:
                r.add(3, "WARN", "VWAP computed without .shift(1) - signal on bar i must not use bar-i VWAP unless the fill is bar i+1 open")
        elif re.search(r"\.rolling\(|\.ewm\(", code) and not re.search(r"shift\(\s*1\s*\)", code):
            r.add(3, "WARN", "rolling/EMA indicators present but no .shift(1) anywhere - confirm signals use only completed bars")
        tp_l = next((i for i, l in enumerate(lines) if re.search(r"(?i)(?:high|\bh\b)[^\n]*>=?[^\n]*(?:tp|target|take)", l)), None)
        sl_l = next((i for i, l in enumerate(lines) if re.search(r"(?i)(?:low|\bl\b)[^\n]*<=?[^\n]*(?:sl|stop)", l)), None)
        if tp_l is not None and sl_l is not None and tp_l < sl_l:
            r.add(3, "WARN", f"run script evaluates TP (L{tp_l + 1}) before SL (L{sl_l + 1}) - optimistic same-bar ordering; SL must win when both touched")
        if hard == 0 and not any(c == 3 and s != "PASS" for c, s, _ in r.findings):
            r.add(3, "PASS", "run script: no negative shifts, centered windows, forward indices or leak patterns found")
        elif hard == 0:
            r.add(3, "PASS", "run script: no hard lookahead constructs (see warnings above)")

    # ---- 3b. Pine lookahead (details in check 8) -----------------------------
    # ---- 3c. data-driven probes on the intraday book -------------------------
    df, g = ctx.books.get("intraday"), ctx.ground.get("intraday")
    if df is None or g is None:
        r.add(3, "WARN", "data-driven lookahead probes skipped (no grounded intraday trades)")
        return
    # Probe 1: entries systematically at the favourable extreme of the bar = peeking.
    hi, lo, e = g["e_high"].to_numpy(), g["e_low"].to_numpy(), df["entry_price"].to_numpy(float)
    d = df["_d"].to_numpy()
    rng = hi - lo
    ok = ~np.isnan(rng) & (rng > 0)
    if ok.sum() >= 20:
        fav = np.where(d == 1, (hi - e) / np.where(ok, rng, 1), (e - lo) / np.where(ok, rng, 1))
        mean_fav = float(np.clip(fav[ok], 0, 1).mean())
        near_ext = float((np.clip(fav[ok], 0, 1) > 0.9).mean())
        if mean_fav > 0.75 or near_ext > 0.35:
            r.add(3, "FAIL", f"entries sit at the FAVOURABLE extreme of their bar (mean position {mean_fav:.2f}, "
                             f"{near_ext:.0%} within 10% of the best price) - classic lookahead signature")
        elif mean_fav > 0.65 or near_ext > 0.25:
            r.add(3, "WARN", f"entries skew to the favourable end of the bar (mean {mean_fav:.2f}, {near_ext:.0%} near best price)")
        else:
            r.add(3, "PASS", f"entry placement within bar looks unbiased (mean favourable position {mean_fav:.2f})")
    # Probe 2: same-bar trades (open & close inside one 15m bar).
    same = (df["_entry"].dt.floor("15min") == df["_exit"].dt.floor("15min")).to_numpy()
    reason = df["exit_reason"].astype(str).str.lower().to_numpy()
    is_tp = np.array([bool(re.search(r"tp|target|profit|take", s)) for s in reason])
    is_sl = np.array([bool(re.search(r"sl|stop|loss", s)) for s in reason])
    scol, tcol = pick_col(df, SL_COLS), pick_col(df, TP_COLS)
    if same.sum() == 0:
        r.add(3, "PASS", "no same-bar (open+close in one bar) trades")
    elif scol:
        sl = pd.to_numeric(df[scol], errors="coerce").to_numpy(float)
        bl, bh = g["e_low"].to_numpy(), g["e_high"].to_numpy()
        sl_hit = np.where(d == 1, bl <= sl, bh >= sl)
        viol = same & sl_hit & is_tp & ~np.isnan(sl)
        if viol.any():
            r.add(3, "FAIL", f"{int(viol.sum())} same-bar trade(s) booked as TP although the bar also touched the SL - SL must win")
            log_mask(r, 3, df, viol, "exit_reason", "SL (conservative rule)", df["exit_reason"].to_numpy(),
                     "bar range touched SL and TP; TP assumed first")
        else:
            r.add(3, "PASS", f"{int(same.sum())} same-bar trades obey the SL-first conservative rule")
    else:
        sw, sl_n = int(((df["net_pnl"] > 0).to_numpy() & same).sum()), int(((df["net_pnl"] < 0).to_numpy() & same).sum())
        if sw >= 5 and sl_n == 0:
            r.add(3, "FAIL", f"{int(same.sum())} same-bar trades: {sw} wins and ZERO losses - TP/SL ordering is being resolved in the strategy's favour "
                             f"(no SL column to prove otherwise)")
        else:
            r.add(3, "WARN", f"{int(same.sum())} same-bar trades ({sw} wins/{sl_n} losses); no SL/TP columns in CSV so intrabar ordering cannot be proven")
    # Probe 3: stop-loss was breached on earlier bars yet the trade stayed open.
    bars_ok = scol is not None and not np.isnan(g["e_idx"].to_numpy()).all()
    if bars_ok:
        sl = pd.to_numeric(df[scol], errors="coerce").to_numpy(float)
        missed, checked = np.zeros(len(df), dtype=bool), 0
        for i in range(len(df)):
            ei, xi = g["e_idx"].iloc[i], g["x_idx"].iloc[i]
            if np.isnan(ei) or np.isnan(xi) or np.isnan(sl[i]) or is_sl[i]:
                continue
            bars = ctx.get_bars(df["_sym"].iloc[i])
            if bars is None:
                continue
            fill_at_close = abs(e[i] - g["e_close"].iloc[i]) < abs(e[i] - g["e_open"].iloc[i])
            a, b = int(ei) + (1 if fill_at_close else 0), int(xi)
            if b <= a:
                continue
            checked += 1
            seg = bars.iloc[a:b]
            missed[i] = bool((seg["low"].min() <= sl[i]) if d[i] == 1 else (seg["high"].max() >= sl[i]))
        if missed.any():
            r.add(3, "FAIL", f"{int(missed.sum())}/{checked} trades stayed open although a bar BEFORE exit already breached the stop-loss")
            log_mask(r, 3, df, missed, "stop-loss", "exit at SL when breached", df["exit_reason"].to_numpy(), "SL breached earlier")
        elif checked:
            r.add(3, "PASS", f"stop-loss never breached before exit across {checked} audited trades")


# ---- Check 4 ----------------------------------------------------------------
def check4(ctx: Ctx) -> None:
    r = ctx.rep
    for key, df in ctx.books.items():
        d, mode = df["_d"].to_numpy(), df["_mode"].iloc[0]
        sh = df["shares"].to_numpy(float)
        ee, xx = df["_e"].to_numpy(float), df["_x"].to_numpy(float)
        buy = sh * np.where(d == 1, ee, xx)
        sell = sh * np.where(d == 1, xx, ee)
        calc = dhan_charges(buy, sell, key)
        csv = df["charges"].to_numpy(float)
        diff = np.abs(calc - csv)
        n = len(df)
        r.add(4, "PASS" if mode == "A" else "WARN",
              f"[{key}] price convention: " + ("CSV prices already include slippage" if mode == "A"
                                                 else "CSV prices are RAW; slippage applied on top by backtester"))
        if (csv <= 0).any():
            r.add(4, "FAIL", f"[{key}] {int((csv <= 0).sum())} trade(s) with zero/negative charges (costs evaded)")
            log_mask(r, 4, df, csv <= 0, "charges", calc, csv, "non-positive charges")
        bad = diff > CHARGE_TOL
        if bad.any():
            r.add(4, "FAIL", f"[{key}] {int(bad.sum())}/{n} trades: CSV charges differ from Dhan schedule by > Rs {CHARGE_TOL:.2f} "
                             f"(max Rs {diff.max():.2f})")
            log_mask(r, 4, df, bad, "charges", np.round(calc, 2), csv, "recomputed from Dhan schedule")
        else:
            r.add(4, "PASS", f"[{key}] charges match Dhan schedule on all {n} trades (max diff Rs {diff.max():.3f}, mean {diff.mean():.3f})")
        bias = float((csv - calc).mean())
        if bias < -0.05:
            r.add(4, "WARN", f"[{key}] CSV charges are systematically LOW vs official (mean Rs {bias:+.3f} per trade)")
        turn = float((buy + sell).sum())
        ratio = float(csv.sum() / turn) if turn > 0 else 0.0
        lo, hi = CHARGE_RATIO_BAND
        if not lo <= ratio <= hi:
            r.add(4, "WARN", f"[{key}] charges are {ratio:.4%} of turnover (sane band {lo:.4%}..{hi:.4%}) - "
                             f"mean Rs {float(csv.mean()):.2f}/trade")
        pd_bad = df["_pnl_diff"].to_numpy() > df["_pnl_tol"].to_numpy()
        if pd_bad.any():
            r.add(4, "FAIL", f"[{key}] {int(pd_bad.sum())}/{n} trades: net_pnl not reproducible from shares*(exit-entry) - charges "
                             f"(max Rs {df['_pnl_diff'].max():.2f}; tolerance Rs {df.loc[pd_bad, '_pnl_tol'].max():.2f} incl. 2-dp price rounding allowance)")
            log_mask(r, 4, df, pd_bad, "net_pnl", np.round(df["_exp_net"].to_numpy(), 2), df["net_pnl"].to_numpy(),
                     "price-derived net PnL")
        else:
            r.add(4, "PASS", f"[{key}] net_pnl reproduces from prices, shares and charges on all {n} trades")


# ---- Check 5 ----------------------------------------------------------------
def audit_constitution(ctx: Ctx) -> None:
    """The CLI may only TIGHTEN the constitution; loosening any value is a FAIL (anti-self-loosening)."""
    r = ctx.rep
    for k, base in CONSTITUTION.items():
        now = globals()[k]
        if now == base:
            continue
        sign = LOOSER_IF.get(k)
        if sign is not None and (now - base) * sign > 0:
            r.add(5, "FAIL", f"constitution LOOSENED: {k} = {now} (default {base}) - the CLI may only tighten limits; "
                             f"to loosen deliberately edit the CONSTITUTION block in the gate file itself")
        else:
            r.add(5, "WARN", f"non-default parameter in force: {k} = {now} (default {base})")


def build_ledger(ctx: Ctx) -> pd.DataFrame:
    """
    One shared account across BOTH books, in chronological order of entry.  For every trade k the
    gate derives - only from verified closed trades - the numbers its limits are judged against:

      E_k      realized equity = CAPITAL + sum(net_pnl of trades whose exit is strictly before entry_k)
      HWM_k    high-water mark of that realized-equity path (>= CAPITAL)
      R_k      risk budget = min(RISK_PCT * E_k, RISK_CEILING) * (DD_THROTTLE_FACTOR if drawdown > trigger else 1)
      used_k   margin already tied up at entry_k: open intraday notional/lev, open swing notional (100% cash),
               plus swing sale proceeds still unsettled (T+SWING_SETTLE_BDAYS)
      req_k    this trade's margin (intraday: notional/lev_symbol, swing: notional)
    """
    parts = []
    for key, df in ctx.books.items():
        scol = pick_col(df, SL_COLS)
        sl = pd.to_numeric(df[scol], errors="coerce").to_numpy(float) if scol else np.full(len(df), np.nan)
        parts.append(pd.DataFrame({
            "book": key, "_row": df["_row"].to_numpy(), "sym": df["_sym"].to_numpy(),
            "entry": df["_entry"].to_numpy("datetime64[ns]"), "exit": df["_exit"].to_numpy("datetime64[ns]"),
            "shares": df["shares"].to_numpy(float), "e": df["entry_price"].to_numpy(float),
            "net": df["net_pnl"].to_numpy(float), "sl": sl, "swing": key == "swing",
            "reason": df["exit_reason"].astype(str).to_numpy()}))
    L = pd.concat(parts, ignore_index=True).sort_values(["entry", "book", "_row"], kind="stable").reset_index(drop=True)
    n = len(L)
    ent = L["entry"].to_numpy("datetime64[ns]").astype("int64")
    ex = L["exit"].to_numpy("datetime64[ns]").astype("int64")
    net, sh, e = (L[c].to_numpy(float) for c in ("net", "shares", "e"))
    is_sw = L["swing"].to_numpy(bool)
    notional = sh * e

    lev_raw = np.array([1.0 if s else (ctx.lev_for(sym) if ctx.lev_for(sym) is not None else np.nan)
                        for s, sym in zip(is_sw, L["sym"])], dtype=float)
    lev_eff = np.where(np.isnan(lev_raw), LEVERAGE, lev_raw)
    margin_per = notional / lev_eff

    # realized equity / high-water mark at each entry (closed trades only, strictly earlier exit)
    order = np.argsort(ex, kind="stable")
    ex_s = ex[order]
    cum = CAPITAL + np.cumsum(net[order])
    hwm_run = np.maximum.accumulate(np.maximum(cum, CAPITAL))
    pos = np.searchsorted(ex_s, ent, side="left")
    prev = np.maximum(pos - 1, 0)
    E = np.where(pos > 0, cum[prev], CAPITAL)
    HWM = np.where(pos > 0, hwm_run[prev], CAPITAL)
    dd = 1.0 - E / HWM
    thr = np.where(dd > DD_THROTTLE_TRIGGER + 1e-12, DD_THROTTLE_FACTOR, 1.0)
    base_rcap = np.where(E <= 2500.0, RISK_PCT * np.maximum(E, 0.0), np.minimum(210.0 + 0.040 * (E - 2500.0), RISK_CEILING))
    Rcap = base_rcap * thr

    # margin already in use when trade k enters (positions j<k entered no later than k)
    settle = np.full(n, np.iinfo(np.int64).max, dtype="int64")
    for i in np.where(is_sw)[0]:
        settle[i] = (pd.Timestamp(int(ex[i])).normalize() + pd.offsets.BDay(SWING_SETTLE_BDAYS)).value
    used = np.zeros(n)
    for k in range(1, n):
        still_open = ex[:k] >= ent[k]
        locked = is_sw[:k] & (ex[:k] < ent[k]) & (settle[:k] > ent[k])
        used[k] = margin_per[:k][still_open].sum() + notional[:k][locked].sum()
    avail = MARGIN_BUFFER * E
    maxnot = np.maximum(avail - used, 0.0) * lev_eff

    # planned stop risk and R-multiples (denominator falls back to the budget R_k without a stop column)
    with np.errstate(invalid="ignore"):
        planned = sh * np.abs(e - L["sl"].to_numpy(float))
    has_sl = np.isfinite(planned) & (planned > 0)
    risk_den = np.maximum(np.where(has_sl, planned, Rcap), 0.01)
    L["notional"], L["lev"], L["lev_missing"] = notional, lev_eff, (~is_sw) & np.isnan(lev_raw)
    L["E"], L["HWM"], L["dd"], L["thr"], L["Rcap"] = E, HWM, dd, thr, Rcap
    L["used"], L["req"], L["avail"], L["maxnot"] = used, margin_per, avail, maxnot
    L["planned"], L["risk_den"], L["risk_src"] = np.where(has_sl, planned, np.nan), risk_den, np.where(has_sl, "stop", "budget")
    L["rmult"] = net / risk_den
    L["ret"] = np.where(E > 0, net / np.where(E > 0, E, 1.0), np.nan)
    return L


def check5(ctx: Ctx) -> None:
    r = ctx.rep
    audit_constitution(ctx)
    if ctx.lev_err:
        r.add(5, "FAIL", f"leverage table unreadable ({ctx.lev_err}) - per-symbol MIS leverage cannot be enforced")
    elif ctx.lev_table is None:
        r.add(5, "WARN", f"no --leverage-table: assuming {LEVERAGE:g}x MIS for EVERY symbol (Dhan's real per-stock "
                         f"leverage is often lower; supply it to enforce reality)")
    L = build_ledger(ctx)
    ctx.ledger = L
    for key, df in ctx.books.items():
        Lk = L[L["book"] == key].set_index("_row").reindex(df["_row"].to_numpy())
        net, E, R = (Lk[c].to_numpy(float) for c in ("net", "E", "Rcap"))
        loss = np.maximum(-net, 0.0)
        tag = f"[{key}]"
        # ---- 5a. dynamic loss budget  R(E_k) --------------------------------------------------
        hard = loss > R + RISK_HARD_EXTRA + 1e-9
        warn = (loss > R + 1e-9) & ~hard
        exp_ = [f"<= {c + RISK_HARD_EXTRA:.2f} (R={c:.2f}, E_k={q:,.0f})" for c, q in zip(R, E)]
        if hard.any():
            w = int(np.argmax(np.where(hard, loss - R, -np.inf)))
            r.add(5, "FAIL", f"{tag} {int(hard.sum())} trade(s) lost more than R(E_k)+Rs {RISK_HARD_EXTRA:.2f} "
                             f"(worst Rs {loss[w]:.2f} vs budget Rs {R[w]:.2f} at E_k=Rs {E[w]:,.0f})")
            log_mask(r, 5, df, hard, "loss vs R(E_k)", exp_, np.round(loss, 2), "risk budget breached")
        elif warn.any():
            r.add(5, "WARN", f"{tag} {int(warn.sum())} trade(s) exceed R(E_k) by <= Rs {RISK_HARD_EXTRA:.2f} (charges/slippage overshoot)")
            log_mask(r, 5, df, warn, "loss vs R(E_k)", exp_, np.round(loss, 2), "slightly over budget")
        else:
            r.add(5, "PASS", f"{tag} every loss within R(E_k) (worst Rs {loss.max():.2f}; budget scaled {R.min():.2f}..{R.max():.2f})")
        # ---- 5b. planned stop-risk (position sizing honesty) ----------------------------------
        pl = Lk["planned"].to_numpy(float)
        if np.isfinite(pl).any():
            over = np.isfinite(pl) & (pl > R + RISK_HARD_EXTRA + 1e-9)
            if over.any():
                r.add(5, "FAIL", f"{tag} {int(over.sum())} position(s) sized so that shares*|entry-stop| exceeds R(E_k) - "
                                 f"over-sized even if the stop happened to hold")
                log_mask(r, 5, df, over, "shares*|entry-sl|", exp_, np.round(pl, 2), "planned risk above budget")
            else:
                r.add(5, "PASS", f"{tag} planned stop-risk shares*|entry-sl| within R(E_k) on all {int(np.isfinite(pl).sum())} sized trades")
        # ---- 5c. drawdown throttle applied ----------------------------------------------------
        thr_n = int((Lk["thr"].to_numpy(float) < 1).sum())
        if thr_n:
            r.add(5, "PASS", f"{tag} {thr_n} trade(s) sized under the {DD_THROTTLE_FACTOR:.0%} drawdown throttle "
                             f"(equity > {DD_THROTTLE_TRIGGER:.0%} below high-water mark) and obeyed it")
        # ---- 5d. leverage availability + shared-pool margin -----------------------------------
        miss = Lk["lev_missing"].to_numpy(bool)
        if miss.any():
            r.add(5, "FAIL", f"{tag} {int(miss.sum())} trade(s) in symbols with no MIS leverage in the table "
                             f"({sorted(set(df.loc[miss, '_sym']))}) - MIS not available, trade cannot exist")
            log_mask(r, 5, df, miss, "symbol", "present in leverage table", df["_sym"].to_numpy(), "no MIS leverage")
        used, req, avail = (Lk[c].to_numpy(float) for c in ("used", "req", "avail"))
        viol = (used + req) > avail + 0.01
        if viol.any():
            nt = (Lk["notional"].to_numpy(float))
            mx = Lk["maxnot"].to_numpy(float)
            w = int(np.argmax(np.where(viol, used + req - avail, -np.inf)))
            r.add(5, "FAIL", f"{tag} {int(viol.sum())} trade(s) needed more margin than {MARGIN_BUFFER:.0%} of realized equity E_k "
                             f"(worst: notional Rs {nt[w]:,.0f} vs max allowed Rs {mx[w]:,.0f} at E_k=Rs {E[w]:,.0f}, "
                             f"margin already in use Rs {used[w]:,.0f}) - over-leveraged / sized off equity that did not exist yet")
            exp2 = [f"notional <= {m:,.0f} (E_k={q:,.0f}, in use {u:,.0f})" for m, q, u in zip(mx, E, used)]
            log_mask(r, 5, df, viol, "notional vs E_k", exp2, np.round(nt, 2), "margin > buffered equity")
        else:
            util = np.where(avail > 0, (used + req) / np.where(avail > 0, avail, 1), np.inf)
            r.add(5, "PASS", f"{tag} margin <= {MARGIN_BUFFER:.0%} of realized equity at every entry "
                             f"(peak utilisation {util.max():.0%}, largest notional Rs {Lk['notional'].max():,.0f} at E_k=Rs {E[int(np.argmax(Lk['notional'].to_numpy(float)))]:,.0f})")
    # ---- 5e. account-level path -------------------------------------------------------------
    ordx = np.argsort(L["exit"].to_numpy("datetime64[ns]").astype("int64"), kind="stable")
    path = CAPITAL + np.cumsum(L["net"].to_numpy(float)[ordx])
    peak = np.maximum.accumulate(np.maximum(path, CAPITAL))
    mdd = float(((peak - path) / peak).max()) if len(path) else 0.0
    final = float(path[-1]) if len(path) else CAPITAL
    r.meta["account"] = dict(start=float(CAPITAL), final=final, growth=final / CAPITAL, max_dd=mdd, trades=int(len(L)),
                             throttled=int((L["thr"] < 1).sum()), min_equity_at_entry=float(L["E"].min()),
                             peak_equity=float(path.max()) if len(path) else CAPITAL,
                             largest_notional=float(L["notional"].max()))
    if final <= 0 or (path <= 0).any():
        r.add(5, "FAIL", f"account equity reached Rs {path.min():.2f} (<= 0): blown account")
    else:
        r.add(5, "PASS", f"shared-pool equity path Rs {CAPITAL:,.0f} -> Rs {final:,.2f} ({final / CAPITAL:.2f}x), "
                         f"max drawdown {mdd:.0%}, {len(L)} trades across {L['book'].nunique()} book(s)")


# ---- Check 6 ----------------------------------------------------------------
def check6(ctx: Ctx) -> None:
    r = ctx.rep
    df = ctx.books.get("intraday")
    if df is None:
        r.add(6, "FAIL", "trades_intraday.csv unusable - square-off cannot be verified")
        return
    ent, ex = df["_entry"], df["_exit"]
    te, tx = mins(ent), mins(ex)
    n = len(df)
    tests = [
        ("FAIL", "exit date != entry date (OVERNIGHT hold in MIS)", ent.dt.normalize() != ex.dt.normalize(), "exit_time"),
        ("FAIL", "exit_time earlier than entry_time", ex < ent, "exit_time"),
        ("FAIL", f"exit after {SQUAREOFF_HARD[0]}:{SQUAREOFF_HARD[1]:02d} (MIS auto-square-off cut-off)", tx > hm(SQUAREOFF_HARD), "exit_time"),
        ("FAIL", "entry outside NSE session 09:15-15:30", (te < hm(SESSION_OPEN)) | (te >= hm(SESSION_CLOSE)), "entry_time"),
        ("FAIL", "entry on a weekend (no NSE session)", ent.dt.dayofweek >= 5, "entry_time"),
        ("WARN", f"exit after the {SQUAREOFF_POLICY[0]}:{SQUAREOFF_POLICY[1]:02d} square-off policy (but before hard cut-off)",
         (tx > hm(SQUAREOFF_POLICY)) & (tx <= hm(SQUAREOFF_HARD)), "exit_time"),
        ("WARN", f"new entry at/after {SQUAREOFF_POLICY[0]}:{SQUAREOFF_POLICY[1]:02d}", te >= hm(SQUAREOFF_POLICY), "entry_time"),
    ]
    clean = True
    for sev, msg, mask, col in tests:
        mask = np.asarray(mask)
        if mask.any():
            clean = clean and sev != "FAIL"
            r.add(6, sev, f"{int(mask.sum())}/{n} trades: {msg}")
            if sev == "FAIL":
                log_mask(r, 6, df, mask, col, msg.split(" (")[0], df[col].to_numpy(), "square-off/session rule")
    # Timestamp ambiguity.  With bar-START labels the 15:00 bar covers 15:00-15:15: an exit at its OPEN
    # is 15:00 (inside the 15:10 policy) but an exit at its CLOSE is really 15:15 (outside it).
    g = ctx.ground.get("intraday")
    if g is not None:
        x = df["exit_price"].to_numpy(float)
        xo, xc = g["x_open"].to_numpy(), g["x_close"].to_numpy()
        timey = df["exit_reason"].astype(str).str.contains(r"(?i)time|eod|square|sqoff|close|session", regex=True).to_numpy()
        lastbar = (tx >= hm((15, 0))) & (tx <= hm(SQUAREOFF_POLICY))
        dd_ = df["_d"].to_numpy()
        slip = (1 - SLIPPAGE * dd_) if df["_mode"].iloc[0] == "A" else np.ones(len(df))   # exit leg slips against us
        tol = np.maximum(2e-4 * x, 0.0065)              # 2-dp price rounding / tick allowance
        at_close = ~np.isnan(xc) & (np.abs(x - xc * slip) <= tol) & (np.abs(x - xo * slip) > tol)
        amb = lastbar & timey & at_close
        if amb.any():
            r.add(6, "WARN", f"{int(amb.sum())}/{n} square-off exits are labelled on the 15:00 bar (start label) but filled at that "
                             f"bar's CLOSE = 15:15 real time, after the {SQUAREOFF_POLICY[0]}:{SQUAREOFF_POLICY[1]:02d} policy "
                             f"(still before the broker's MIS auto-square-off) - exit at the 15:00 bar OPEN to stay inside policy")
    r.add(6, "PASS", f"timestamps read as bar-{TS_CONVENTION.upper()} labels (a 15:00-labelled bar spans 15:00-15:15)")
    if clean and not any(c == 6 and s != "PASS" for c, s, _ in r.findings):
        r.add(6, "PASS", f"all {n} trades flat same day, exits <= {SQUAREOFF_POLICY[0]}:{SQUAREOFF_POLICY[1]:02d}, entries inside session")
    elif clean:
        r.add(6, "PASS", f"zero overnight positions across {n} trades (see warnings)")


# ---- Check 7 ----------------------------------------------------------------
_PRIORITY = ["win_rate", "avg_win", "avg_loss", "pf", "net", "wins", "losses", "trades"]
_PAT = {
    "win_rate": r"win\s*-?\s*rate|win\s*%|hit\s*rate",
    "avg_win": r"(?:avg|average|mean)\.?\s*(?:win|winner|profit)\w*",
    "avg_loss": r"(?:avg|average|mean)\.?\s*(?:loss|loser)\w*",
    "pf": r"profit\s*factor|\bpf\b",
    "net": r"(?:total\s*)?net\s*(?:p\s*&\s*l|p\s*/\s*l|pnl|profit|return)|total\s*(?:net\s*)?pnl",
    "wins": r"winning\s*trades?|\bwins?\b|\bwinners?\b",
    "losses": r"losing\s*trades?|\blosses\b|\blosers?\b|\bloss\s*count",
    "trades": r"total\s*trades?|(?:number|no\.?|num)\s*of\s*trades?|trades?\s*(?:count|taken)|\btrades?\b",
}
_NUM = r"(?P<num>[-+]?\s*(?:\u20b9|rs\.?|inr)?\s*[-+]?\d[\d,]*(?:\.\d+)?|inf(?:inity)?)"


def to_float(tok: str) -> float:
    t = tok.lower().replace(",", "").replace("\u20b9", "").replace("rs.", "").replace("rs", "").replace("inr", "").replace(" ", "")
    return math.inf if "inf" in t else float(t)


def split_sections(text: str) -> Tuple[Dict[str, List[str]], bool]:
    secs: Dict[str, List[str]] = {}
    cur, found = None, False
    lines = text.splitlines()
    for ln in lines:
        core = re.sub(r"^[\s#=\-\*\[\]_~>|]+", "", ln).strip()
        m = re.match(r"(?i)^(intraday|swing)\b", core)
        if m and not re.search(r"[:=]\s*[-+]?[\u20b9rs.\s]*\d", core, flags=re.I):
            cur = m.group(1).lower()
            secs.setdefault(cur, [])
            found = True
            continue
        if cur:
            secs[cur].append(ln)
    if not found:
        secs["intraday"] = lines
    return secs, found


def parse_section(lines: List[str]) -> Dict[str, Tuple[float, str]]:
    found: Dict[str, Tuple[float, str]] = {}
    for ln in lines:
        for chunk in re.split(r"\||;|\t|\s{3,}", ln):
            for metric in _PRIORITY:
                if metric in found:
                    continue
                pats = [rf"(?i)(?:{_PAT[metric]})[^0-9\n]{{0,30}}?[:=|]\s*{_NUM}",
                        rf"(?i)(?:{_PAT[metric]})\s+{_NUM}"]
                hit = None
                for p in pats:
                    hit = re.search(p, chunk)
                    if hit:
                        break
                if hit:
                    try:
                        found[metric] = (to_float(hit.group("num")), chunk.strip())
                    except ValueError:
                        continue
                    break                                       # one metric per chunk
    return found


def reconcile(ctx: Ctx, key: str, rep_vals: Dict[str, Tuple[float, str]]) -> None:
    r = ctx.rep
    m = metrics(ctx.books[key])
    r.meta[key] = m
    actual = {"trades": m["n"], "wins": m["wins"], "losses": m["losses"], "win_rate": m["win_rate"],
              "net": m["net"], "pf": m["pf"], "avg_win": m["avg_win"], "avg_loss": m["avg_loss"]}
    # alternative (gross-basis) win count - used only to explain a mismatch
    gross = (ctx.books[key]["net_pnl"] + ctx.books[key]["charges"])
    gross_wins = int((gross > 0).sum())
    required = ["trades", "win_rate", "net"]
    if not any(k in rep_vals for k in required):
        r.add(7, "FAIL", f"[{key}] results.txt: could not parse trades / win rate / net PnL - report is unverifiable")
        return
    for k in ["trades", "wins", "losses", "win_rate", "net", "pf", "avg_win", "avg_loss"]:
        if k not in rep_vals:
            r.add(7, "FAIL" if k in required else "WARN", f"[{key}] results.txt does not report '{k}' (cannot verify)")
            continue
        val, raw = rep_vals[k]
        if k == "win_rate" and val <= 1.0 and "%" not in raw and actual[k] > 1.5:
            val *= 100.0
        a = actual[k]
        if math.isinf(val) or math.isinf(a):
            ok = math.isinf(val) == math.isinf(a) or (k == "pf" and min(val, a) > 50)
        elif k == "pf":
            ok = abs(val - a) <= max(RESULT_TOL["pf"], 0.005 * abs(a))
        else:
            ok = abs(val - a) <= RESULT_TOL[k] + 1e-9
        if ok:
            r.add(7, "PASS", f"[{key}] {k}: results.txt {fmt(val)} == CSV {fmt(a)}")
        else:
            note = "FABRICATED / STALE REPORT"
            if k in ("wins", "win_rate") and gross_wins and abs(val - (gross_wins if k == 'wins' else 100.0 * gross_wins / m['n'])) < 0.11:
                note += " (matches GROSS-of-charges win count - flattering definition)"
            r.add(7, "FAIL", f"[{key}] {k}: results.txt says {fmt(val)} but trades CSV gives {fmt(a)}")
            r.disc(7, f"results.txt [{key}]", k, fmt(a), fmt(val), note)


def check7(ctx: Ctx) -> None:
    r = ctx.rep
    if not ctx.results.strip():
        r.add(7, "FAIL", "results.txt missing/empty - nothing to reconcile")
        return
    secs, found = split_sections(ctx.results)
    if not found:
        r.add(7, "WARN", "results.txt has no 'Intraday'/'Swing' section headers; whole file treated as Intraday")
    for key in ("intraday", "swing"):
        if key not in ctx.books:
            if key == "intraday":
                r.add(7, "FAIL", "intraday trades CSV unusable - cannot reconcile")
            continue
        if key not in secs:
            r.add(7, "FAIL" if key == "intraday" else "WARN", f"results.txt has no {key.capitalize()} section")
            if key == "intraday":
                continue
            r.meta[key] = metrics(ctx.books[key])
            continue
        reconcile(ctx, key, parse_section(secs[key]))


# ---- Check 8 ----------------------------------------------------------------
def check8(ctx: Ctx) -> None:
    r = ctx.rep
    txt = ctx.pine
    if not txt.strip():
        r.add(8, "FAIL", "Pine script missing/empty")
        return
    lines = txt.splitlines()
    first = next((l.strip() for l in lines if l.strip()), "")
    m = re.match(r"//\s*@version\s*=\s*(\d+)", first)
    if m and m.group(1) in ("5", "6"):
        r.add(8, "PASS", f"Pine header //@version={m.group(1)}")
    elif re.search(r"//\s*@version\s*=\s*[56]", "\n".join(lines[:10])):
        r.add(8, "WARN", "//@version=5/6 present but not the first line")
    else:
        r.add(8, "FAIL", "Pine script does not start with //@version=5 or //@version=6")
    code = re.sub(r"//.*", "", txt)
    if not re.search(r"\bstrategy\s*\(", code):
        r.add(8, "FAIL", "no strategy(...) declaration (indicator() cannot backtest)")
    entry = re.search(r"\bstrategy\.entry\s*\(", code)
    exit_ = re.search(r"\bstrategy\.(?:exit|close|close_all)\s*\(", code)
    if entry and exit_:
        r.add(8, "PASS", "contains strategy.entry and strategy.exit/close commands")
    else:
        r.add(8, "FAIL", f"missing trading commands: entry={'yes' if entry else 'NO'}, exit/close={'yes' if exit_ else 'NO'}")
    ct = re.search(r"commission_type\s*=\s*strategy\.commission\.cash_per_order", code)
    cv = re.search(r"commission_value\s*=\s*([\d.]+)", code)
    if ct and cv and float(cv.group(1)) > 0:
        r.add(8, "PASS", f"commission: cash_per_order, value {cv.group(1)}")
    else:
        r.add(8, "FAIL", "commission not set as commission_type=strategy.commission.cash_per_order with commission_value > 0")
    sl = re.search(r"\bslippage\s*=\s*(\d+)", code)
    if sl and int(sl.group(1)) > 0:
        r.add(8, "PASS", f"slippage set ({sl.group(1)} ticks)")
    else:
        r.add(8, "FAIL", "slippage missing or zero in strategy() declaration")
    ic = re.search(r"initial_capital\s*=\s*([\d.]+)", code)
    if ic and abs(float(ic.group(1)) - CAPITAL) > 1e-9:
        r.add(8, "WARN", f"initial_capital={ic.group(1)} differs from Rs {CAPITAL:.0f}")
    elif not ic:
        r.add(8, "WARN", "initial_capital not set (TradingView default is 1,000,000)")
    ml = re.search(r"margin_long\s*=\s*([\d.]+)", code)
    if not ml or abs(float(ml.group(1)) - 100.0 / LEVERAGE) > 1e-9:
        r.add(8, "WARN", f"margin_long should be {100 // LEVERAGE} for {LEVERAGE}x MIS (found {ml.group(1) if ml else 'default'})")
    if re.search(r"lookahead\s*=\s*barmerge\.lookahead_on", code):
        r.add(8, "FAIL", "request.security with lookahead_on - peeks at the future HTF bar")
    elif re.search(r"request\.security\s*\(", code) and not re.search(r"lookahead\s*=\s*barmerge\.lookahead_off", code):
        r.add(8, "WARN", "request.security without explicit lookahead_off")
    if re.search(r"\w\[\s*-\s*\d", code):
        r.add(8, "FAIL", "negative history index in Pine (future reference)")
    if re.search(r"calc_on_every_tick\s*=\s*true", code, flags=re.I):
        r.add(8, "WARN", "calc_on_every_tick=true repaints intrabar - backtest not reproducible live")
    pm = re.search(r"pyramiding\s*=\s*(\d+)", code)
    if pm and int(pm.group(1)) > 0:
        r.add(8, "WARN", f"pyramiding={pm.group(1)} allows stacked positions beyond the margin cap")
    if not re.search(r"1510|15\s*:\s*10|hour\s*==\s*15|strategy\.close_all|session\.|timestamp\(", code):
        r.add(8, "WARN", "no visible 15:10 intraday square-off logic (strategy.close_all / time filter)")


# ---- Check 9 ----------------------------------------------------------------
def wilson_lower(wins: int, n: int, z: float = 1.96) -> float:
    if n == 0:
        return 0.0
    p = wins / n
    den = 1 + z * z / n
    return 100 * ((p + z * z / (2 * n)) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / den


def check9(ctx: Ctx) -> None:
    """
    Plausibility / curve-fit alarms that stay valid when position sizes grow with equity.
    Rs-based profit factor, t-stat and tail concentration are dominated by the biggest (latest) positions
    under compounding, so everything here is measured in R-multiples (net_pnl / planned stop risk, or
    / R(E_k) when the CSV has no stop column) and in growth per trade (net_pnl / E_k).
    """
    r = ctx.rep
    if ctx.ledger is None:
        ctx.ledger = build_ledger(ctx)
    L = ctx.ledger
    # ---- per book, in R ------------------------------------------------------------------------
    for key, df in ctx.books.items():
        Lk = L[L["book"] == key]
        pnl = Lk["net"].to_numpy(float)
        Rm = Lk["rmult"].to_numpy(float)
        n = len(pnl)
        tag = f"[{key}]"
        wins, losses = int((pnl > 0).sum()), int((pnl < 0).sum())
        wr = 100.0 * wins / n
        gp, gl = float(Rm[Rm > 0].sum()), float(-Rm[Rm < 0].sum())
        pf = gp / gl if gl > 0 else (math.inf if gp > 0 else 0.0)
        avg_loss_r = float(Rm[Rm < 0].mean()) if losses else 0.0
        exp_r = float(Rm.mean())
        if float((Lk["risk_src"] == "stop").mean()) < 0.5:
            r.add(9, "WARN", f"{tag} R-multiples are NOMINAL (denominator = the R(E_k) budget) because the CSV has no stop-loss "
                             f"column - add 'sl' so risk-at-entry is real")
        if losses == 0:
            r.add(9, "FAIL" if n >= 5 else "WARN", f"{tag} ZERO losing trades out of {n} - 100% win rate is a bug or fraud")
        if wr > 85 and n > 15:
            r.add(9, "FAIL", f"{tag} win rate {wr:.1f}% on {n} trades - implausible for equity intraday without lookahead")
        if pf > 10:
            r.add(9, "FAIL" if n >= 10 else "WARN", f"{tag} profit factor (in R) {fmt(pf)} > 10")
        if losses and avg_loss_r > -AVG_LOSS_MIN_R:
            r.add(9, "FAIL", f"{tag} average loser is {avg_loss_r:.3f}R (> -{AVG_LOSS_MIN_R}R) - stops are not real stops")
        if n < 30:
            r.add(9, "WARN", f"{tag} only {n} trades - statistically weak sample")
        if float(pnl.sum()) < 0:
            r.add(9, "WARN", f"{tag} net NEGATIVE after costs (Rs {pnl.sum():.2f})")
        if n >= 10 and Rm.std(ddof=1) > 0:
            t = exp_r / (Rm.std(ddof=1) / math.sqrt(n))
            if t < 1.65:
                r.add(9, "WARN", f"{tag} expectancy {exp_r:+.3f}R/trade, t-stat {t:.2f} (< 1.65): edge indistinguishable from noise")
        if n >= 10 and gp > 0:
            top3 = float(np.sort(Rm)[::-1][:3].sum() / gp)
            if top3 > TOP3_R_WARN:
                r.add(9, "WARN", f"{tag} top-3 trades carry {top3:.0%} of all positive R - result hinges on a few outliers")
        if n >= 20:
            h = n // 2
            a, b = Rm[:h], Rm[h:]
            pfa = a[a > 0].sum() / max(-a[a < 0].sum(), 1e-9)
            pfb = b[b > 0].sum() / max(-b[b < 0].sum(), 1e-9)
            if pfa > 2 and pfb < 1:
                r.add(9, "WARN", f"{tag} edge decays: PF(R) {fmt(float(pfa))} in first half -> {fmt(float(pfb))} in second half (curve-fit signature)")
            top = Lk["reason"].astype(str).str.lower().value_counts(normalize=True)
            if top.iloc[0] > 0.95:
                r.add(9, "WARN", f"{tag} {top.iloc[0]:.0%} of trades share one exit reason ('{top.index[0]}')")
        r.meta.setdefault(f"{key}_extra", {}).update(pf_r=float(pf), expectancy_r=exp_r, avg_loss_r=avg_loss_r,
                                                      wilson_wr_lo=wilson_lower(wins, n))
        r.add(9, "PASS", f"{tag} n={n}, WR {wr:.1f}% (95% lower bound {wilson_lower(wins, n):.1f}%), PF(R) {fmt(pf)}, "
                         f"expectancy {exp_r:+.3f}R, avg loser {avg_loss_r:.2f}R")
    # ---- whole account: compounding plausibility ---------------------------------------------
    n = len(L)
    ret = L["ret"].to_numpy(float)
    final = float(CAPITAL + L["net"].sum())
    mult = final / CAPITAL
    span = int((L["exit"].max() - L["entry"].min()) / np.timedelta64(1, "D"))
    cagr = (mult ** (365.0 / span) - 1.0) if (span >= 60 and mult > 0) else float("nan")
    acct = r.meta.setdefault("account", {})
    if not np.isfinite(ret).all():
        r.add(9, "FAIL", "[account] a trade was taken while realized equity was <= 0")
        acct.update(span_days=span)
        return
    logret = np.log1p(np.clip(ret, -0.999999, None))
    g = float(np.expm1(logret.mean()))
    if g > GEO_GROWTH_FAIL:
        r.add(9, "FAIL", f"[account] geometric growth {g:+.2%} per trade (> {GEO_GROWTH_FAIL:.0%}) - not credible for equity intraday")
    elif g > GEO_GROWTH_WARN:
        r.add(9, "WARN", f"[account] geometric growth {g:+.2%} per trade (> {GEO_GROWTH_WARN:.0%}) - exceptional, demand out-of-sample proof")
    if mult > GROWTH_MULT_WARN:
        r.add(9, "WARN", f"[account] {mult:.1f}x growth (> {GROWTH_MULT_WARN:.0f}x) in {span} days - extraordinary; "
                         f"accept only with hidden-holdout / walk-forward evidence")
    pos_lr = logret[logret > 0]
    if n >= 20 and pos_lr.sum() > 0:
        share = float(np.sort(pos_lr)[::-1][:3].sum() / pos_lr.sum())
        if share > TAIL_SHARE_WARN:
            r.add(9, "WARN", f"[account] top-3 trades produce {share:.0%} of all compounded gains - tail-dependent equity curve")
    mdd = float(acct.get("max_dd", 0.0))
    if mdd > 0.40:
        r.add(9, "WARN", f"[account] max drawdown {mdd:.0%} of equity peak")
    acct.update(span_days=span, cagr=cagr, geo_growth_per_trade=g)
    r.add(9, "PASS", f"[account] {n} trades, {mult:.2f}x in {span}d"
                     + (f" (annualised {cagr:.0%})" if np.isfinite(cagr) else " (span < 60d: not annualised)")
                     + f", geometric growth {g:+.2%}/trade, max DD {mdd:.0%}")


# =============================================================================
# 6. RUNNER
# =============================================================================
CHECKS = [(1, check1), (2, check2), (3, check3), (4, check4), (5, check5),
          (6, check6), (7, check7), (8, check8), (9, check9)]
NEEDS_INTRADAY = {2, 4, 5, 6, 9}


def audit_version(v_dir: Path, args, root: Path) -> Report:
    ctx = Ctx(v_dir, args, root)
    try:
        ctx.rep.files["gate"] = f"{Path(__file__).name} sha256:{hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]}"
    except OSError:
        ctx.rep.files["gate"] = "unavailable"
    ctx.rep.files["rules"] = (f"R(E)=min({RISK_PCT:.1%}*E,Rs{RISK_CEILING:g}) x{DD_THROTTLE_FACTOR:g} if DD>{DD_THROTTLE_TRIGGER:.0%} | "
                              f"margin<={MARGIN_BUFFER:.0%}*E @<= {LEVERAGE:g}x | part<={MAX_PARTICIPATION:.0%} | "
                              f"ts={TS_CONVENTION} | swing T+{SWING_SETTLE_BDAYS}")
    for num, fn in CHECKS:
        if num in NEEDS_INTRADAY and num != 1 and "intraday" not in ctx.books and num != 3:
            ctx.rep.add(num, "FAIL", "not executed: trades_intraday.csv missing or invalid (see Check 1)")
            continue
        try:
            fn(ctx)
        except Exception as exc:                                         # noqa: BLE001  fail closed
            tb = traceback.format_exc().strip().splitlines()[-1]
            ctx.rep.add(num, "FAIL", f"check crashed ({type(exc).__name__}: {exc}) | {tb}")
    # a "PASS"-only check never hides a hard fail: nothing to do, status = worst finding
    return ctx.rep


# =============================================================================
# 7. TERMINAL REPORT
# =============================================================================
class Color:
    on = False
    codes = {"PASS": "\033[92m", "WARN": "\033[93m", "FAIL": "\033[91m", "BOLD": "\033[1m", "END": "\033[0m"}

    @classmethod
    def p(cls, text: str, key: str) -> str:
        return f"{cls.codes[key]}{text}{cls.codes['END']}" if cls.on else text


def badge(status: str) -> str:
    return Color.p(f"[{status}]", status)


def print_report(rep: Report, strict: bool, max_log: int) -> None:
    W = 100
    print("=" * W)
    print(f" FORENSIC STRATEGY AUDIT | {rep.version} | {datetime.now():%Y-%m-%d %H:%M}")
    print("=" * W)
    for k, v in rep.files.items():
        print(f"   {k:<14} {v}")
    print("-" * W)
    print(f" {'#':<3}{'CHECK':<44}STATUS")
    for num, title in CHECK_TITLES.items():
        print(f" {num:<3}{title:<44}{badge(rep.check_status(num))}")
    print(f" {10:<3}{'Report & exit code':<44}{badge(rep.verdict(strict))}")
    print("-" * W)
    print(" DETAIL")
    for num, title in CHECK_TITLES.items():
        print(f"  Check {num}: {title}")
        for c, s, msg in rep.findings:
            if c == num:
                print(f"    {badge(s)} {msg}")
    for key in ("intraday", "swing"):
        m = rep.meta.get(key)
        if isinstance(m, dict) and "n" in m:
            print("-" * W)
            print(f" {key.upper()} (recomputed from CSV): trades {m['n']} | WR {m['win_rate']:.1f}% | net Rs {m['net']:.2f} | "
                  f"PF {fmt(m['pf'])} | avg win Rs {m['avg_win']:.2f} | avg loss Rs {m['avg_loss']:.2f}")
    a = rep.meta.get("account")
    if isinstance(a, dict) and "final" in a:
        print("-" * W)
        print(f" ACCOUNT (shared pool, derived by the gate): Rs {a['start']:,.0f} -> Rs {a['final']:,.2f} ({a['growth']:.2f}x) | "
              f"max DD {a['max_dd']:.0%} | largest notional Rs {a['largest_notional']:,.0f} | trades under DD throttle {a['throttled']}")
    if rep.discs:
        print("-" * W)
        print(f" FORENSIC DISCREPANCY LOG ({len(rep.discs)} entries)")
        print(f" {'chk':<4}{'where':<20}{'field':<22}{'expected':<24}{'actual':<24}note")
        for num in CHECK_TITLES:
            rows = [d for d in rep.discs if d["check"] == num]
            for d in rows[:max_log]:
                print(f" {d['check']:<4}{d['where'][:19]:<20}{d['field'][:21]:<22}{d['expected'][:23]:<24}{d['actual'][:23]:<24}{d['note']}")
            if len(rows) > max_log:
                print(f"     ... {len(rows) - max_log} more for check {num} (use --max-log)")
    fails, warns = rep.counts()
    print("=" * W)
    v = rep.verdict(strict)
    msg = {"PASS": "all critical checks passed", "WARN": f"{warns} warning(s), no failures",
           "FAIL": f"{fails} failure(s), {warns} warning(s)" + (" [strict mode]" if strict else "")}[v]
    print(f" VERDICT: {badge(v)}  {rep.version}: {msg}")
    print("=" * W)


def print_summary(reps: List[Report], strict: bool) -> None:
    print("\n" + "=" * 100)
    print(" SUMMARY - ALL VERSIONS")
    print("=" * 100)
    print(f" {'version':<10}{'verdict':<10}{'fails':<7}{'warns':<7}{'trades':<8}{'WR%':<8}{'net Rs':<10}{'PF':<8}")
    for r in reps:
        f, w = r.counts()
        m = r.meta.get("intraday") if isinstance(r.meta.get("intraday"), dict) else None
        n_s = str(m["n"]) if m else "-"
        wr_s = "%.1f" % m["win_rate"] if m else "-"
        net_s = "%.2f" % m["net"] if m else "-"
        pf_s = fmt(m["pf"]) if m else "-"
        print(f" {r.version:<10}{badge(r.verdict(strict)):<10}{f:<7}{w:<7}{n_s:<8}{wr_s:<8}{net_s:<10}{pf_s:<8}")
    print("=" * 100)


# =============================================================================
# 8. CLI
# =============================================================================
VERSION_RE = re.compile(r"^[vV](\d+)$")


def list_versions(root: Path) -> List[Path]:
    vs = [p for p in root.iterdir() if p.is_dir() and VERSION_RE.match(p.name)]
    return sorted(vs, key=lambda p: int(VERSION_RE.match(p.name).group(1)))


def default_root() -> Path:
    here = Path(__file__).resolve().parent
    for c in (here, Path.cwd(), Path.cwd() / "strategies", Path.cwd() / "scratch" / "new_trade" / "strategies", here / "strategies"):
        if c.is_dir() and any(VERSION_RE.match(p.name) for p in c.iterdir() if p.is_dir()):
            return c
    return here


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Independent forensic gate for strategy version folders.")
    ap.add_argument("version", nargs="?", help="version folder name (e.g. V5) or path; with --all, optionally the strategies root")
    ap.add_argument("--all", action="store_true", help="audit every V<n> folder under the strategies root")
    ap.add_argument("--root", help="strategies root (default: auto-detect)")
    ap.add_argument("--data-dir", help="folder holding the raw *_15m_1y.pkl files")
    ap.add_argument("--strict", action="store_true", help="treat WARN as FAIL for the exit code")
    ap.add_argument("--allow-missing-data", action="store_true", help="downgrade missing raw bar data from FAIL to WARN")
    ap.add_argument("--max-log", type=int, default=15, help="max discrepancy-log lines per check (default 15)")
    ap.add_argument("--json", help="also write machine-readable results to this file")
    ap.add_argument("--no-color", action="store_true")
    ap.add_argument("--leverage-table", help="CSV (symbol,leverage) or JSON {symbol: leverage} of Dhan's real per-stock MIS leverage")
    ap.add_argument("--ts-convention", choices=["start", "end"], help="trade timestamps are bar-START (default) or bar-END labels")
    ap.add_argument("--capital", type=float, help="starting equity (default 500)")
    ap.add_argument("--risk-pct", type=float, help="TIGHTEN only: per-trade risk as fraction of E_k (default 0.031)")
    ap.add_argument("--risk-ceiling", type=float, help="TIGHTEN only: absolute Rs risk ceiling (default 250)")
    ap.add_argument("--margin-buffer", type=float, help="TIGHTEN only: fraction of E_k usable as margin (default 0.95)")
    ap.add_argument("--max-participation", type=float, help="TIGHTEN only: max fraction of bar volume (default 0.05)")
    ap.add_argument("--swing-settle-days", type=int, help="LOOSENING (<1) is a FAIL: business days before swing proceeds are usable")
    args = ap.parse_args(argv)
    for flag, name in (("capital", "CAPITAL"), ("risk_pct", "RISK_PCT"), ("risk_ceiling", "RISK_CEILING"),
                       ("margin_buffer", "MARGIN_BUFFER"), ("max_participation", "MAX_PARTICIPATION"),
                       ("swing_settle_days", "SWING_SETTLE_BDAYS"), ("ts_convention", "TS_CONVENTION")):
        if getattr(args, flag) is not None:
            globals()[name] = getattr(args, flag)       # check 5 FAILs any value looser than CONSTITUTION

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")      # Windows consoles
    except Exception:                                                   # noqa: BLE001
        pass
    if sys.stdout.isatty() and not args.no_color and not os.environ.get("NO_COLOR"):
        Color.on = True
        os.system("")                                                   # enable ANSI on Windows 10+

    root = Path(args.root) if args.root else default_root()
    targets: List[Path] = []
    if args.all:
        if args.version and Path(args.version).is_dir() and list_versions(Path(args.version)):
            root = Path(args.version)
        if not root.is_dir():
            print(f"[FAIL] strategies root not found: {root}")
            return 1
        targets = list_versions(root)
        if not targets:
            print(f"[FAIL] no V<n> folders under {root}")
            return 1
    else:
        if not args.version:
            ap.print_usage()
            print("[FAIL] give a version folder or --all")
            return 1
        cands = [Path(args.version), root / args.version, root / args.version.upper(), root / args.version.lower()]
        v_dir = next((c for c in cands if c.is_dir()), cands[0])
        if v_dir.parent.is_dir() and not args.root and v_dir.resolve().parent != root.resolve() and list_versions(v_dir.resolve().parent):
            root = v_dir.resolve().parent
        targets = [v_dir]

    reports = []
    for t in targets:
        rep = audit_version(t, args, root)
        print_report(rep, args.strict, args.max_log)
        reports.append(rep)
    if args.all:
        print_summary(reports, args.strict)
    if args.json:
        payload = [dict(version=r.version, verdict=r.verdict(args.strict), files=r.files,
                        findings=[dict(check=c, status=s, msg=m) for c, s, m in r.findings],
                        discrepancies=r.discs,
                        metrics={k: v for k, v in r.meta.items() if isinstance(v, dict)}) for r in reports]
        Path(args.json).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return 0 if all(r.verdict(args.strict) != "FAIL" for r in reports) else 1


if __name__ == "__main__":
    sys.exit(main())
