import sys
import time
import json
import urllib.request
import urllib.error
import pandas as pd
from pathlib import Path

# Security ID Mapping for the 30 Core Stocks
SEC_ID_MAP = {
    'CENTRALBK': '14894',
    'DELTACORP': '15044',
    'ELECTCAST': '18116',
    'GSFC': '1247',
    'IDEA': '14366',
    'IFCI': '1491',
    'IOB': '9348',
    'IRCON': '4986',
    'J&KBANK': '5633',
    'JYOTHYLAB': '15146',
    'MANALIPETC': '1855',
    'MTNL': '2256',
    'NBCC': '2042',
    'NCC': '2304',
    'NIACL': '21731',
    'NMDC': '15332',
    'PCJEWELLER': '15865',
    'PNB': '10666',
    'RBLBANK': '18391',
    'RCF': '1655',
    'RENUKA': '14304',
    'RITES': '4990',
    'RPOWER': '14428',
    'SUZLON': '12018',
    'SWSOLAR': '22708',
    'TATASTEEL': '3499',
    'TEXRAIL': '14732',
    'TTML': '13936',
    'UTKARSHBNK': '21033',
    'ZEEL': '3812'
}

QUARTERS = [
    ('2021-01-01', '2021-03-31', 'Q1'),
    ('2021-04-01', '2021-06-30', 'Q2'),
    ('2021-07-01', '2021-09-30', 'Q3'),
    ('2021-10-01', '2021-12-31', 'Q4'),
]

def fetch_quarter(sec_id, client_id, access_token, from_date, to_date):
    url = "https://api.dhan.co/v2/charts/intraday"
    payload = {
        "securityId": str(sec_id),
        "exchangeSegment": "NSE_EQ",
        "instrument": "EQUITY",
        "interval": "15",
        "oi": "false",
        "fromDate": from_date,
        "toDate": to_date
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "access-token": access_token.strip()
    }
    if client_id:
        headers["client-id"] = client_id.strip()

    req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            if resp.status == 200:
                body = resp.read().decode('utf-8')
                return json.loads(body)
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode('utf-8', errors='ignore')
        return {"error": f"HTTP {e.code}: {err_msg}"}
    except Exception as e:
        return {"error": str(e)}
    return None

def main():
    if len(sys.argv) >= 3:
        client_id = sys.argv[1].strip()
        access_token = sys.argv[2].strip()
    else:
        print("\n" + "="*70)
        print(" DHAN 2021 15-MINUTE DATA DOWNLOADER (SECURE LOCAL MODE)")
        print("="*70)
        client_id = input("Enter Dhan Client ID: ").strip()
        access_token = input("Enter Dhan Access Token: ").strip()

    if not client_id or not access_token:
        print("[ERROR] Client ID and Access Token are required!")
        sys.exit(1)

    out_dir = Path("data/2021-22")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print(f" TARGET: 30 Core Stocks | Year: 2021 Full (Jan 1 - Dec 31, 2021)")
    print(f" OUTPUT: {out_dir.resolve()}")
    print("="*70 + "\n")

    universe_30 = {}
    total_symbols = len(SEC_ID_MAP)
    success_count = 0
    first_error_shown = False

    for i, (symbol, sec_id) in enumerate(sorted(SEC_ID_MAP.items()), 1):
        print(f"[{i:02d}/{total_symbols}] {symbol:<12} (SecID: {sec_id:<6}) ... ", end="", flush=True)

        dfs = []
        quarter_statuses = []

        for from_d, to_d, q_label in QUARTERS:
            data = fetch_quarter(sec_id, client_id, access_token, from_d, to_d)
            time.sleep(0.20)  # Safe throttle

            if data and isinstance(data, dict) and "open" in data and len(data["open"]) > 0:
                try:
                    df_q = pd.DataFrame({
                        "open": data["open"],
                        "high": data["high"],
                        "low": data["low"],
                        "close": data["close"],
                        "volume": data["volume"]
                    })
                    df_q["datetime"] = pd.to_datetime(data["timestamp"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
                    df_q.set_index("datetime", inplace=True)
                    dfs.append(df_q)
                    quarter_statuses.append(f"{q_label}:OK({len(df_q)})")
                except Exception as ex:
                    quarter_statuses.append(f"{q_label}:PARSE_ERR")
            elif data and "error" in data:
                err_text = data["error"]
                quarter_statuses.append(f"{q_label}:ERR")
                if not first_error_shown:
                    print(f"\n   [API RESPONSE DETAIL] -> {err_text}")
                    first_error_shown = True
            else:
                quarter_statuses.append(f"{q_label}:EMPTY")

        if dfs:
            df_full = pd.concat(dfs).sort_index()
            df_full = df_full[~df_full.index.duplicated(keep="first")]

            ind_path = out_dir / f"{symbol.lower()}_15m_1y.pkl"
            df_full.to_pickle(ind_path)

            universe_30[symbol] = df_full
            universe_30[f"{symbol}.NS"] = df_full
            success_count += 1
            print(f"[{' | '.join(quarter_statuses)}] -> Total: {len(df_full):,} bars [SAVED]")
        else:
            print(f"[{' | '.join(quarter_statuses)}] -> FAILED (No Data)")

    if universe_30:
        master_path = out_dir / "universe_30_15m.pkl"
        pd.to_pickle(universe_30, master_path)
        print("\n" + "="*70)
        print(f" SUCCESS! {success_count}/{total_symbols} stocks downloaded.")
        print(f" Master Dataset Saved at: {master_path}")
        print("="*70)
    else:
        print("\n" + "="*70)
        print(" [FAILED] Zero symbols downloaded. Please inspect the API Response Detail above.")
        print("="*70)

if __name__ == "__main__":
    main()
