import json
import random
import string
import re
import datetime
import pandas as pd
from websocket import create_connection

def generate_session(prefix):
    letters = string.ascii_lowercase
    return f"{prefix}_" + "".join(random.choice(letters) for _ in range(12))

def prepend_header(st):
    return f"~m~{len(st)}~m~{st}"

def construct_message(func, param_list):
    return json.dumps({"m": func, "p": param_list}, separators=(",", ":"))

def create_msg(func, param_list):
    return prepend_header(construct_message(func, param_list))

def parse_bars(raw_text):
    bars = []
    # TradingView sends bars in "s":[{"i":..., "v":[timestamp, open, high, low, close, volume]}]
    matches = re.findall(r'"v":\[([0-9\.,\-e]+)\]', raw_text)
    for m in matches:
        parts = m.split(',')
        if len(parts) >= 6:
            ts = datetime.datetime.fromtimestamp(float(parts[0]))
            o, h, l, c, v = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            bars.append({"datetime": ts, "open": o, "high": h, "low": l, "close": c, "volume": v})
    return bars

def test_fetch_older():
    chart_session = generate_session("cs")
    session = generate_session("qs")
    series_id = "s1"
    symbol = "NSE:PCJEWELLER"
    interval = "15"
    
    headers = json.dumps({"Origin": "https://data.tradingview.com"})
    print("Connecting to TradingView WebSocket...")
    ws = create_connection("wss://data.tradingview.com/socket.io/websocket", headers=headers, timeout=10)
    
    # 1. Handshake
    ws.send(create_msg("set_auth_token", ["unauthorized_user_token"]))
    ws.send(create_msg("chart_create_session", [chart_session, ""]))
    ws.send(create_msg("quote_create_session", [session]))
    
    # 2. Resolve Symbol
    sym_spec = '={"symbol":"' + symbol + '","adjustment":"splits","session":"regular"}'
    ws.send(create_msg("resolve_symbol", [chart_session, "symbol_1", sym_spec]))
    
    # 3. Create Series (Initial Batch)
    print("Sending create_series for first 5000 bars...")
    ws.send(create_msg("create_series", [chart_session, series_id, "s1", "symbol_1", interval, 5000]))
    ws.send(create_msg("switch_timezone", [chart_session, "exchange"]))
    
    all_bars = []
    earliest_date = None
    
    # Listen for initial batch
    raw_acc = ""
    for _ in range(50):
        try:
            res = ws.recv()
            raw_acc += res
            if "timescale_update" in res or "series_completed" in res:
                parsed = parse_bars(res)
                if parsed:
                    all_bars.extend(parsed)
            if "series_completed" in res:
                print(f"Initial batch completed! Bars captured so far: {len(all_bars)}")
                break
        except Exception as e:
            print("Recv timeout/err:", e)
            break
            
    if all_bars:
        df_init = pd.DataFrame(all_bars).drop_duplicates(subset=["datetime"]).sort_values("datetime")
        print(f"Initial Range: {df_init['datetime'].min()} to {df_init['datetime'].max()} ({len(df_init)} bars)")
        
        # 4. Now send request_more_data to page backward into 2025!
        for page in range(1, 4):
            print(f"\n[Page {page}] Sending request_more_data (requesting 5000 older bars)...")
            ws.send(create_msg("request_more_data", [chart_session, series_id, 5000]))
            
            new_bars_count = 0
            for _ in range(30):
                try:
                    res = ws.recv()
                    if "~h~" in res: # echo heartbeat
                        ws.send(res)
                        continue
                    print(f"Received msg (page {page}): {res[:180]}...")
                    if "timescale_update" in res:
                        older = parse_bars(res)
                        if older:
                            all_bars.extend(older)
                            new_bars_count += len(older)
                    if "series_completed" in res:
                        print("series_completed received for page", page)
                        break
                    if "critical_error" in res or "error" in res.lower():
                        print("Error on page:", res[:300])
                        break
                except Exception as e:
                    print("Exception on recv:", e)
                    break
                    
            df_cur = pd.DataFrame(all_bars).drop_duplicates(subset=["datetime"]).sort_values("datetime")
            print(f"[Page {page}] New bars added: {new_bars_count} | New Earliest Date: {df_cur['datetime'].min()}")
            
            # Check if we hit Jan 2025 or stopped receiving older bars
            if new_bars_count == 0 or df_cur['datetime'].min().year <= 2025:
                break
                
        df_final = pd.DataFrame(all_bars).drop_duplicates(subset=["datetime"]).sort_values("datetime")
        print("\n=======================================================")
        print(f"REVERSE ENGINEERING RESULT FOR {symbol}:")
        print(f"Total Unique Bars: {len(df_final)}")
        print(f"Earliest Date: {df_final['datetime'].min()}")
        print(f"Latest Date: {df_final['datetime'].max()}")
        print("=======================================================")
    else:
        print("No bars parsed from initial batch.")
        
    ws.close()

if __name__ == "__main__":
    test_fetch_older()
