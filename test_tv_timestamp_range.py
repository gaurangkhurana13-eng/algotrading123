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
    matches = re.findall(r'"v":\[([0-9\.,\-e]+)\]', raw_text)
    for m in matches:
        parts = m.split(',')
        if len(parts) >= 6:
            ts = datetime.datetime.fromtimestamp(float(parts[0]))
            o, h, l, c, v = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
            bars.append({"datetime": ts, "open": o, "high": h, "low": l, "close": c, "volume": v})
    return bars

def test_timestamp_call(mode="seconds"):
    chart_session = generate_session("cs")
    session = generate_session("qs")
    series_id = "s1"
    symbol = "NSE:PCJEWELLER"
    interval = "15"
    
    dt_from = datetime.datetime(2025, 1, 1, 9, 15)
    dt_to = datetime.datetime(2025, 10, 31, 15, 30)
    
    if mode == "seconds":
        from_ts = int(dt_from.timestamp())
        to_ts = int(dt_to.timestamp())
    else:
        from_ts = int(dt_from.timestamp() * 1000)
        to_ts = int(dt_to.timestamp() * 1000)
        
    print(f"\n--- Testing mode: {mode} (from={from_ts}, to={to_ts}) ---")
    headers = json.dumps({"Origin": "https://data.tradingview.com"})
    ws = create_connection("wss://data.tradingview.com/socket.io/websocket", headers=headers, timeout=10)
    
    ws.send(create_msg("set_auth_token", ["unauthorized_user_token"]))
    ws.send(create_msg("chart_create_session", [chart_session, ""]))
    ws.send(create_msg("quote_create_session", [session]))
    
    sym_spec = '={"symbol":"' + symbol + '","adjustment":"splits","session":"regular"}'
    ws.send(create_msg("resolve_symbol", [chart_session, "symbol_1", sym_spec]))
    
    # Try passing from and to
    payload = [chart_session, series_id, "s1", "symbol_1", interval, from_ts, to_ts]
    print("Sending create_series with payload:", payload)
    ws.send(create_msg("create_series", payload))
    ws.send(create_msg("switch_timezone", [chart_session, "exchange"]))
    
    all_bars = []
    for _ in range(25):
        try:
            res = ws.recv()
            if "~h~" in res:
                ws.send(res)
                continue
            if "error" in res.lower():
                print("Server Error Response:", res[:200])
            if "timescale_update" in res:
                parsed = parse_bars(res)
                if parsed:
                    all_bars.extend(parsed)
            if "series_completed" in res:
                print("Series completed received!")
                break
        except Exception as e:
            print("Recv error:", e)
            break
            
    if all_bars:
        df = pd.DataFrame(all_bars).drop_duplicates(subset=["datetime"]).sort_values("datetime")
        print(f"Success! Captured {len(df)} bars. Range: {df['datetime'].min()} to {df['datetime'].max()}")
    else:
        print("No bars captured in this mode.")
    ws.close()

if __name__ == "__main__":
    test_timestamp_call("seconds")
    test_timestamp_call("milliseconds")
