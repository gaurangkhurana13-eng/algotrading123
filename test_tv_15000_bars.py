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

def test_15000():
    chart_session = generate_session("cs")
    session = generate_session("qs")
    series_id = "s1"
    symbol = "NSE:PCJEWELLER"
    interval = "15"
    requested_bars = 15000
    
    print(f"Connecting to TradingView WebSocket and requesting {requested_bars} bars directly in create_series...")
    headers = json.dumps({"Origin": "https://data.tradingview.com"})
    ws = create_connection("wss://data.tradingview.com/socket.io/websocket", headers=headers, timeout=10)
    
    ws.send(create_msg("set_auth_token", ["unauthorized_user_token"]))
    ws.send(create_msg("chart_create_session", [chart_session, ""]))
    ws.send(create_msg("quote_create_session", [session]))
    
    sym_spec = '={"symbol":"' + symbol + '","adjustment":"splits","session":"regular"}'
    ws.send(create_msg("resolve_symbol", [chart_session, "symbol_1", sym_spec]))
    
    # Send create_series with 15000 bars
    ws.send(create_msg("create_series", [chart_session, series_id, "s1", "symbol_1", interval, requested_bars]))
    ws.send(create_msg("switch_timezone", [chart_session, "exchange"]))
    
    all_bars = []
    for i in range(50):
        try:
            res = ws.recv()
            if "~h~" in res:
                ws.send(res)
                continue
            if "timescale_update" in res:
                parsed = parse_bars(res)
                if parsed:
                    all_bars.extend(parsed)
            if "series_completed" in res:
                print("series_completed packet received from server!")
                break
        except Exception as e:
            print("Recv err:", e)
            break
            
    df = pd.DataFrame(all_bars).drop_duplicates(subset=["datetime"]).sort_values("datetime")
    print("\n=======================================================")
    print(f"RESULT FOR create_series with {requested_bars} bars:")
    print(f"Total Unique Bars Received: {len(df)}")
    if len(df) > 0:
        print(f"Earliest Date Received: {df['datetime'].min()}")
        print(f"Latest Date Received: {df['datetime'].max()}")
    print("=======================================================")
    ws.close()

if __name__ == "__main__":
    test_15000()
