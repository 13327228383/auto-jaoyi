# -*- coding: utf-8 -*-
"""抓取 7 只实盘 ETF 的完整日线 OHLC（新浪），供盘中高点回落(快循环)回测。
存 cache/backtest/ohlc_<code>.csv，列：date,open,high,low,close。"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import pandas as pd
import akshare as ak

CACHE = os.path.join(os.path.dirname(HERE), "cache", "backtest")
UNIVERSE = {"510300": "sh510300", "510500": "sh510500", "159915": "sz159915",
            "518880": "sh518880", "513100": "sh513100", "511010": "sh511010",
            "513500": "sh513500"}

if __name__ == "__main__":
    ok = 0
    for code, sym in UNIVERSE.items():
        cf = os.path.join(CACHE, f"ohlc_{code}.csv")
        if os.path.exists(cf) and os.path.getsize(cf) > 1024:
            df = pd.read_csv(cf, dtype={"date": str})
            print(f"{code}: 已缓存 n={len(df)} {df['date'].min()}~{df['date'].max()}")
            ok += 1
            continue
        df = ak.fund_etf_hist_sina(symbol=sym)
        df = df[["date", "open", "high", "low", "close"]].copy()
        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
        for c in ["open", "high", "low", "close"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["close"])
        df.to_csv(cf, index=False)
        print(f"{code}({sym}): n={len(df)} {df['date'].min()}~{df['date'].max()}  last_close={df['close'].iloc[-1]:.3f}")
        ok += 1
    print(f"\n共 OK {ok}/{len(UNIVERSE)}")