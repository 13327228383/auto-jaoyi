# -*- coding: utf-8 -*-
"""
backtest_realtime_mom.py —— 实时价翻正买入 vs 收盘价决策 回测
================================================================================
回应用户："动量用实时价翻正买，然后立即监控止损。为什么不用实时？"

此前只能收盘价，因 data_center 只给日 close。但新浪 ETF 日K含 open/high/low！
本脚本用 high/low 做"盘中实时"近似（无前视，只用当日已发生极值）：

  - 实时动量翻正：动量用【当日 high】判盘中已翻正 -> 当日买入
  - 实时止损：持仓当日 low 跌破 峰值*0.97(回撤3%) -> 当日即离场
  - 对比: 收盘价版本(close 判动量 + close 判止损)

仅 T+0 标的（全 T0 扩展池），可当日回转。含万2.38双边摩擦。
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)

import akshare as ak

COM_RATE = 0.000238
ROUND_TRIP = COM_RATE * 2
TRAIL = 0.03
MOM_WIN = 20
MA_WIN = 60

T0_EXP = ["513100", "513500", "513000", "513030", "510900", "513050",
          "511010", "511520", "511380", "518880", "512400"]
SYMS = {c: ("sh" if c.startswith("5") else "sz") + c for c in T0_EXP}


def load():
    cols_o, cols_h, cols_l, cols_c = {}, {}, {}, {}
    for c, sym in SYMS.items():
        df = ak.fund_etf_hist_sina(symbol=sym)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date")
        cols_o[c] = df["open"].astype(float)
        cols_h[c] = df["high"].astype(float)
        cols_l[c] = df["low"].astype(float)
        cols_c[c] = df["close"].astype(float)
    idx = pd.DataFrame(cols_c).dropna().index
    o = pd.DataFrame(cols_o).reindex(idx)
    h = pd.DataFrame(cols_h).reindex(idx)
    l = pd.DataFrame(cols_l).reindex(idx)
    c = pd.DataFrame(cols_c).reindex(idx)
    return o, h, l, c


def slope_r2(ser):
    s = ser.iloc[-MA_WIN:].astype(float)
    y = s.values; x = np.arange(len(y)); x = x - x.mean(); y = y - y.mean()
    den = (x**2).sum()
    if den == 0: return float("-inf")
    slope = (x*y).sum()/den
    if np.std(y) == 0: return float("-inf")
    r2 = np.corrcoef(x, y)[0, 1]**2
    return (slope/(np.std(y)+1e-9))*r2


def pick(c, d):
    best, bs = None, float("-inf")
    for cod in T0_EXP:
        s = c[cod].loc[:d]
        if len(s) < MA_WIN+1: continue
        if cod != "518880" and s.iloc[-1] < s.iloc[-MA_WIN:].mean(): continue
        sc = slope_r2(s)
        if sc > bs: bs, best = sc, cod
    return best if best and bs > 0 else "cash"


def simulate(hl, close_based=False):
    o, h, l, c = hl
    dates = c.index
    daily_c = c.pct_change().fillna(0.0)
    port = pd.Series(0.0, index=dates)
    adj = pd.Series(1.0, index=dates)
    switches = invested = 0
    held = None; peak = None; entered = False
    for d in dates:
        if d == dates[0]: continue
        # 1) 止损
        if held is not None:
            hi = h[held].loc[d] if not close_based else c[held].loc[d]
            lo = l[held].loc[d] if not close_based else c[held].loc[d]
            if close_based:
                ref_peak = peak
                if hi > ref_peak: peak = hi
                if c[held].loc[d] <= ref_peak * (1 - TRAIL):
                    port.loc[d] = daily_c[held].loc[d]
                    adj.loc[dates > d] *= (1 - ROUND_TRIP)
                    switches += 1; held = None; peak = None; entered = True
                    continue
            else:
                # 实时版：用 low 判定跌破
                if hi > peak: peak = hi
                if lo <= peak * (1 - TRAIL):
                    ret = lo / peak - 1.0
                    port.loc[d] = ret
                    adj.loc[dates > d] *= (1 - ROUND_TRIP)
                    switches += 1; held = None; peak = None; entered = True
                    continue
        # 2) 买入
        if held is None:
            tgt = pick(c, d)
            if tgt != "cash":
                s = c[tgt].loc[:d]
                ref = h[tgt].loc[d] if not close_based else s.iloc[-1]
                mom = 0.0
                if len(s) > MOM_WIN:
                    mom = ref / s.iloc[-1-MOM_WIN] - 1.0
                if mom > 0.005:
                    held = tgt
                    peak = h[tgt].loc[d] if not close_based else s.iloc[-1]
                    entered = True
        if held is None:
            port.loc[d] = 0.0
        else:
            port.loc[d] = daily_c[held].loc[d]
            invested += 1
    eq = (1+port).cumprod()*adj
    n = max(len(eq),1)
    cum = float(eq.iloc[-1])
    ann = cum**(252.0/n)-1 if cum > 0 else -1.0
    maxdd = float(((eq-eq.cummax())/eq.cummax()).min())
    calmar = ann/abs(maxdd) if maxdd < 0 else float("nan")
    return dict(cum=cum, ann=ann, maxdd=maxdd, calmar=calmar, switches=switches,
                tim=invested/max(n-1,1))


def main():
    hl = load()
    c = hl[3]
    print(f"数据 {c.index[0].date()} ~ {c.index[-1].date()}  {len(c)}交易日, T+0扩展{len(T0_EXP)}只\n")
    hdr = f"{'模式':28s}{'累计x':>8}{'年化':>9}{'回撤':>8}{'Calmar':>8}{'换手':>6}"
    print(hdr); print("-"*len(hdr))
    for name, cb in [("收盘价决策(close判)", True), ("实时价(high翻正+low止损)", False)]:
        r = simulate(hl, close_based=cb)
        print(f"{name:28s}{r['cum']:>8.2f}{r['ann']*100:>9.2f}{r['maxdd']*100:>8.2f}"
              f"{r['calmar']:>8.2f}{r['switches']:>6d}")


if __name__ == "__main__":
    main()