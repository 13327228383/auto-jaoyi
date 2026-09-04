# -*- coding: utf-8 -*-
"""
backtest_gold_mom_net.py —— 黄金(518880)防御择时 【含真实摩擦】 动量窗口对比
================================================================================
修正 backtest_gold_mom_window.py 未扣换手成本的 bug：每次持仓状态切换扣真实双边
成本(万2.38单边 -> 0.0476%双边 ROUN_TRIP)，得到"净收益"。

口径（对齐实盘 auto_run.py DEF_MOM 迟滞带）：
  - 标的 518880 黄金全历史真实收盘(新浪.fund_etf_hist_sina 日K)。黄金 T+0。
  - 动量 m = close[t]/close[t-W]-1
    m > +0.5% 未持 -> 买入；m < -0.8% 已持 -> 卖出；否则维持(迟滞防抖)。
  - 成本：状态切换每次扣 ROUND_TRIP=0.000476 (万2.38单边x2，满仓换仓)。
  - 指标：累计/年化/最大回撤/Calmar/换手次数/持仓占比
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from data_center import get_etf_k_history

ENTER = 0.005
EXIT = -0.008
COM_RATE = 0.000238
ROUND_TRIP = COM_RATE * 2


def load_gold():
    df = get_etf_k_history("518880", days=4000)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    return df["close"].astype(float)


def simulate(close, window):
    hist = close.values
    ret = close.pct_change().fillna(0.0).values
    n = len(hist)
    held = 0
    equity = 1.0
    peak = 1.0
    maxdd = 0.0
    switches = held_days = 0
    for i in range(1, n):
        if i >= window:
            m = hist[i] / hist[i - window] - 1.0
            if held == 0 and m > ENTER:
                held = 1
                switches += 1
                equity *= (1 - ROUND_TRIP)          # 买入扣双边成本
            elif held == 1 and m < EXIT:
                held = 0
                switches += 1
                equity *= (1 - ROUND_TRIP)          # 卖出扣双边成本
        port = ret[i] if held else 0.0
        if held:
            held_days += 1
        equity *= (1 + port)
        peak = max(peak, equity)
        dd = (equity - peak) / peak
        maxdd = min(maxdd, dd)
    ann = equity ** (252.0 / n) - 1 if equity > 0 else -1.0
    calmar = ann / abs(maxdd) if maxdd < 0 else float("nan")
    return {"window": window, "cum": equity, "ann": ann, "maxdd": maxdd,
            "calmar": calmar, "switches": switches, "hold%": held_days / n}


def main():
    close = load_gold()
    print(f"黄金518880 {close.index[0].date()}~{close.index[-1].date()} {len(close)}交易日 (T+0黄金, 含手续费万2.38双边)\n")
    hdr = f"{'动量窗口':>8s}{'累计x':>9s}{'净年化':>9s}{'回撤':>8s}{'Calmar':>8s}{'换手':>6s}{'持仓%':>8s}"
    print(hdr); print("-" * len(hdr))
    for w in [1, 3, 5, 10, 20, 30]:
        r = simulate(close, w)
        print(f"{r['window']:>4d}日  {r['cum']:>8.2f}{r['ann']*100:>9.2f}{r['maxdd']*100:>8.2f}"
              f"{r['calmar']:>8.2f}{r['switches']:>6d}{r['hold%']*100:>8.1f}")


if __name__ == "__main__":
    main()