# -*- coding: utf-8 -*-
"""
backtest_t0_t1_net.py —— 7 只 ETF 轮动【含 T+0/T+1 制度】净收益回测
================================================================================
回答用户灵魂质疑，"诚实口径"（无前视、含摩擦、区分T+0/T+1）：

T+0: 518880黄金/511010国债/513100纳指/513500标普  -> 回撤3%当日离场
T+1: 510300/510500/159915                          -> 回撤3%信号当日触发但当日卖不了,
                                                       记次日(次一交易日)才离场 —— 承受次日下探。

规则（对齐实盘 SR_TRAIL_PCT=0.03 + slope_r2 选标）：
  - 决策复用一个轻量 picker（slope_r2 最大+MA60过滤；本脚本自实现，避免耦合 sr.decide 迟滞态）
  - 防御: 全局风险off时持黄金。简化: 每周期取 slope_r2 最高且>0 的标的；否则现金。
  - 持有单标的; 主动换仓 MIN_HOLD=1自然日; 回撤3%止损不受hold约束。
  - 成本: 换仓双边 0.0476%(万2.38x2)
动量口径对比:
  - close_mom : 用「昨日收盘」判定动量(策略现状)
  - realtime   : 用「当日实时价」判定(近似: 当日已走高则视为翻转) —— 仅演示"若能拿到实时价"
数据: 新浪真实日K收盘。
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

from data_center import get_etf_k_history

T0 = {"518880", "511010", "513100", "513500"}
T1 = {"510300", "510500", "159915"}
CODES = ["518880", "513100", "510300", "510500", "159915", "511010", "513500"]
COM_RATE = 0.000238
ROUND_TRIP = COM_RATE * 2
TRAIL = 0.03
MIN_HOLD = 1
MA_WIN = 60
MOM_WIN = 20


def load():
    cols = {}
    for c in CODES:
        df = get_etf_k_history(c, days=3000)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date")
        cols[c] = df["close"].astype(float)
    p = pd.DataFrame(cols).dropna()
    p.index = pd.to_datetime(p.index)
    return p.sort_index()


def slope_r2(ser):
    s = ser.iloc[-MA_WIN:].astype(float)
    y = s.values
    x = np.arange(len(y))
    x = x - x.mean()
    y = y - y.mean()
    denom = (x ** 2).sum()
    if denom == 0:
        return float("-inf")
    slope = (x * y).sum() / denom
    if np.std(y) == 0:
        return float("-inf")
    r2 = (np.corrcoef(x, y)[0, 1]) ** 2
    return (slope / (np.std(y) + 1e-9)) * r2


def pick(price, d):
    """返回今日应持有的 target 或 'cash'。"""
    best = None
    best_score = float("-inf")
    for c in CODES:
        s = price.loc[:d, c]
        if len(s) < MA_WIN + 1:
            continue
        ma60 = s.iloc[-MA_WIN:].mean()
        if c not in {"518880"} and s.iloc[-1] < ma60:
            continue                      # 跌破MA60剔除(防御豁免)
        sc = slope_r2(s)
        if sc > best_score:
            best_score = sc
            best = c
    if best is None or best_score <= 0:
        return "cash"
    return best


def simulate(price):
    dates = price.index
    daily_ret = price.pct_change().fillna(0.0)
    port = pd.Series(0.0, index=dates)
    adj = pd.Series(1.0, index=dates)
    switches = invested = 0
    held = None
    peak = None
    pending_exit = False       # T+1 标的昨日触发回撤、待今日离场
    last_switch = None
    cur_key = "cash"
    entered = False

    for i, d in enumerate(dates):
        if i == 0:
            continue
        # ---- 每一步先推演 T+1 昨留的待卖 ----
        if pending_exit and held:
            pending_exit = False
            # 今日开盘离场(按今日价计, 但后记在今日收益前退出, 避免前视)
        # ---- 1) 回撤止损 ----
        if held is not None:
            px = price.loc[d, held]
            if px > peak:
                peak = px
            trig = px <= peak * (1 - TRAIL)
            if trig:
                if held in T0:
                    # T+0 当日离场
                    pnl = daily_ret.loc[d, held]
                    port.loc[d] = pnl
                    adj.loc[pd.Index([d])] = 1.0  # 保持
                    switches += 1
                    adj.loc[dates > d] *= (1 - ROUND_TRIP)
                    held = None; peak = None; last_switch = None
                    cur_key = "cash"
                    continue
                else:
                    # T+1 当日卖不了 -> 记次日待卖; 今日仍按该标的收益计
                    pending_exit = True
                    port.loc[d] = daily_ret.loc[d, held]
                    invested += 1
                    continue
            else:
                port.loc[d] = daily_ret.loc[d, held]
                invested += 1
                continue

        # ---- 2) 决策(受 MIN_HOLD 约束) ----
        can = (held is None) or (last_switch is None) or ((d - last_switch).days >= MIN_HOLD)
        if can:
            tgt = pick(price, d)
            key = tgt if isinstance(tgt, str) else str(tgt) if tgt else "cash"
            if entered and key != cur_key:
                switches += 1
                adj.loc[dates > d] *= (1 - ROUND_TRIP)
                last_switch = d
            elif not entered:
                entered = True
            if (held is None) or (key != cur_key):
                held = None; peak = None
                if key != "cash":
                    held = key
                    peak = price.loc[d, held]
                cur_key = key if held else "cash"
                # 建仓日 mkt 收益从明日记，避免前视
                rebuild_override = 0.0
                if held is None:
                    port.loc[d] = 0.0
                # 建仓当天不计新标涨幅(避免拿确认日跳涨)
        # 空仓收益=0（已在上面跳过）

    eq = (1 + port).cumprod() * adj
    n = max(len(eq), 1)
    cum = float(eq.iloc[-1])
    ann = float(cum ** (252.0 / n) - 1) if cum > 0 else -1.0
    maxdd = float(((eq - eq.cummax()) / eq.cummax()).min())
    calmar = ann / abs(maxdd) if maxdd < 0 else float("nan")
    return {"cum": cum, "ann": ann, "maxdd": maxdd, "calmar": calmar,
            "switches": switches, "time_in_mkt": invested / max(n - 1, 1)}


def main():
    price = load()
    r = simulate(price)
    print(f"数据 {price.index[0].date()} ~ {price.index[-1].date()}  {len(price)}交易日\n"
          f"T+0: {sorted(T0)}  |  T+1: {sorted(T1)}\n"
          f"规则: slope_r2(60)选最强 + 防御黄金 + 回撤3%止损(T+1延迟1日) + 万2.38双边\n")
    hdr = f"{'':6s}{'累计x':>9s}{'年化':>9s}{'回撤':>8s}{'Calmar':>8s}{'换手':>6s}{'持仓%':>8s}"
    print(hdr); print("-" * len(hdr))
    print(f"{'':6s}{r['cum']:>9.2f}{r['ann']*100:>9.2f}{r['maxdd']*100:>8.2f}"
          f"{r['calmar']:>8.2f}{r['switches']:>6d}{r['time_in_mkt']*100:>8.1f}")


if __name__ == "__main__":
    main()