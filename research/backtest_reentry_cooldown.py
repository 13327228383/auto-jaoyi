# -*- coding: utf-8 -*-
"""
backtest_reentry_cooldown.py —— 止损后「再入冷却 N 天」扫描：N 该设多少 / 要不要设
================================================================================
背景：用户动机 —— 跟踪止损卖出后，若目标仍在该标的，下一个决策日会立即买回，
  形成「卖了马上买」的抖振，白耗手续费。拟加「止损后再入冷却 N 天」：某标的触发
  落袋止损后，N 个交易日内禁止再被买回，防抄底挨打 / 防抖振。

本脚本回答两件事：
  1) 要不要设冷却（N=0 即不设，对照）；
  2) 若设，N 取多少最优 —— 扫描 N ∈ {0,1,2,3,5,7,10}。

口径：复用 backtest_per_target_trail 的决策序列(实盘参数+迟滞带)与日线逐段仿真，只增加
  「每标的停止再入锁」。当某标的在段内触发跟踪止损(non-cash→cash)，记下其 locked_date；
  之后若决策/cash重建要把该标的买回，仅当 距 lock ≥ N 交易日 才放行，否则跳过该标的
  （宁可少持空，也不立即接回）。段内已持有的其他目标正常持有。
  —— 核心对比 N=0(不限) vs N>0 的年化/回撤/换手/落袋差异。

数据：cache/backtest/etf_*.csv（7 只真实 ETF 日线）
输出：命令行对比表
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from backtest_per_target_trail import (UNIVERSE, BASE_PARAMS, load_prices,
                                       decide_targets_stateful)

COOLDOWNS = [0, 1, 2, 3, 5, 7, 10]   # 交易日


def simulate_cooldown(price, dec_dates, targets, trail_map, cooldown):
    """新增 cooldown 逻辑的逐段仿真。cooldown=止损后禁止再入的交易日数(0=不限)。"""
    dates = price.index
    day_pos = {d: k for k, d in enumerate(dates)}       # 日期 -> 交易日序号
    daily_ret = price.pct_change().fillna(0.0)
    port = pd.Series(0.0, index=dates)
    switches = 0
    peak_exits = 0
    invested_days = 0
    reentries = 0            # 冷却放行后重新买入的标的-段次数(抖振信号)
    locked = {}              # code -> 触发落袋的交易日序号(触发后立即锁)
    prev_key = None
    for i in range(1, len(dec_dates)):
        d0, d1 = dec_dates[i - 1], dec_dates[i]
        tgt = targets[d0]
        seg = dates[(dates > d0) & (dates <= d1)]
        if len(seg) == 0:
            continue
        _past = price.index[price.index <= d0]
        d0_t = _past[-1] if len(_past) else d0
        key = "cash" if tgt == "cash" else tuple(sorted(tgt))
        if prev_key is not None and key != prev_key:
            switches += 1
        prev_key = key
        if tgt == "cash":
            port.loc[seg] = 0.0
            continue
        held = list(tgt)
        weights = {c: 1.0 for c in tgt}
        peak = {c: float(price.loc[d0_t, c]) for c in tgt}
        for t in seg:
            # 先放行到期目标：段起始就把"仍在冷却内"的目标剔除（不接回）
            if t == seg[0] and cooldown > 0 and locked:
                cur_pos = day_pos[t]
                held = [c for c in held
                        if locked.get(c) is None or (cur_pos - locked[c]) >= cooldown]
                if len(held) < len(tgt):
                    reentries += (len(tgt) - len(held))   # 被冷却挡住的接回
            for c in list(held):
                tr = trail_map.get(c, 0) or 0
                if tr > 0:
                    px = float(price.loc[t, c])
                    peak[c] = max(peak[c], px)
                    if px <= peak[c] * (1 - tr):
                        held.remove(c)
                        peak_exits += 1
                        switches += 1
                        locked[c] = day_pos[t]           # 落袋 -> 锁
            if not held:
                port.loc[t] = 0.0
            else:
                port.loc[t] = sum(weights[c] * float(daily_ret.loc[t, c]) for c in held)
            if port.loc[t] != 0.0:
                invested_days += 1
    equity = (1 + port).cumprod()
    return {"equity": equity, "switches": switches, "peak_exits": peak_exits,
            "invested_days": invested_days, "reentries": reentries}


def metrics(equity, start_ts=None):
    if start_ts is not None:
        equity = equity.loc[equity.index >= start_ts]
    if len(equity) == 0:
        return None
    cum = float(equity.iloc[-1])
    ann = float(cum ** (252.0 / max(len(equity), 1)) - 1) if cum > 0 else -1.0
    maxdd = float(((equity - equity.cummax()) / equity.cummax()).min())
    return {"cum": cum, "ann": ann, "maxdd": maxdd}


def main():
    price = load_prices()
    dec_dates = list(price.resample("2W").last().index)
    dec_dates = [d for d in dec_dates if len(price.loc[:d]) >= 200]
    targets = decide_targets_stateful(price, dec_dates, BASE_PARAMS)
    oos_ts = pd.Timestamp("2021-01-01")
    trail = {c: 0.04 for c in UNIVERSE}   # 统一 4%(已落地)
    print(f"数据 {price.index[0].date()} ~ {price.index[-1].date()}，{len(price)} 交易日，调仓 2W")
    print(f"跟踪止损统一 4%，仅增加'止损后再入冷却 N 交易日'\n")
    hdr = (f"{'N天':>4s}{'累计x':>7}{'年化':>8}{'回撤':>8}{'Calmar':>8}"
           f"{'换手':>5}{'落袋':>5}{'接回挡':>6}{'持仓%':>7}{'OOS年化':>9}{'OOS回撤':>9}")
    print(hdr); print("-" * len(hdr))
    rows = {}
    for n in COOLDOWNS:
        r = simulate_cooldown(price, dec_dates, targets, trail, n)
        eq = r["equity"]
        m = metrics(eq); mo = metrics(eq, oos_ts)
        calmar = m["ann"] / abs(m["maxdd"]) if m["maxdd"] < 0 else float("nan")
        tag = " <- 不设" if n == 0 else ""
        print(f"{n:>3d}天{m['cum']:>7.3f}{m['ann']*100:>8.2f}{m['maxdd']*100:>8.2f}"
              f"{calmar:>8.2f}{r['switches']:>5d}{r['peak_exits']:>5d}"
              f"{r['reentries']:>6d}{r['invested_days']/max(len(eq)-1,1)*100:>7.1f}"
              f"{mo['ann']*100:>9.2f}{mo['maxdd']*100:>9.2f}{tag}")
        rows[n] = (r, m, mo)

    print("\n=== 结论：以 N=0(不限) 为基线 ===")
    b = rows[0]; bm = b[1]
    for n in COOLDOWNS[1:]:
        r, m, mo = rows[n]
        print(f"  N={n:>2d}天: Δ年化{(m['ann']-bm['ann'])*100:+7.2f}pp  Δ回撤"
              f"{(m['maxdd']-bm['maxdd'])*100:+7.2f}pp  Δ换手{r['switches']-b[0]['switches']:+4d}  "
              f"Δ落袋{r['peak_exits']-b[0]['peak_exits']:+4d}  OOSΔ年化{(mo['ann']-b[2]['ann'])*100:+7.2f}pp")


if __name__ == "__main__":
    main()