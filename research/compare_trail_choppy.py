# -*- coding: utf-8 -*-
"""
compare_trail_choppy.py —— 4% vs 6% 跟踪止损「震荡市」对比
================================================================================
目标：回答"4% 更紧，在震荡市会不会被来回扇(高换手)而更差？"

口径：复用 backtest_per_target_trail 的决策序列与仿真。额外识别「震荡/区间」交易日：
  对宽基 510300 收盘取 20 日对数线性回归，slope 的 t 统计量 |t|<1.5 → 无显著趋势 → 判定为震荡段。
  把全部 4% 和全部 6% 组合的日收益按「震荡日 / 趋势日」切分，分别看年化、回撤、落袋数。
  另统计"落袋日是否落在震荡日"（即磨止损的换手是否集中在震荡市）。

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

LOOKBACK = 60       # 判定趋势的回看窗口（交易日，对齐策略 MA_FILTER=60）
CHOPPY_SHARPE = 0.40  # |60日 Sharpe|=|漂移/波动| 低于此值 → 视为震荡/区间段


def simulate_trail_dates(price, dec_dates, targets, trail_map):
    """同 backtest_per_target_trail.simulate_trail，但返回日收益+每次落袋日期，便于按震荡日切分。"""
    dates = price.index
    daily_ret = price.pct_change().fillna(0.0)
    port = pd.Series(0.0, index=dates)
    exit_days = []
    switches = 0
    for i in range(1, len(dec_dates)):
        d0, d1 = dec_dates[i - 1], dec_dates[i]
        tgt = targets[d0]
        seg = dates[(dates > d0) & (dates <= d1)]
        if len(seg) == 0:
            continue
        if tgt == "cash":
            continue
        _past = price.index[price.index <= d0]
        d0_t = _past[-1] if len(_past) else d0
        held = list(tgt)
        weights = {c: 1.0 for c in tgt}
        peak = {c: float(price.loc[d0_t, c]) for c in tgt}   # 建仓价重置峰值
        for t in seg:
            for c in list(held):
                tr = trail_map.get(c, 0) or 0
                if tr > 0:
                    px = float(price.loc[t, c])
                    peak[c] = max(peak[c], px)
                    if px <= peak[c] * (1 - tr):
                        held.remove(c)
                        exit_days.append(t)
                        switches += 1
            if not held:
                port.loc[t] = 0.0
            else:
                port.loc[t] = sum(weights[c] * float(daily_ret.loc[t, c]) for c in held)
        # 段结束（决策日切换）时重置 peek 初始化
    return port, exit_days, switches


def main():
    price = load_prices()
    dec_dates = list(price.resample("2W").last().index)
    dec_dates = [d for d in dec_dates if len(price.loc[:d]) >= 200]
    targets = decide_targets_stateful(price, dec_dates, BASE_PARAMS)
    dates = price.index

    # 震荡日判定：宽基 510300 的 60日 Sharpe = 60日漂移 / 60日波动（低绝对Sharpe=区间震荡）
    broad_ret = price["510300"].astype(float).pct_change()
    sharpe = broad_ret.rolling(LOOKBACK).mean() * LOOKBACK / broad_ret.rolling(LOOKBACK).std()
    choppy = pd.Series(sharpe.abs() < CHOPPY_SHARPE, index=dates)
    n_c = int(choppy.sum()); n_t = int((~choppy).sum())
    a4, e4, s4 = simulate_trail_dates(price, dec_dates, targets, {c: 0.04 for c in UNIVERSE})
    a6, e6, s6 = simulate_trail_dates(price, dec_dates, targets, {c: 0.06 for c in UNIVERSE})

    def seg_metrics(ret, mask, exits, label):
        seg = ret[mask]
        day_ret = (seg + 1.0).prod() - 1.0
        eq = (1.0 + seg.fillna(0.0)).cumprod()
        mdd = float(((eq - eq.cummax()) / eq.cummax()).min())
        n_exit = sum(1 for d in exits if choppy.loc[d]) if mask is choppy else sum(
            1 for d in exits if not choppy.loc[d])
        print(f"  {label:<16s} 段内日数{mask.sum():5d} 段内累计{day_ret*100:+7.2f}% "
              f"段内回撤{mdd*100:7.2f}% 段内落袋{n_exit:3d} 总落袋{len(exits):3d}")
        return day_ret, mdd

    print(f"数据 {price.index[0].date()} ~ {price.index[-1].date()}  震荡日{n_c} / 趋势日{n_t} (|60日Sharpe|<{CHOPPY_SHARPE})")
    print("\n方案  全部4%  总落袋", len(e4), " 全部6%  总落袋", len(e6))
    print("---- 震荡日(区间/无趋势)表现 ----")
    seg_metrics(a4, choppy, e4, "4% 震荡日")
    seg_metrics(a6, choppy, e6, "6% 震荡日")
    print("---- 趋势日表现 ----")
    seg_metrics(a4, ~choppy, e4, "4% 趋势日")
    seg_metrics(a6, ~choppy, e6, "6% 趋势日")

    # 落袋换手集中在哪：震荡日占比
    c4 = sum(1 for d in e4 if choppy.loc[d]); c6 = sum(1 for d in e6 if choppy.loc[d])
    print(f"\n落袋发生在震荡日占比：4% = {c4}/{len(e4)} = {c4/max(len(e4),1)*100:.0f}%  "
          f"6% = {c6}/{len(e6)} = {c6/max(len(e6),1)*100:.0f}%")


if __name__ == "__main__":
    main()