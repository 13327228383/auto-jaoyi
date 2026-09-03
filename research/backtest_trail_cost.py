# -*- coding: utf-8 -*-
"""
backtest_trail_cost.py —— 跟踪止损「卖后再买」手续费敏感性回测
================================================================================
背景：用户质疑 —— 4% 更紧 → 更容易"卖了又马上买回"，白白浪费手续费。
  需验证：计入真实交易成本后，4% 是否仍优于 6%（还是被换手成本抹平）。

口径：复用 backtest_per_target_trail 决策序列 + 日线仿真。
  · 每次「组合切换」(跨决策段目标变化) = 卖出旧 + 买入新 = 2 边订单。
  · 每次「落袋」(跟踪止损触发) = 1 边卖出；其后若再买入算 2 边。
  · 成本按「每边订单」扣：port_net = port_gross - 当日成本。
  对比成本 0 / 0.05% / 0.1% / 0.2% 每边时 4% vs 6% 的净年化与净回撤。
  （实盘 ETF 佣金约 0.01-0.03%，此处取 0.05-0.1% 已含滑点，0.2% 为极端压力。）

数据：cache/backtest/etf_*.csv（7 只真实 ETF 日线）
输出：命令行对比表
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from backtest_per_target_trail import UNIVERSE, BASE_PARAMS, load_prices, decide_targets_stateful

COST_LEVELS = [0.0, 0.0005, 0.001, 0.002]


def sim_net(price, dec_dates, targets, trail_map, cost):
    """返回(净收益序列, 累计换手订单边数, 落袋数)。cost=每边订单成本(0=毛收益)。"""
    dates = price.index
    daily_ret = price.pct_change().fillna(0.0)
    port = pd.Series(0.0, index=dates)          # 毛收益
    cost_s = pd.Series(0.0, index=dates)        # 当日成本回撤
    prev_key = "cash"
    orders = 0                                  # 订单边数
    exits = 0
    for i in range(1, len(dec_dates)):
        d0, d1 = dec_dates[i - 1], dec_dates[i]
        tgt = targets[d0]
        seg = dates[(dates > d0) & (dates <= d1)]
        if len(seg) == 0:
            continue
        key = "cash" if tgt == "cash" else tuple(sorted(tgt))
        if cost > 0 and key != prev_key:
            # 组合切换：卖出旧(若有) + 买入新(若非cash)；基准按账户满仓切分，保守估每标的各1边
            n = 0
            if prev_key != "cash":
                n += len(prev_key)              # 卖旧
            if key != "cash":
                n += len(key)                   # 买新
            cost_s.loc[seg[0]] = cost * n
            orders += n
        prev_key = key
        if tgt == "cash":
            port.loc[seg] = 0.0
            continue
        _past = price.index[price.index <= d0]
        d0_t = _past[-1] if len(_past) else d0
        held = list(tgt)
        weights = {c: 1.0 for c in tgt}
        peak = {c: float(price.loc[d0_t, c]) for c in tgt}
        for t in seg:
            for c in list(held):
                tr = trail_map.get(c, 0) or 0
                if tr > 0:
                    px = float(price.loc[t, c])
                    peak[c] = max(peak[c], px)
                    if px <= peak[c] * (1 - tr):
                        held.remove(c)
                        exits += 1
                        if cost > 0:
                            cost_s.loc[t] += cost       # 落袋卖出 1 边
                            orders += 1
            if not held:
                port.loc[t] = 0.0
            else:
                port.loc[t] = sum(weights[c] * float(daily_ret.loc[t, c]) for c in held)
    net = port - cost_s          # 净值日收益
    return net, orders, exits


def metrics(eq):
    cum = float(eq.iloc[-1])
    ann = float(cum ** (252.0 / max(len(eq), 1)) - 1) if cum > 0 else -1.0
    maxdd = float(((eq - eq.cummax()) / eq.cummax()).min())
    return ann, maxdd


def main():
    price = load_prices()
    dec_dates = list(price.resample("2W").last().index)
    dec_dates = [d for d in dec_dates if len(price.loc[:d]) >= 200]
    targets = decide_targets_stateful(price, dec_dates, BASE_PARAMS)
    eq0 = (1 + price["510300"].pct_change().fillna(0.0)).cumprod()  # 基准引用
    print("净收益(扣手续费) 4% vs 6% ：")
    print(f"{'每边成本':<9s} {'4%净年化':>9s} {'4%净回撤':>9s} {'4%订单边':>8s} "
          f"{'6%净年化':>9s} {'6%净回撤':>9s} {'6%订单边':>8s} {'Δ净年化(4-6)':>10s}")
    for cost in COST_LEVELS:
        n4, o4, x4 = sim_net(price, dec_dates, targets, {c: 0.04 for c in UNIVERSE}, cost)
        n6, o6, x6 = sim_net(price, dec_dates, targets, {c: 0.06 for c in UNIVERSE}, cost)
        a4, d4 = metrics((1 + n4.fillna(0.0)).cumprod())
        a6, d6 = metrics((1 + n6.fillna(0.0)).cumprod())
        tag = "毛(0)" if cost == 0 else f"{cost*100:.2f}%"
        print(f"{tag:<9s} {a4*100:8.2f}% {d4*100:8.2f}% {o4:<8d} "
              f"{a6*100:8.2f}% {d6*100:8.2f}% {o6:<8d} {(a4-a6)*100:+9.2f}pp")
    print("\n表内 订单边=累计买卖笔数(每边1笔)；实盘ETF佣金约0.01-0.03%/边，0.1-0.2% 已含滑点与压力。")


if __name__ == "__main__":
    main()