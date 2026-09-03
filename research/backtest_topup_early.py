# -*- coding: utf-8 -*-
"""backtest_topup_early.py —— 「持仓≠目标(被持仓锁卡住)时，抢投新目标 vs 不买」回测
================================================================================
背景(实盘咨询)：当策略的目标已换成新 ETF B、但当前持仓还是旧 A（被 MIN_HOLD 持仓锁
或未到止损、暂时不能换仓）时，账上若有闲钱：
  - 政策"抢投(买 B)"：立刻把闲钱买成新目标 B（提前上车）
  - 政策"不买(等)"：闲钱先放着，等锁过正常再平衡再配 B
本脚本量化哪种政策收益更大，以及 收益率 / 保本率 / 亏损率。

方法：逐日模拟（与生产同配置：GOLD黄金防御+动量门10d+7标的+跟踪6%+入场过滤），
每天(不受锁门限制)先算"模型新目标"，再模拟真实持仓切换。凡出现
「当前持仓≠模型新目标」的【背离窗口】(即被锁/未来得及换的窗口)，记录该窗口内
新目标 B 的区间收益 —— 这正是"抢投 B"相对"不买(0收益)"多赚(或多亏)的部分。
汇总所有窗口：均值收益率、保本率(B窗口收益≥0)、亏损率(B窗口收益<0)、平均赢/亏，
以及单位资金两政策下的累计倍数对比。
run(MIN_HOLD) 取不同锁长：1(当前落地) / 3 / 5(旧版) / 10 —— 看锁越长抢投越划算与否。
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import backtest_minhold as mh
import backtest_universe_ext as bue
import strategy_rotator as sr

UNI = bue.BASE_CODES + ["513500"]
GOLD = bue.GOLD
DEF = "518880"
MOMD = 10


def run(price, min_hold):
    dates = price.index
    held = set()
    entry, stop, peak = {}, {}, {}
    last_switch = None
    eps = []                 # 背离窗口列表: dict(start,end,B,ret,days)
    cur_ep = None

    for i, d in enumerate(dates):
        if i == 0:
            continue
        # 1) 模型新目标（每天都不受锁限制地算，模拟"实盘预判到但暂时没换"）
        tgt, _ = mh.day_target(price, d, GOLD, universe=UNI)
        if isinstance(tgt, (list, tuple)) and tgt and str(tgt[0]) == DEF:
            dm = sr.momentum(pd.Series(price.loc[:d, DEF]), MOMD)
            if dm is None or np.isnan(dm) or dm <= 0:
                tgt = "cash"
        mkey = "cash" if tgt == "cash" else tuple(sorted(str(c) for c in tgt))

        # 2) 持仓内部止损（与生产一致）
        if held:
            changed = False
            for c in list(held):
                px = price.loc[d, c]
                if c == DEF:
                    continue                      # 防御资产不加跟踪/S-R(动量门已接管)
                if peak.get(c) is not None and px > peak[c]:
                    peak[c] = px
                    ts = peak[c] * 0.94
                    if stop.get(c) is None or ts > stop[c]:
                        stop[c] = ts
                if stop.get(c) is not None and px <= stop[c]:
                    held.discard(c); changed = True
            if changed:
                last_switch = None

        # 3) 真实换仓（受锁约束）
        can = (not held) or (last_switch is None) or ((d - last_switch).days >= min_hold)
        if can and mkey != "cash" and mkey != tuple(sorted(held)):
            held = set(); entry, stop, peak = {}, {}, {}
            for c in mkey:
                px = price.loc[d, c]
                held.add(c); entry[c] = px; stop[c] = None; peak[c] = px
            last_switch = d
        held_key = tuple(sorted(held)) if held else None

        # 4) 背离窗口检测
        divergence = bool(held) and mkey != "cash" and mkey != held_key
        if divergence:
            if cur_ep is None:
                cur_ep = {"start": d, "B": str(mkey[0])}
            cur_ep["end"] = d
        else:
            if cur_ep is not None:
                _close_ep(cur_ep, price, eps)
                cur_ep = None
    if cur_ep is not None:
        _close_ep(cur_ep, price, eps)
    return eps


def _close_ep(ep, price, eps):
    B = ep.get("B")
    if B not in price.columns:
        return
    p0 = price.loc[ep["start"], B]
    p1 = price.loc[ep["end"], B]
    if p0 and p0 > 0 and p1 > 0:
        ep["ret"] = p1 / p0 - 1.0
        ep["days"] = (ep["end"] - ep["start"]).days
        eps.append(ep)


def report(label, eps):
    if not eps:
        print(f"  {label:16s} 无背离窗口")
        return
    rets = [e["ret"] for e in eps]
    n = len(rets)
    wins = [r for r in rets if r >= 0]
    loss = [r for r in rets if r < 0]
    mean = float(np.mean(rets))
    br = len(wins) / n                 # 保本率 P(B收益≥0)
    lr = len(loss) / n                 # 亏损率 P(B收益<0)
    avg_win = float(np.mean(wins)) if wins else 0.0
    avg_loss = float(np.mean(loss)) if loss else 0.0
    mult_buy = float(np.prod([1 + r for r in rets]))     # 1元每次抢投滚动复利(不含手续费)
    print(f"  {label:16s} 窗口{n}个 | 均值涨 {mean*100:6.2f}% | 保本率 {br*100:5.1f}% | "
          f"亏损率 {lr*100:5.1f}% | 平均赢 {avg_win*100:6.2f}% 平均亏 {avg_loss*100:6.2f}%  "
          f"(每次投1元复利→{mult_buy:.3f}x)")


def main():
    price = bue.load_prices()
    print(f"数据 {price.index[0].date()} ~ {price.index[-1].date()}，{len(price)} 交易日，标的={len(UNI)}只")
    print(f"配置 = GOLD黄金防御+动量门{MOMD}d + 7标的 + 跟踪6% ；问：背离窗口['新目标B']抢投 vs 不等\n")
    for hold in [1, 3, 5, 10]:
        eps = run(price, hold)
        report(f"MIN_HOLD={hold}d", eps)
    print("\n说明：收益率均值>0 且保本率高 → 抢投新目标划算；为负/保本率低 → 说明追新 top 易踩坑，"
          "等锁过再配更稳。窗口收益即「买新(相对不买多赚)」的部分，不买=0收益。")


if __name__ == "__main__":
    main()