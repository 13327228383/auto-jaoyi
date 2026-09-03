# -*- coding: utf-8 -*-
"""backtest_cash_deploy.py —— 「多余现金：买(满仓) vs 不买(闲置)」回测
================================================================================
背景(实盘咨询)：用户持有了 5 万目标持仓后又充入 2 万 → 目标未变时程序只做心跳、
不再补仓 → 那 2 万就闲置 0 收益。用户质疑："有多余的钱能买就买啊，不然充钱干啥？"

本脚本回答：多余现金到底是「买(投入目标)」好还是「放着(闲置)」好。
方法：沿用生产完全一致的逐日模拟(mh.simulate_daily)，同一决策路径上只改一个
参数 invest_ratio = 投入持仓的比例（剩余比例即闲置现金，0 收益）：
  - invest_ratio = 1.00  = 满仓 · 买（现金全部投进目标）
  - invest_ratio = 0.75  = 假设 2万/8万≈25% 闲置不买
  - invest_ratio = 0.80  / 0.90 阈值扫描
比较：全期 + 样本外(2021+) 的年化/回撤/Calmar/累计，判断「买」是否值得。

配置与当前落地完全一致：MIN_HOLD=1、universe=BASE6+513500、黄金防御+GOLD、
防御动量门 10d、跟踪止损 6%、HOLD_N=1 + S/R止损 + 入场过滤。
结论若 满仓(买) 明显更优 → 建议实盘也应把多余现金及时补入目标（即使目标未变）。
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

SPLIT = pd.Timestamp("2021-01-01")
# 与生产一致的目标池：原6只 + 已落地的 513500
UNI = bue.BASE_CODES + ["513500"]
RATIOS = [1.00, 0.90, 0.80, 0.75, 0.70]


def row(tag, r):
    so = mh.slice_metrics(r, SPLIT)
    oa = so[0]*100 if so else float("nan")
    od = so[2]*100 if so else float("nan")
    print(f"{tag:16s}{r['cum']:>7.2f}{r['ann']*100:>8.2f}{r['maxdd']*100:>8.2f}"
          f"{r['calmar']:>9.2f}{r['sharpe']:>8.2f}{r['switches']:>6d}"
          f"{r['time_in_mkt']*100:>7.1f}{oa:>9.2f}{od:>9.2f}")
    return r


def main():
    price = bue.load_prices()
    print(f"真实ETF数据 {price.index[0].date()} ~ {price.index[-1].date()}，{len(price)} 交易日")
    print(f"配置 = MIN_HOLD=1 + 黄金防御+GOLD动量门10d + 跟踪6% + universe={len(UNI)}只"
          f"({','.join(UNI)})\n")
    print("「买(满仓) vs 不买(闲置现金)」逐日模拟对比: invest_ratio=投入目标比例，1-ratio=闲置现金")
    hdr = f"{'方案':16s}{'累计x':>7}{'年化%':>8}{'回撤%':>8}{'Calmar':>9}{'Sharpe':>8}{'换仓':>6}{'持仓%':>7}{'OOS年化%':>9}{'OOS回撤%':>9}"
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for ratio in RATIOS:
        r = mh.simulate_daily(price, 1, sr_params=bue.GOLD, def_trail_pct=None,
                              def_mom_days=10, universe=UNI, invest_ratio=ratio)
        tag = "100%满仓·买" if ratio == 1.00 else f"{ratio*100:.0f}%投入"
        rows.append((ratio, row(tag, r)))

    print("\n=== 关键对比（同一条决策路径，只差现金是否投入）===")
    base_ratio, base = rows[0]
    for ratio, r in rows[1:]:
        so = mh.slice_metrics(r, SPLIT)
        oa = so[0]*100 if so else float("nan")
        ob = mh.slice_metrics(base, SPLIT)
        oa_base = ob[0]*100 if ob else float("nan")
        print(f"  {ratio*100:.0f}%投入(闲置{1-ratio:.0%}) → 全期年化 {r['ann']*100:.2f}% "
              f"(满仓 {base['ann']*100:.2f}%) = {r['ann']*100-base['ann']*100:+.2f}pp；"
              f"OOS年化 {oa:.2f}% (满仓 {oa_base:.2f}%)")

    # 更清晰的量化：闲置多少 → 年化损失多少
    print()
    print("  闲置现金的代价（相对满仓·买 的年化损失，同一条决策路径）：")
    for ratio, r in rows[1:]:
        idle = 1 - ratio
        loss = base["ann"] - r["ann"]
        print(f"     闲置 {idle:.0%} → 年化差 {loss*100:+.2f}pp  "
              f"(每闲置10%资金约拖累 {loss*100/idle/10:.2f}pp/年)")
    print("\n结论：若 满仓(买) 全期与样本外均更优、回撤可控 → 冗余现金应补入目标；"
          "若 部分闲置 能显著压回撤而年化损失很小 → 可留小量现金垫底。")


if __name__ == "__main__":
    main()