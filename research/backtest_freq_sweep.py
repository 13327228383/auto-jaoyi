# -*- coding: utf-8 -*-
"""
backtest_freq_sweep.py —— 调仓频率 × 持仓数 扫描：找「赚得多、回撤小、操作少」组合
====================================================================================
复用 backtest_alloc_v4 的最优保护（S/R动态止损 + 入场过滤 + 跟踪止损6%）+ mom2 动量平方加权。

三个维度同时看：
  赚得多   = 年化CAGR / 累计
  回撤小   = 最大回撤 / Calmar / Sharpe
  操作少   = 换仓次数(switches) + 止损次数(stop_exits) = 总操作次数（调仓一次≈卖+买各1笔）

缩「操作频率」= 拉长调仓周期(FREQ)。动量策略通常越长周期换手越低、越抗抖动，
但也不能过长以免错失趋势切换。本扫描找平衡点。
"""
import os, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import backtest_alloc_v4 as alloc

SPLIT = pd.Timestamp("2021-01-01")
FREQS = ["1W", "2W", "3W", "4W", "6W", "8W"]
HOLDNS = [1, 2]
ALLOC = "mom2"


def dec_dates_for(price, freq):
    return [d for d in price.resample(freq).last().index if len(price.loc[:d]) >= 200]


def main():
    price = alloc.bse.load_prices()
    print(f"数据 {price.index[0].date()} ~ {price.index[-1].date()}，{len(price)} 交易日\n")
    print(f"保护=S/R动态止损+入场过滤+跟踪止损6%  权重={ALLOC}\n")
    hdr = (f"{'频率':6s}{'N':>3s}{'累计x':>7}{'年化':>7}{'回撤':>7}{'Calmar':>7}"
           f"{'Sharpe':>7}{'换仓':>5}{'止损':>5}{'总操作':>6}{'持仓%':>6}")
    print(hdr)
    print("-" * len(hdr))

    best_list = []
    for freq in FREQS:
        dec_dates = dec_dates_for(price, freq)
        targets = alloc.decide_targets(price, dec_dates, hold_n=1)
        r1 = alloc.simulate(price, dec_dates, targets, ALLOC)
        targets2 = alloc.decide_targets(price, dec_dates, hold_n=2)
        r2 = alloc.simulate(price, dec_dates, targets2, ALLOC)
        for n, r in ((1, r1), (2, r2)):
            ops = r["switches"] + r["stop_exits"]
            print(f"{freq:6s}{n:>3d}{r['cum']:>8.2f}{r['ann']*100:>7.2f}"
                  f"{r['maxdd']*100:>7.2f}{r['calmar']:>7.2f}{r['sharpe']:>7.2f}"
                  f"{r['switches']:>5d}{r['stop_exits']:>5d}{ops:>6d}{r['time_in_mkt']*100:>6.1f}")
            best_list.append((r["ann"] * 100
                              - abs(r["maxdd"]) * 100   # 扣回撤惩罚
                              - ops * 0.05,             # 扣操作次数惩罚
                              f"{freq}-N{n}", r))

    print("\n综合得分（年化 − |回撤| − 0.05×操作次数）排序：")
    best_list.sort(key=lambda x: x[0], reverse=True)
    for score, name, r in best_list[:4]:
        ops = r["switches"] + r["stop_exits"]
        print(f"   {name:8s} score={score:7.2f} 年化{r['ann']*100:6.2f}% 回撤{r['maxdd']*100:6.2f}% "
              f"Calmar{r['calmar']:5.2f} Sharpe{r['sharpe']:5.2f} 总操作{ops}")

    print("\n样本外(2021+)对比：")
    for freq in FREQS:
        dec_dates = dec_dates_for(price, freq)
        for n in HOLDNS:
            targets = alloc.decide_targets(price, dec_dates, hold_n=n)
            r = alloc.simulate(price, dec_dates, targets, ALLOC)
            so = alloc.slice_metrics(r, SPLIT)
            if not so:
                continue
            ops = r["switches"] + r["stop_exits"]
            print(f"   {freq}-N{n:1d}: 年化{so[0]*100:6.2f}% 回撤{so[1]*100:6.2f}% "
                  f"Sharpe{so[2]:5.2f} Calmar{so[3]:5.2f} 总操作{ops}")


if __name__ == "__main__":
    main()