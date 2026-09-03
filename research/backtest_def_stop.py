# -*- coding: utf-8 -*-
"""backtest_def_stop.py —— 「黄金防御 + 止损收紧」回测：压缩 -20.76% 回撤
================================================================================
背景：backtest_defense_assets.py 证明黄金防御(518880)年化从国债18.27%→27.27%，
但回撤放大到 -20.76%。原因：防御资产黄金在 simulate_daily 里被豁免止损(stop=None)，
熊市里一路被套扛到最深处。

本脚本固定黄金防御 + MIN_HOLD=1，对比「止损收紧程度」，找能显著压回撤、
且不牺牲太多收益的最优档：
  V_g0  基线     ：黄金不设止损 + 普通标的 S/R(4~15%)+跟踪6%(原 def_gold)
  V_g6  黄金6%尾 : 黄金启用跟踪止损6%
  V_g4  黄金4%尾 : 黄金启用跟踪止损4%
  V_g4+ 黄金4%尾 + 普通标的收紧(S/R封顶10%、回退-6%)   ← 整体更激进
  V_g3  黄金3%尾 : 看极限收紧是否过拟合/换手爆炸

口径与 backtest_defense_assets 完全一致(逐日+MIN_HOLD=1+S/R+跟踪+入场过滤+费0.1%)，
仅改止损参数。输出 全期 + 样本外(2021+) 对比表。
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import backtest_minhold as mh
import backtest_sr_enhance as bse

SPLIT = pd.Timestamp("2021-01-01")
GOLD = {"DEFENSIVE": "518880"}          # 固定黄金防御


def run(trail, def_trail, caps=(0.04, 0.15), fallback=0.08, def_mom_days=None):
    """在给定止损参数下跑黄金防御回测。caps=(LO,HI)，fallback=无支撑/支撑太近回退幅度。"""
    saved = (bse.STOP_CAP_LO, bse.STOP_CAP_HI, mh.TRAIL_PCT)
    bse.STOP_CAP_LO, bse.STOP_CAP_HI = caps
    mh.TRAIL_PCT = trail
    try:
        return mh.simulate_daily(bse.load_prices(), MIN_HOLD_D, trail_pct=trail,
                                 sr_params=GOLD, def_trail_pct=def_trail,
                                 def_mom_days=def_mom_days)
    finally:
        bse.STOP_CAP_LO, bse.STOP_CAP_HI, mh.TRAIL_PCT = saved


def fmt(r):
    return (f"{r['cum']:>8.2f}{r['ann']*100:>8.2f}{r['maxdd']*100:>8.2f}"
            f"{r['calmar']:>8.2f}{r['sharpe']:>8.2f}{r['switches']:>6d}"
            f"{r['stop_exits']:>6d}{r['time_in_mkt']*100:>8.1f}")


MIN_HOLD_D = 1
VARIANTS = [
    ("gold 基线(无防损)",      dict(trail=0.06, def_trail=None)),
    ("gold+防御6%尾",          dict(trail=0.06, def_trail=0.06)),
    ("gold+防御4%尾",          dict(trail=0.06, def_trail=0.04)),
    # 动量门：防御黄金自身动量转负→空仓等待(不买回)——真正压缩防御回落
    ("gold+黄金动量门20d",     dict(trail=0.06, def_trail=None, def_mom_days=20)),
    ("gold+黄金动量门10d",     dict(trail=0.06, def_trail=None, def_mom_days=10)),
    ("gold+动量门20d+4%尾",    dict(trail=0.06, def_trail=0.04, def_mom_days=20)),
]


def main():
    price = bse.load_prices()
    print(f"数据 {price.index[0].date()} ~ {price.index[-1].date()}，{len(price)} 交易日")
    print(f"黄金防御 + MIN_HOLD={MIN_HOLD_D}d + S/R止损+跟踪6%+入场过滤；仅「止损参数」不同\n")

    hdr = f"{'方案':20s}{'累计x':>8}{'年化':>8}{'回撤':>8}{'Calmar':>8}{'Sharpe':>8}{'换仓':>6}{'止损':>6}{'持仓%':>8}"
    print(hdr)
    print("-" * len(hdr))
    rows = {}
    for name, kw in VARIANTS:
        r = run(**kw)
        rows[name] = r
        print(f"{name:20s}{fmt(r)}")

    print("\n样本外(2021+)校验：")
    for name, r in rows.items():
        so = mh.slice_metrics(r, SPLIT)
        if so:
            print(f"   {name:20s} 年化{so[0]*100:6.2f}%  累计{so[1]:5.2f}x  回撤{so[2]*100:6.2f}%")
        else:
            print(f"   {name:20s} 样本外数据不足")

    # 结论
    base = rows["gold 基线(无防损)"]
    best = min(rows.values(), key=lambda r: (r["maxdd"], -r["ann"]))
    bname = max(rows, key=lambda n: rows[n]["calmar"])
    print("\n=== 结论 ===")
    print(f"  基线: 年化{base['ann']*100:.2f}%  回撤{base['maxdd']*100:.2f}%  Calmar{base['calmar']:.2f}")
    br = rows[bname]
    print(f"  最优Calmar[{bname}]: 年化{br['ann']*100:.2f}%  回撤{br['maxdd']*100:.2f}%  Calmar{br['calmar']:.2f}")
    min_dd = max(r['maxdd'] for r in rows.values())   # 数越小越接近0(回撤最浅)
    print(f"  回撤最浅档: 回撤{min_dd*100:.2f}%")


if __name__ == "__main__":
    main()