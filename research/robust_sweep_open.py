# -*- coding: utf-8 -*-
"""
robust_sweep_open.py —— 稳健对比 · 次日开盘价成交 + 滑点 口径
================================================================================
背景：原来"当日收盘价成交"存在前视偏差——决策日(收盘)生成 target 当天即以收盘价
      建仓，等于白拿新强势的"确认日跳涨"。真实买入只能 T+1 且更贴近"次日开盘价"。
      本脚本把执行口径改为【决策日收盘 → 次一交易日开盘价成交】，并在每次买卖
      外加滑点(slip)磨损，重做一次 SW20 vs SW40 的稳健对比，回答：
        换用真实执行口径后，SW20 还是最优吗？操作磨损(次数)差在哪？

稳健方法(与 robust_sweep 一致)：滚动预热起点 {2015..2020} 各自冷启动全表模拟，
  只评估共同覆盖的固定窗口 2021+ (EVAL) 的年化；报 中位/min/max/std。
  中位数=稳健水平，min~max 宽度=对路径依赖/起始点的敏感度。

评估对象：BASE7 池(生产最优)、真实 T+1、黄金防御动量门10d、跟踪止损6%、
  最小持仓 MIN_HOLD ∈ {1,5}，滑点 slip=0.001(0.1%)。固定 MOM=20。
"""
import os, sys, itertools
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import backtest_universe_ext as bue
import backtest_minhold as mh

CODES = {'510300', '510500', '159915', '518880', '513100', '511010', '513500'}
T1 = {'510300', '510500', '159915'}
EVAL = pd.Timestamp('2021-01-01')
WARMUPS = ['2015-01-01', '2016-01-01', '2017-01-01', '2018-01-01', '2019-01-01', '2020-01-01']
SWS = [20, 40]
HOLDS = [1, 5]
TRAIL = 0.06
MOM = 20
SLIP = 0.001          # 单侧滑点 0.1%（模拟真实交易磨损）


def run_one(close, open_, sw, hold, wu):
    """冷启动一个滚动预热起点，完整模拟，返回 EVAL 段年化 + 全表操作次数。"""
    pr = close[close.index >= wu]
    po = open_[open_.index >= wu]
    GOLD = dict(bue.GOLD)
    GOLD['SLOPE_WINDOW'] = sw
    GOLD['MOM_WINDOW'] = MOM
    r = mh.simulate_daily(pr, hold, trail_pct=TRAIL, sr_params=GOLD,
                          def_trail_pct=None, def_mom_days=10,
                          universe=sorted(CODES), t1_codes=T1,
                          price_open=po, slip=SLIP)
    so = mh.slice_metrics(r, EVAL)
    ann = so[0] * 100 if so else float('nan')
    return ann, r['switches'], r['stop_exits'], r['time_in_mkt']


def main():
    close, open_ = bue.load_prices_open(sorted(CODES))
    close = close[close.index >= '2014-01-01']
    open_ = open_.reindex(close.index)
    print(f"真实ETF 共同区间 {close.index[0].date()} ~ {close.index[-1].date()}，"
          f"{len(close)} 交易日；标的 {sorted(CODES)}", flush=True)
    print(f"执行口径=次日开盘价成交 + 滑点 {SLIP*100:.2f}%/次；评估窗口 2021+；"
          f"预热起点 {WARMUPS}\n", flush=True)

    rows = []
    for sw, hold in itertools.product(SWS, HOLDS):
        anns, swi, sto = [], [], []
        for wu in WARMUPS:
            a, s, st, _ = run_one(close, open_, sw, hold, wu)
            anns.append(a)
            swi.append(s)
            sto.append(st)
        arr = np.asarray([x for x in anns if not np.isnan(x)], dtype=float)
        rows.append({
            'SW': sw, 'hold': hold,
            'med': float(np.median(arr)) if len(arr) else float('nan'),
            'min': float(arr.min()) if len(arr) else float('nan'),
            'max': float(arr.max()) if len(arr) else float('nan'),
            'std': float(arr.std()) if len(arr) else float('nan'),
            'sw_int': float(np.median(swi)),
            'sw_min': min(swi), 'sw_max': max(swi),
            'stop_med': float(np.median(sto)),
        })
        print(f"  SW={sw:>2d} hold={hold}d → 2021+年化中位 {rows[-1]['med']:6.2f}% "
              f"(min {rows[-1]['min']:6.2f} ~ max {rows[-1]['max']:6.2f}) | "
              f"换仓中位 {rows[-1]['sw_int']:.0f} 次 (min {rows[-1]['sw_min']}~{rows[-1]['sw_max']})",
              flush=True)

    df = pd.DataFrame(rows).sort_values(['hold', 'med'], ascending=[True, False])
    print("\n======== 次日开盘+滑点 稳健对比（2021+ 年化中位数排序） ========")
    print(f"{'SW':>4}{'hold':>5}{'中位年化':>9}{'min':>8}{'max':>8}{'极差':>8}{'std':>7}"
          f"{'换仓中位':>8}{'换仓区间':>14}{'止损中位':>8}")
    print('-' * 76)
    for _, r in df.iterrows():
        print(f"{r['SW']:>4}{r['hold']:>5}{r['med']:>9.2f}{r['min']:>8.2f}{r['max']:>8.2f}"
              f"{r['max']-r['min']:>8.2f}{r['std']:>7.2f}{r['sw_int']:>8.0f}"
              f"{int(r['sw_min']):>5d}~{int(r['sw_max']):>8d}{r['stop_med']:>8.0f}")

    # SW20 vs SW40 同 hold 速览（聚焦答复）
    print("\n=== SW20 vs SW40 同口径速览（取 hold 中最常用的对比） ===")
    for hold in HOLDS:
        a = df[(df['hold'] == hold) & (df['SW'] == 20)]
        b = df[(df['hold'] == hold) & (df['SW'] == 40)]
        if len(a) and len(b):
            a, b = a.iloc[0], b.iloc[0]
            print(f"  MIN_HOLD={hold}d: SW20 年化{a['med']:.2f}% 换仓{a['sw_int']:.0f}  "
                  f"vs  SW40 年化{b['med']:.2f}% 换仓{b['sw_int']:.0f}")
            delta = a['med'] - b['med']
            dsw = a['sw_int'] - b['sw_int']
            print(f"         → 年化差 = {delta:+.2f}pp，换仓差 = {dsw:+.0f} 次"
                  f"{'（SW40 磨损更低）' if dsw > 0 else '（SW20 磨损更低）'}")

    best = df.iloc[0]
    print(f"\n>>> 次日开盘+滑点 口径下的稳健最优：SW={best['SW']} hold={best['hold']}d，"
          f"2021+年化中位 {best['med']:.2f}% (min {best['min']:.2f} ~ max {best['max']:.2f}%)，"
          f"换仓中位 {best['sw_int']:.0f} 次")


if __name__ == '__main__':
    main()