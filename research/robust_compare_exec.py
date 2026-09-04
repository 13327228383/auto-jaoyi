# -*- coding: utf-8 -*-
"""
robust_compare_exec.py —— SW20 vs SW40 · 三种成交口径并排稳健对比
================================================================================
目的：厘清"成交模型"对结论的影响，避免把"更悲观假设"误当成当前实盘表现。
口径（对 SW20 / SW40，滚动预热6起点 + 评估2021+，年化用中位数）：
  A close0      当日收盘成交，仅佣金        → 旧口径（相对最优的"理想"基准）
  B close_slip  当日收盘成交 + 滑点0.1%（双边）→ 对齐当前实盘（程序当日尾盘按现价买）
  C open_slip   次日开盘成交 + 滑点0.1%（双边）→ 悲观下界（假设真拖到次日开盘才买）
实盘程序(auto_run)是当日尾盘买，所以 B 才是当前真正可执行的对比；A 是理想、C 是下界。
"""
import os, sys, itertools
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)

import backtest_universe_ext as bue
import backtest_minhold as mh

CODES = sorted({'510300', '510500', '159915', '518880', '513100', '511010', '513500'})
T1 = {'510300', '510500', '159915'}
EVAL = pd.Timestamp('2021-01-01')
WARMUPS = ['2015-01-01', '2016-01-01', '2017-01-01', '2018-01-01', '2019-01-01', '2020-01-01']
SWS = [20, 40]
HOLDS = [1, 5]
SLIP = 0.001


def run(close, open_, mode, sw, hold, wu):
    pr = close[close.index >= wu]
    po = open_[open_.index >= wu]
    G = dict(bue.GOLD); G['SLOPE_WINDOW'] = sw; G['MOM_WINDOW'] = 20
    if mode == 'close0':
        r = mh.simulate_daily(pr, hold, sr_params=G, def_mom_days=10, universe=CODES, t1_codes=T1)
    elif mode == 'close_slip':
        r = mh.simulate_daily(pr, hold, sr_params=G, def_mom_days=10, universe=CODES, t1_codes=T1,
                              slip=SLIP)
    else:  # open_slip
        r = mh.simulate_daily(pr, hold, sr_params=G, def_mom_days=10, universe=CODES, t1_codes=T1,
                              price_open=po, slip=SLIP)
    so = mh.slice_metrics(r, EVAL)
    return (so[0] * 100 if so else float('nan')), r['switches']


def agg(close, open_, mode, sw, hold):
    anns, swi = [], []
    for wu in WARMUPS:
        a, s = run(close, open_, mode, sw, hold, wu)
        anns.append(a); swi.append(s)
    arr = np.asarray([x for x in anns if not np.isnan(x)], dtype=float)
    return (np.median(arr), np.median(swi), arr.min(), arr.max())


def main():
    close, open_ = bue.load_prices_open(CODES)
    close = close[close.index >= '2014-01-01']; open_ = open_.reindex(close.index)

    MODES = [('close0', '当日收盘·仅佣金'), ('close_slip', '当日收盘+滑点0.1%(实盘)'), ('open_slip', '次日开盘+滑点0.1%(下界)')]
    print(f"滚动预热+评估2021+ 年化中位数  |  滑点{SLIP*100:.1f}%/单边(双边)"
          f"  |  标的{sorted(CODES)}\n")

    hdr = f"{'口径':>22s}{'hold':>5s}{'SW20年化':>10s}{'SW40年化':>10s}{'差(40-20)':>12s}{'SW20换仓':>9s}{'SW40换仓':>9s}"
    print(hdr); print('-' * len(hdr))
    for mode, label in MODES:
        for hold in HOLDS:
            m20, s20, lo20, hi20 = agg(close, open_, mode, 20, hold)
            m40, s40, lo40, hi40 = agg(close, open_, mode, 40, hold)
            d = m40 - m20
            tag = '  <-实盘口径' if mode == 'close_slip' and hold == 5 else ''
            print(f"{label:>22s}{hold:>5d}{m20:>10.2f}{m40:>10.2f}{d:>+12.2f}"
                  f"{s20:>9.0f}{s40:>9.0f}{tag}")
        print()

    print("解读：")
    print("  数据/市场从未变，变的是'成交模型(A/B/C)'这一输入 → 结论随口径而变。")
    print("  实盘程序是当日尾盘按现价买 → B 才是当前真实可执行对比；A 理想、C 悲观下界。")


if __name__ == '__main__':
    main()