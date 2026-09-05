# -*- coding: utf-8 -*-
"""局部精扫 SW 峰值 + 邻域滑点敏感性（实盘口径，当日收盘+双边滑点）。"""
import os, sys
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
HOLD = 1; TRAIL = 0.03


def run(close, sw, slip, wu):
    pr = close[close.index >= wu]
    G = dict(bue.GOLD); G['SLOPE_WINDOW'] = sw; G['MOM_WINDOW'] = 20
    G['DEF_MOM_ENTER'] = 0.005; G['DEF_MOM_EXIT'] = -0.008  # 对齐实盘防御迟滞带
    r = mh.simulate_daily(pr, HOLD, trail_pct=TRAIL, sr_params=G, def_trail_pct=TRAIL,
                          def_mom_days=10, universe=CODES, t1_codes=T1, slip=slip)
    full = r['cum'] ** (252.0 / len(r['equity'])) - 1 if r['cum'] > 0 else -1.0
    so = mh.slice_metrics(r, EVAL)
    return full * 100, (so[0] * 100 if so else float('nan')), r['switches'] if r['switches'] else 0


def oos_med(close, sw, slip):
    return np.median([run(close, sw, slip, wu)[1] for wu in WARMUPS])


def main():
    close, _ = bue.load_prices_open(CODES)
    close = close[close.index >= '2014-01-01']

    print("== 局部精扫 SW（slip=0.1%）2021+ 年化中位 ==")
    sws = [24, 26, 28, 30, 32, 34, 36]
    med = {sw: oos_med(close, sw, 0.001) for sw in sws}
    for sw in sws:
        print(f"  SW{sw:>2d}: {med[sw]:6.1f}%", flush=True)
    best = max(med, key=med.get)
    print(f"  → 精扫峰值 SW={best} ({med[best]:.1f}%)")

    print("\n== 邻域滑点敏感性（SW28/30/32 · slip 0.05%/0.1%/0.2%）2021+ 中位年化 ==")
    print(f"{'slip':>8}{'SW28':>9}{'SW30':>9}{'SW32':>9}")
    for sp in [0.0005, 0.0010, 0.0020]:
        print(f"{sp*100:>7.2f}%"
              f"{oos_med(close,28,sp):>9.1f}{oos_med(close,30,sp):>9.1f}{oos_med(close,32,sp):>9.1f}",
              flush=True)


if __name__ == '__main__':
    main()