# -*- coding: utf-8 -*-
"""
grid_exec_close.py —— 实盘口径(当天买+卖一价滑点)下，SW 参数稳健网格扫描
================================================================================
对齐当前生产对齐复用：MIN_HOLD=1、tracking trail=3%(普通+黄金统一)、DEF=518880 黄金、
DEF_MOM_DAYS=10。只在"实盘口径"下作【当日收盘成交 + 双边滑点 slip】扫描。

稳健方法：滚动预热起点 {2015..2020} 冷启动全表 → 全期年化 + 固定 2021+ 窗口年化，
  均取 6 起点中位数（含磨损，磨损已由引擎成本扣入）。主要判据 = 2021+ 中位年化。
"""
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
HOLD = 1          # 对齐生产 MIN_HOLD_DAYS
TRAIL = 0.03      # 对齐生产 SR_TRAIL_PCT / DEF_PEAK_STOP
SLIP = 0.001      # 基准滑点（单侧 0.1%，卖一价买入近似）


def run(close, sw, wu, slip=SLIP):
    pr = close[close.index >= wu]
    G = dict(bue.GOLD); G['SLOPE_WINDOW'] = sw; G['MOM_WINDOW'] = 20
    G['DEF_MOM_ENTER'] = 0.005; G['DEF_MOM_EXIT'] = -0.008  # 对齐实盘防御迟滞带
    r = mh.simulate_daily(pr, HOLD, trail_pct=TRAIL, sr_params=G, def_trail_pct=TRAIL,
                          def_mom_days=10, universe=CODES, t1_codes=T1, slip=slip)
    full = r['cum'] ** (252.0 / len(r['equity'])) - 1 if r['cum'] > 0 else -1.0
    so = mh.slice_metrics(r, EVAL)
    oos = so[0] * 100 if so else float('nan')
    return full * 100, oos, r['switches'], r['maxdd'] * 100


def main():
    close, _ = bue.load_prices_open(CODES)
    close = close[close.index >= '2014-01-01']
    print(f"实盘口径: 当日收盘成交 + 双边滑点{SLIP*100:.1f}% + 佣金 | HOLD={HOLD}d trail={TRAIL:.0%} "
          f"DEF摩=10 | 滚动预热6起点 | 平滑器 SW 扫描\n", flush=True)
    hdr = f"{'SW':>4}{'全期中位':>9}{'2021+中位':>10}{'min':>8}{'max':>8}{'Std':>7}{'换仓:中位':>9}{'回撤中位':>9}"
    print(hdr); print('-' * len(hdr))
    rows = []
    for sw in [20, 30, 40, 60]:
        fulls, ooss, swi, dd = [], [], [], []
        for wu in WARMUPS:
            f, o, s, d = run(close, sw, wu)
            fulls.append(f); ooss.append(o); swi.append(s); dd.append(d)
        a = np.asarray(ooss)
        row = {'SW': sw, 'full_med': np.median(fulls), 'oos_med': np.median(a),
               'oos_min': a.min(), 'oos_max': a.max(), 'oos_std': a.std(),
               'sw_med': np.median(swi), 'dd_med': np.median(dd)}
        rows.append(row)
        print(f"{sw:>4}{row['full_med']:>9.1f}{row['oos_med']:>10.1f}{row['oos_min']:>8.1f}"
              f"{row['oos_max']:>8.1f}{row['oos_std']:>7.1f}{row['sw_med']:>9.0f}{row['dd_med']:>9.1f}",
              flush=True)
    df = pd.DataFrame(rows).sort_values('oos_med', ascending=False)
    best = df.iloc[0]
    print(f"\n>>> 实盘口径(含磨损)下最优：SW={best['SW']}  2021+中位年化{best['oos_med']:.1f}%"
          f"(min{best['oos_min']:.1f}~max{best['oos_max']:.1f}，std{best['oos_std']:.1f})  "
          f"全期中位{best['full_med']:.1f}%，换仓{best['sw_med']:.0f}次，回撤中位{best['dd_med']:.1f}%")


if __name__ == '__main__':
    main()