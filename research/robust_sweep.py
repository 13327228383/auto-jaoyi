# -*- coding: utf-8 -*-
"""
robust_sweep.py —— 稳健最优判定（滚动预热起点 + 固定评估窗口 2021+）
================================================================================
动机：已证明"单点年化"对回测价格表起始年份极端敏感(路径依赖/前向预热)，
     2014起点 vs 2017起点 → OOS2021+ 98.8% vs 26.2%。单点不可信。

本脚本做法：对每个参数组合(SW × trail × hold)，
  - 用【滚动预热起点】 W = {2015,2016,...,2020} 各自冷启动全表模拟一次；
  - 但只评估它们共同覆盖的【固定窗口 2021+】(EVAL) 那一段的年化；
  - 得到一组 2021+ 年化（样本 = 预热起点数），报 中位数 / min / max / std。
预热越长的模拟="带着历史仓位进入2021"；预热越短的="接近冷启动进2021"。
中位数代表稳健水平，min~max 宽度代表对该路径依赖的敏感度。
可信的是"中位数相对排序"，不可信的是任何单个数值。

评估对象：BASE7 池(生产最优)、真实 T+1 约束。MW 固定 20(已证无影响)。
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

CODES = {'510300': 'sh510300', '510500': 'sh510500', '159915': 'sz159915',
         '518880': 'sh518880', '513100': 'sh513100', '511010': 'sh511010',
         '513500': 'sh513500'}
T1 = {'510300', '510500', '159915'}
EVAL = pd.Timestamp('2021-01-01')         # 评估窗口：2021+（不受起始年份差异影响）
WARMUPS = ['2015-01-01', '2016-01-01', '2017-01-01', '2018-01-01', '2019-01-01', '2020-01-01']
SWS = [20, 40, 60]
TRAILS = [0.03, 0.06]
HOLDS = [1, 5]
MW = 20


def load():
    cols = {}
    for c, sym in CODES.items():
        s = bue.fetch_sina(c, sym)
        s.index = pd.to_datetime(s.index)
        cols[c] = s
    price = pd.DataFrame(cols).sort_index()   # 不从最晚列 dropna，保留各列自有全历史
    price = price[price.index >= '2014-01-01']
    return price.dropna().sort_index()


def main():
    price = load()
    print(f"价格 {price.index[0].date()} ~ {price.index[-1].date()}，{len(price)}交易日，",
          f"标的中位数排序 评估窗口 2021+；预热起点 {WARMUPS}", flush=True)

    rows = []
    combos = list(itertools.product(SWS, TRAILS, HOLDS))
    print(f"组合数 {len(combos)} × 预热 {len(WARMUPS)} = {len(combos)*len(WARMUPS)} 次全表模拟", flush=True)
    i = 0
    for sw, trail, hold in combos:
        i += 1
        GOLD = dict(bue.GOLD)
        GOLD['SLOPE_WINDOW'] = sw
        GOLD['MOM_WINDOW'] = MW
        a21 = []
        for wu in WARMUPS:
            pr = price[price.index >= wu]        # 滚动预热：从该起点冷启动切片
            r = mh.simulate_daily(pr, hold, trail_pct=trail, sr_params=GOLD,
                                  def_trail_pct=None, def_mom_days=10,
                                  universe=list(CODES), t1_codes=T1)
            so = mh.slice_metrics(r, EVAL)
            a21.append(so[0] * 100 if so else float('nan'))
        arr = np.asarray([x for x in a21 if not np.isnan(x)], dtype=float)
        rows.append({
            'SW': sw, 'trail': int(trail * 100 if isinstance(trail, float) else trail * 100), 'hold': hold,
            'med': float(np.median(arr)) if len(arr) else float('nan'),
            'min': float(arr.min()) if len(arr) else float('nan'),
            'max': float(arr.max()) if len(arr) else float('nan'),
            'std': float(arr.std()) if len(arr) else float('nan'),
            'n': len(arr),
        })
        print(f"  [{i}/{len(combos)}] SW={sw} trail={trail*100:.0f}% hold={hold}d "
              f"→ 2021+年化 中位={rows[-1]['med']:.1f} (min={rows[-1]['min']:.1f} max={rows[-1]['max']:.1f})",
              flush=True)

    df = pd.DataFrame(rows).sort_values('med', ascending=False)
    print("\n======== 按 2021+年化中位数 排序（可信口径） ========")
    print(f"{'SW':>4}{'trail':>6}{'hold':>5}{'中位年化':>9}{'min':>8}{'max':>8}{'极差':>8}{'std':>7}{'样本':>5}")
    print("-" * 60)
    for _, r in df.iterrows():
        print(f"{r['SW']:>4}{r['trail']:>5}%{r['hold']:>5}{r['med']:>9.1f}{r['min']:>8.1f}"
              f"{r['max']:>8.1f}{r['max']-r['min']:>8.1f}{r['std']:>7.1f}{r['n']:>5}")

    b = df.iloc[0]
    print(f"\n>>> 稳健最优：SW={b['SW']} trail={b['trail']}% hold={b['hold']}d  "
          f"2021+年化中位 {b['med']:.1f}% (min {b['min']:.1f} ~ max {b['max']:.1f}%)")


if __name__ == '__main__':
    main()