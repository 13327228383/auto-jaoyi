# -*- coding: utf-8 -*-
"""
backtest_t0_final.py —— 「精简纯T+0池」落地前定板回测
================================================================================
用同一套真实ETF数据 + 与生产一致的策略配置，对比三套池，决定落地池子：

  BASE  原7只   (510300/510500/159915/518880/513100/511010/513500)  基准
  T0_4  纯T+0四只(518880/513100/511010/513500)                      已验证T+0
  T0_5  纯T+0五只(518880/513100/511010/513500/159920恒生)            用户候选"5只"

背景冲突：代码注释曾写"逐只隔离 159920 是拖累"。本脚本用真实数据定位
"到底该落地4只还是5只"，避免把被证拖累的恒生硬塞进池。

口径：GET 真实ETF收盘(新浪缓存 cache/backtest/etf_<code>.csv)，
      逐日模拟 = backtest_minhold.simulate_daily，GOLD参数与生产一致
      (DEFENSIVE=518880, DEFENSIVE_MODE=defensive, DEF_MOM_DAYS=10, MIN_HOLD=1)。
"""
import os, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import backtest_universe_ext as bue   # fetch_sina / ALL_CODES
import backtest_minhold as mh

SPLIT = pd.Timestamp("2021-01-01")
MIN_HOLD = 1
GOLD = bue.GOLD          # DEFENSIVE=518880 / DEFENSE_MODE=defensive / DEF_MOM_DAYS=10
DEF_TRAIL = None
DEF_MOM = 10

BASE = ["510300", "510500", "159915", "518880", "513100", "511010", "513500"]
T0_4 = ["518880", "513100", "511010", "513500"]
T0_5 = ["518880", "513100", "511010", "513500", "159920"]


def load(subset):
    cols = {}
    for c in subset:
        s = bue.fetch_sina(c, bue.ALL_CODES[c])
        cols[c] = s
    p = pd.DataFrame(cols).dropna()
    p.index = pd.to_datetime(p.index)
    return p.sort_index()


def main():
    # 三种池共拉超市集，保证同一模拟器、同一索引行数
    allc = sorted(set(BASE) | set(T0_4) | set(T0_5))
    print("拉取共用超市集：", allc)
    price = load(list(allc))
    print(f"数据 {price.index[0].date()} ~ {price.index[-1].date()}，{len(price)} 交易日\n")

    hdr = f"{'方案':8s}{'池':>38s}{'累计x':>8}{'年化%':>8}{'回撤%':>8}{'Calmar':>7}{'换仓':>5}{'持仓%':>7}"
    print(hdr)
    print("-" * len(hdr))
    rows = {}
    for name, uni in [("BASE", BASE), ("T0_4", T0_4), ("T0_5", T0_5)]:
        r = mh.simulate_daily(price, MIN_HOLD, sr_params=GOLD, def_trail_pct=DEF_TRAIL,
                              def_mom_days=DEF_MOM, universe=list(uni))
        rows[name] = r
        print(f"{name:8s}{','.join(uni):>38s}{r['cum']:>8.2f}{r['ann']*100:>8.2f}{r['maxdd']*100:>8.2f}"
              f"{r['calmar']:>7.2f}{r['switches']:>5d}{r['time_in_mkt']*100:>7.1f}")

    for name in rows:
        so = mh.slice_metrics(rows[name], SPLIT)
        if so:
            print(f"\n样本外(2021+) {name}: 年化{so[0]*100:6.2f}%  累计{so[1]:5.2f}x  回撤{so[2]*100:6.2f}%")
        else:
            print(f"\n样本外(2021+) {name}: 数据不足")

    a, b4, b5 = rows["BASE"], rows["T0_4"], rows["T0_5"]
    print("\n=== 判定 ===")
    print(f"T0_4 vs BASE : 年化{b4['ann']*100- a['ann']*100:+5.2f}pp  回撤{b4['maxdd']*100- a['maxdd']*100:+5.2f}pp")
    print(f"T0_5 vs BASE : 年化{b5['ann']*100- a['ann']*100:+5.2f}pp  回撤{b5['maxdd']*100- a['maxdd']*100:+5.2f}pp")
    print(f"T0_5 vs T0_4 : 年化{b5['ann']*100- b4['ann']*100:+5.2f}pp  回撤{b5['maxdd']*100- b4['maxdd']*100:+5.2f}pp  (恒生159920净贡献，正=该加/负=该舍)")


if __name__ == "__main__":
    main()