# -*- coding: utf-8 -*-
"""
backtest_reduce_t1.py —— 减少/移除 T+1 标的对轮动收益的真实影响
================================================================================
背景：backtest_t1_constraint 证明 3 只 T+1 标的(300/500/创业板)因"当天卖不了"
     拖累整体（全期年化 21.16%->14.54%，回撤 -16.81%->-24.53%）。
本脚本测"减少 T+1 标的参与"是否提升收益/回撤。所有方案统一用 t1_codes 对残余
T+1 正确建模（当日触发回撤->次日离场），T+0 不受影响。

方案：
  A. CURRENT   : 7只全部（4T+0 + 3T+1，现状）
  B. NO_CMBT   : 移除 159915 创业板（T+1 中波动大、贡献弱最常被怀疑的）
  C. T0_ONLY   : 只剩 4 只 T+0（黄金/国债/纳指/标普）
  D. NO_300    : 移除 510300（T+1 大盘，防御同源，去掉减少冗余）

数据/配置同 backtest_current（MIN_HOLD=1，黄金防御，slope_r2，SR+回撤3%）。
"""
import os, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)

import backtest_minhold as mh
import backtest_current as bc

GOLD = {
    "DEFENSIVE": "518880", "DEFENSE_MODE": "defensive", "DEF_MOM_DAYS": 10,
    "GLOBAL_US": "513100", "GLOBAL_GOLD": "518880", "BROAD": "510300",
}
T0 = ["518880", "511010", "513100", "513500"]
ALL = ["510300", "510500", "159915", "518880", "513100", "511010", "513500"]
T1 = {"510300", "510500", "159915"}
MIN_HOLD = 1
SPLIT = pd.Timestamp("2021-01-01")

PLANS = [
    ("A. 现状7只", ALL, None),
    ("B. 移除创业板", [c for c in ALL if c != "159915"], {"510300", "510500"}),
    ("C. 仅T+0四只", T0, None),
    ("D. 移除沪深300", [c for c in ALL if c != "510300"], {"510500", "159915"}),
]


def main():
    price = bc.load_prices7()
    print(f"数据 {price.index[0].date()}~{price.index[-1].date()} {len(price)}交易日\n"
          f"T+1 约束已建模（当日触发回撤->次日离场）。含万2.38双边摩擦。\n")
    hdr = f"{'方案':20s}{'标的数':>5s}{'年化':>9s}{'回撤':>8s}{'Calmar':>8}{'换手':>6}{'OOS年化':>9s}{'OOS回撤':>8s}{'OOS Cal':>8s}"
    print(hdr); print("-" * len(hdr))
    for name, uni, t1 in PLANS:
        if t1 is None:
            t1 = T1 if set(T1) <= set(uni) else (set(uni) & T1)
        r = mh.simulate_daily(price, MIN_HOLD, sr_params=GOLD, def_trail_pct=None,
                              def_mom_days=10, universe=uni, t1_codes=t1)
        oos = mh.slice_metrics(r, SPLIT)
        if oos:
            (ann, cum, dd) = oos
            cal = ann / abs(dd) if dd < 0 else float("nan")
            oos_s = f"{ann*100:>8.2f}{dd*100:>9.2f}{cal:>9.2f}"
        else:
            oos_s = "  N/A"
        print(f"{name:20s}{len(uni):>5d}{r['ann']*100:>9.2f}{r['maxdd']*100:>8.2f}"
              f"{r['calmar']:>8.2f}{r['switches']:>6d}{oos_s}")


if __name__ == "__main__":
    main()