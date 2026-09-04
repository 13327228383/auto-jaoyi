# -*- coding: utf-8 -*-
"""
backtest_t0_expanded.py —— 用更大 T+0 池子重新对比（回应"才4只赚得动吗"）
================================================================================
背景：用户质疑"诺大ETF市场只有4只T+0吗？才4个赚得动？" —— 正确。T+0 ETF有几百只
     (跨境/债券/货币/商品)。之前只测了手头4只，样本偏见。

本脚本拉取重要跨大类 T+0 ETF，拼出更大候选池，统一 T+1 建模对比：

  A. PRIOR7  : 原7只(4T0+3T1)  —— 现状
  B. T0_ONLY : 原4只T+0
  C. T0_EXP  : 扩展到 ~11只 T+0(纳指/标普/日经/德国/恒生/中概/国债/政金/转债/黄金/有色)
  D. T0_EXP_1: 扩展T+0 + 保留 510300(跨度大的A股T+1兜底)

数据: get_etf_k_history 真实日K。规则/成本同 backtest_t1_constraint。
"""
import os, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)

from data_center import get_etf_k_history
import backtest_minhold as mh

GOLD = {
    "DEFENSIVE": "518880", "DEFENSE_MODE": "defensive", "DEF_MOM_DAYS": 10,
    "GLOBAL_US": "513100", "GLOBAL_GOLD": "518880", "BROAD": "510300",
}
T1 = {"510300", "510500", "159915"}
MIN_HOLD = 1
SPLIT = pd.Timestamp("2021-01-01")

# 原7只
PRIOR7 = ["510300", "510500", "159915", "518880", "513100", "511010", "513500"]
# T+0 扩展（跨大类、上市早）
T0_EXP = ["513100", "513500", "513000", "513030", "510900", "513050",
          "511010", "511520", "511380", "518880", "512400"]
ALL_CODES = list(dict.fromkeys(PRIOR7 + T0_EXP))  # 去重保序


def load():
    cols = {}
    for c in ALL_CODES:
        try:
            df = get_etf_k_history(c, days=3200)
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date").set_index("date")
            cols[c] = df["close"].astype(float)
        except Exception as e:
            print(f"!! {c} 拉取失败 {e}")
    p = pd.DataFrame(cols).dropna()
    return p.sort_index()


PLANS = [
    ("A. 原7只(含3只T+1)", PRIOR7, T1),
    ("B. 仅原4只T+0", ["518880", "511010", "513100", "513500"], None),
    ("C. T+0扩展11只", T0_EXP, None),
    ("D. T+0扩展+沪深300", T0_EXP + ["510300"], {"510300"}),
]


def main():
    price = load()
    print(f"数据 {price.index[0].date()} ~ {price.index[-1].date()}  {len(price)}交易日, "
          f"标的 {len(price.columns)} 只:")
    print("  ", list(price.columns), "\n")
    hdr = f"{'方案':24s}{'标的':>4s}{'年化':>9s}{'回撤':>8s}{'Calmar':>8}{'换手':>6}{'OOS年化':>9s}{'OOS回撤':>8s}{'OOS Cal':>8s}"
    print(hdr); print("-" * len(hdr))
    for name, uni, t1 in PLANS:
        uni = [c for c in uni if c in price.columns]
        if t1 is None:
            t1 = {c for c in uni if c in T1}
        elif isinstance(t1, set):
            t1 = {c for c in uni if c in t1}
        r = mh.simulate_daily(price, MIN_HOLD, sr_params=GOLD, def_trail_pct=None,
                              def_mom_days=10, universe=uni, t1_codes=t1 or None)
        oos = mh.slice_metrics(r, SPLIT)
        if oos:
            (ann, cum, dd) = oos
            cal = ann / abs(dd) if dd < 0 else float("nan")
            oos_s = f"{ann*100:>8.2f}{dd*100:>9.2f}{cal:>9.2f}"
        else:
            oos_s = "      N/A"
        print(f"{name:24s}{len(uni):>4d}{r['ann']*100:>9.2f}{r['maxdd']*100:>8.2f}"
              f"{r['calmar']:>8.2f}{r['switches']:>6d}{oos_s}")


if __name__ == "__main__":
    main()