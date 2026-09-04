# -*- coding: utf-8 -*-
"""
backtest_megagrid.py —— 综合网格回测（一锤定音，只看年化）
================================================================================
用户诉求：反复横跳太多次，做一次「全面」网格回测，覆盖所有核心参数组合，
只看年化（其余指标仅作附带参考）。本脚本用同一套真实ETF数据(新浪缓存)、
同一模拟器(simulate_daily)，网格扫描：

  池子 universe:  BASE7(现7只) / T0_4(纯T+0四只) / T0_5(+恒生)
  动量打分窗口 SW: 20 / 40 / 60 / 80   (slope×R² 打分窗口)
  绝对动量窗口 MW: 10 / 20 / 30        (绝对动量过滤窗口)
  跟踪止损 trail:  3% / 6%
  最小调仓间隔 min_hold: 1d / 5d
  T+1 处理 t1:     全T+0口径 / 真实T+1延迟1日

输出：按「年化」降序的 Top 组合 + 存入 megagrid_results.csv 便于核对。
"""
import os, sys, csv, json
import itertools
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import backtest_universe_ext as bue
import backtest_minhold as mh

SPLIT = pd.Timestamp("2021-01-01")
GOLD = dict(bue.GOLD)     # DEFENSIVE=518880 / DEFENSE_MODE=defensive / DEF_MOM_DAYS=10
T1 = {"510300", "510500", "159915"}

UNIVERSES = {
    "BASE7": ["510300", "510500", "159915", "518880", "513100", "511010", "513500"],
    "T0_4":  ["518880", "513100", "511010", "513500"],
    "T0_5":  ["518880", "513100", "511010", "513500", "159920"],
}
MIN_HOLDS = [1, 5]
TRAILS = [0.03, 0.06]
SLOPE_WINS = [20, 40, 60, 80]
MOM_WINS = [10, 20, 30]
T1_MODES = [("full_t0", None), ("real_t1", T1)]

OUT = os.path.join(HERE, "megagrid_results.csv")


def load_superset():
    allc = sorted(set(itertools.chain.from_iterable(UNIVERSES.values())) | set(T1))
    cols = {}
    for c in allc:
        s = bue.fetch_sina(c, bue.ALL_CODES[c])
        cols[c] = s
    p = pd.DataFrame(cols).dropna()
    p.index = pd.to_datetime(p.index)
    return p.sort_index()


def main():
    price = load_superset()
    print(f"数据 {price.index[0].date()} ~ {price.index[-1].date()}，{len(price)} 交易日")
    print(f"网格 = {len(UNIVERSES)}池 × {len(SLOPE_WINS)}SW × {len(MOM_WINS)}MW × {len(TRAILS)}trail "
          f"× {len(MIN_HOLDS)}hold × {len(T1_MODES)}T1 = "
          f"{len(UNIVERSES)*len(SLOPE_WINS)*len(MOM_WINS)*len(TRAILS)*len(MIN_HOLDS)*len(T1_MODES)} 组合\n", flush=True)

    rows = []
    count = 0
    for uni_n, uni_list in UNIVERSES.items():
        for sw in SLOPE_WINS:
            for mw in MOM_WINS:
                for trail in TRAILS:
                    for hold in MIN_HOLDS:
                        for t1_n, t1_set in T1_MODES:
                            sp = dict(GOLD)
                            sp["SLOPE_WINDOW"] = sw
                            sp["MOM_WINDOW"] = mw
                            r = mh.simulate_daily(price, hold, trail_pct=trail,
                                                  sr_params=sp, def_trail_pct=None,
                                                  def_mom_days=10, universe=list(uni_list),
                                                  t1_codes=t1_set)
                            so = mh.slice_metrics(r, SPLIT)
                            rec = {
                                "pool": uni_n, "SW": sw, "MW": mw, "trail": trail,
                                "hold": hold, "t1": t1_n,
                                "ann": r["ann"] * 100, "maxdd": r["maxdd"] * 100,
                                "calmar": r["calmar"], "sharpe": r["sharpe"],
                                "switches": r["switches"], "stop_exits": r["stop_exits"],
                                "tim": r["time_in_mkt"] * 100,
                                "oos_ann": (so[0] * 100 if so else float("nan")),
                            }
                            rows.append(rec)
                            count += 1
                            if count % 60 == 0:
                                print(f"  已跑 {count}/{len(UNIVERSES)*len(SLOPE_WINS)*len(MOM_WINS)*len(TRAILS)*len(MIN_HOLDS)*len(T1_MODES)}", flush=True)

    df = pd.DataFrame(rows).sort_values("ann", ascending=False)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"\n全部 {len(df)} 组合已存 csv：{OUT}\n")
    print("======== 按年化 Top 25 ========")
    cols = ["pool", "SW", "MW", "trail", "hold", "t1", "ann", "maxdd", "calmar", "oos_ann", "switches"]
    hdr = f"{'#':>3}{'池':>7}{'SW':>4}{'MW':>4}{'trail':>6}{'hold':>5}{'T1':>8}{'年化%':>8}{'回撤%':>8}{'Calmar':>7}{'OOS年化%':>9}{'换仓':>5}"
    print(hdr); print("-" * len(hdr))
    for i, _, row in df.head(25).iterrows():
        print(f"{i+1:>3}{row['pool']:>7}{row['SW']:>4}{row['MW']:>4}{row['trail']*100:>5.0f}%{row['hold']:>5}"
              f"{row['t1']:>8}{row['ann']:>8.2f}{row['maxdd']:>8.2f}{row['calmar']:>7.2f}"
              f"{row['oos_ann']:>9.2f}{row['switches']:>5}")

    print("\n======== 按池子各自最优(年化) ========")
    for uni_n in UNIVERSES:
        sub = df[df["pool"] == uni_n]
        if len(sub):
            b = sub.iloc[0]
            print(f"  {uni_n:>6}: 年化{b['ann']:6.2f}%  回撤{b['maxdd']:6.2f}%  "
                  f"SW={b['SW']:.0f} MW={b['MW']:.0f} trail={b['trail']*100:.0f}% hold={b['hold']}d t1={b['t1']}")

    # 全局最优
    b = df.iloc[0]
    print("\n======== 全局最优（年化优先） ========")
    print(f"  池 {b['pool']} / SW={b['SW']} / MW={b['MW']} / trail={b['trail']*100:.0f}% / "
          f"hold={b['hold']}d / T1={b['t1']}")
    print(f"  年化 {b['ann']:.2f}%  回撤 {b['maxdd']:.2f}%  Calmar {b['calmar']:.2f}  "
          f"OOS年化 {b['oos_ann']:.2f}%  换仓 {b['switches']}")


if __name__ == "__main__":
    main()