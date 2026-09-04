# -*- coding: utf-8 -*-
"""
backtest_megagrid_par.py —— 综合网格回测（并行加速，288 组合全跑，只看年化）
================================================================================
完整网格：3池 × 4SW(动量打分) × 3MW(绝对动量) × 2trail(3%/6%) × 2hold(1d/5d) × 2T1(全T0/真实T1)
 = 288 组合。

性能：日线决策(day_target)只由「池+SW+MW」决定，与 trail/hold/T1 无关。
 → 先【并行】(multiprocessing 4核) 预计算 3×4×3=36 个决策序列；
 → 再串行跑 288 组合，全部走决策缓存查表（秒回），只做持仓/止损/复权演进。

输出：按年化降序 Top25 + 各池最优 + 全局最优，并全量存 csv。
"""
import os, sys, itertools
import numpy as np
import pandas as pd
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import backtest_universe_ext as bue
import backtest_minhold as mh

SPLIT = pd.Timestamp("2021-01-01")
GOLD = dict(bue.GOLD)
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


def _load_price():
    allc = sorted(set(itertools.chain.from_iterable(UNIVERSES.values())) | set(T1))
    cols = {}
    for c in allc:
        cols[c] = bue.fetch_sina(c, bue.ALL_CODES[c])
    p = pd.DataFrame(cols).dropna()
    p.index = pd.to_datetime(p.index)
    return p.sort_index()


def _compute_seq(arg):
    """多进程 worker：算某个 (uni,SW,MW) 的逐日决策序列，返回 {date_str: target}。"""
    from backtest_minhold import day_target  # 原始未缓存版
    uni_n, uni_list, sw, mw = arg
    price = _load_price()
    sp = dict(GOLD); sp["SLOPE_WINDOW"] = sw; sp["MOM_WINDOW"] = mw
    seq = {}
    for i, d in enumerate(price.index):
        seq[str(d.date())] = day_target(price, d, sp, universe=list(uni_list))
    return uni_n, sw, mw, seq


# 决策查表（PART B 用）
_DEC = {}
_orig_dt = mh.day_target


def _cached_day_target(price, d, sr_params=None, universe=None):
    key_pool = tuple(sorted(str(c) for c in universe)) if universe else "def"
    key = (key_pool, sr_params.get("SLOPE_WINDOW"), sr_params.get("MOM_WINDOW"))
    seq = _DEC.get(key)
    if seq is None:
        return _orig_dt(price, d, sr_params, universe=universe)
    return seq.get(str(pd.Timestamp(d).date()), "cash")


def main():
    tasks = [(uni_n, uni_list, sw, mw)
             for uni_n, uni_list in UNIVERSES.items()
             for sw in SLOPE_WINS for mw in MOM_WINS]
    total_seq = len(tasks)
    n_tot = len(UNIVERSES) * len(SLOPE_WINS) * len(MOM_WINS) * len(TRAILS) * len(MIN_HOLDS) * len(T1_MODES)
    print(f"任务：{total_seq} 个决策序列 × 3065 日，预计 4 核并行 ~2-4 分钟；总组合 {n_tot}", flush=True)

    # ---- PART A 并行预计算决策序列 ----
    nproc = min(4, os.cpu_count() or 1)
    with Pool(processes=nproc) as pool:
        for k, (uni_n, sw, mw, seq) in enumerate(pool.imap_unordered(_compute_seq, tasks), 1):
            _DEC[(tuple(sorted(str(c) for c in UNIVERSES[uni_n])), sw, mw)] = seq
            print(f"  [决策序列 {k}/{total_seq}] 池={uni_n} SW={sw} MW={mw}", flush=True)
    print(f"决策序列全就绪；开始 288 组合串行演进（查表秒回）...", flush=True)

    # ---- PART B 串行跑 288 组合 ----
    mh.day_target = _cached_day_target   # 注入查表版
    price = _load_price()
    rows = []
    for uni_n, uni_list in UNIVERSES.items():
        for sw in SLOPE_WINS:
            for mw in MOM_WINS:
                for trail in TRAILS:
                    for hold in MIN_HOLDS:
                        for t1_n, t1_set in T1_MODES:
                            sp = dict(GOLD); sp["SLOPE_WINDOW"] = sw; sp["MOM_WINDOW"] = mw
                            r = mh.simulate_daily(price, hold, trail_pct=trail,
                                                  sr_params=sp, def_trail_pct=None,
                                                  def_mom_days=10, universe=list(uni_list),
                                                  t1_codes=t1_set)
                            so = mh.slice_metrics(r, SPLIT)
                            rows.append({
                                "pool": uni_n, "SW": sw, "MW": mw, "trail": trail,
                                "hold": hold, "t1": t1_n,
                                "ann": r["ann"] * 100, "maxdd": r["maxdd"] * 100,
                                "calmar": r["calmar"], "sharpe": r["sharpe"],
                                "switches": r["switches"], "stop_exits": r["stop_exits"],
                                "tim": r["time_in_mkt"] * 100,
                                "oos_ann": (so[0] * 100 if so else float("nan"))})
    df = pd.DataFrame(rows).sort_values("ann", ascending=False)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"\n全部 {len(df)} 组合已存 {OUT}\n", flush=True)

    print("======== 按年化 Top 25 ========")
    print(f"{'#':>3}{'池':>7}{'SW':>4}{'MW':>4}{'trail':>6}{'hold':>5}{'T1':>8}{'年化%':>8}{'回撤%':>8}{'Calmar':>7}{'OOS年化%':>9}{'换仓':>5}")
    print("-" * 76)
    for i, _, r in df.head(25).iterrows():
        print(f"{i+1:>3}{r['pool']:>7}{r['SW']:>4}{r['MW']:>4}{r['trail']*100:>5.0f}%{r['hold']:>5}"
              f"{r['t1']:>8}{r['ann']:>8.2f}{r['maxdd']:>8.2f}{r['calmar']:>7.2f}"
              f"{r['oos_ann']:>9.2f}{r['switches']:>5}")

    print("\n======== 按池子各自最优(年化) ========")
    for uni_n in UNIVERSES:
        sub = df[df["pool"] == uni_n]
        if len(sub):
            b = sub.iloc[0]
            print(f"  {uni_n:>6}: 年化{b['ann']:6.2f}%  回撤{b['maxdd']:6.2f}%  "
                  f"SW={b['SW']:.0f} MW={b['MW']:.0f} trail={b['trail']*100:.0f}% "
                  f"hold={b['hold']}d t1={b['t1']}  换仓{b['switches']}")

    b = df.iloc[0]
    print("\n======== 全局最优（只看年化） ========")
    print(f"  池 {b['pool']} / SW={b['SW']} / MW={b['MW']} / trail={b['trail']*100:.0f}% / "
          f"hold={b['hold']}d / T1={b['t1']}")
    print(f"  年化 {b['ann']:.2f}%  回撤 {b['maxdd']:.2f}%  Calmar {b['calmar']:.2f}  "
          f"OOS年化 {b['oos_ann']:.2f}%  换仓 {b['switches']}")


if __name__ == "__main__":
    main()