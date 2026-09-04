# -*- coding: utf-8 -*-
"""
backtest_megagrid_fast.py —— 综合网格回测（加速版，只看年化）
================================================================================
同 backtest_megagrid.py 的网格 3池×4SW×3MW×2trail×2hold×2T1=288 组合。
加速关键：day_target(日线决策)只由「池 universe + 打分窗口SW + 绝对动量MW」决定，
与 trail/hold/T1 无关。故对每个 (uni,SW,MW) 只需算一次每日决策序列，其余组合
(如改动 host/trail/t1) 直接查缓存 → 决定调用量从 288×3065 降到 (3×4×3)×3065≈1/8。

决策 cached monkeypatch 注入 mh.day_target，复用同一 simulate_daily 逻辑，不复制。
"""
import os, sys
import itertools
import pandas as pd

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

# ---- 决策缓存 monkeypatch ----
_orig_day_target = mh.day_target
_DEC = {}


def _cached_day_target(price, d, sr_params=None, universe=None):
    key = (tuple(sorted(str(c) for c in universe)) if universe else "def",
           sr_params.get("SLOPE_WINDOW"), sr_params.get("MOM_WINDOW"))
    sub = _DEC.get(key)
    if sub is None:
        sub = {}
        _DEC[key] = sub
    pkey = pd.Timestamp(d)
    if pkey not in sub:
        sub[pkey] = _orig_day_target(price, d, sr_params, universe=universe)
    return sub[pkey]


mh.day_target = _cached_day_target


def load_superset():
    allc = sorted(set(itertools.chain.from_iterable(UNIVERSES.values())) | set(T1))
    cols = {}
    for c in allc:
        cols[c] = bue.fetch_sina(c, bue.ALL_CODES[c])
    p = pd.DataFrame(cols).dropna()
    p.index = pd.to_datetime(p.index)
    return p.sort_index()


def main():
    price = load_superset()
    n_tot = len(UNIVERSES) * len(SLOPE_WINS) * len(MOM_WINS) * len(TRAILS) * len(MIN_HOLDS) * len(T1_MODES)
    print(f"数据 {price.index[0].date()} ~ {price.index[-1].date()}，{len(price)} 交易日；总组合 {n_tot}（决策复用 $ (3*4*3) 次）", flush=True)

    rows = []
    cnt = 0
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
                            rows.append({**{
                                "pool": uni_n, "SW": sw, "MW": mw, "trail": trail,
                                "hold": hold, "t1": t1_n,
                                "ann": r["ann"] * 100, "maxdd": r["maxdd"] * 100,
                                "calmar": r["calmar"], "sharpe": r["sharpe"],
                                "switches": r["switches"], "stop_exits": r["stop_exits"],
                                "tim": r["time_in_mkt"] * 100,
                                "oos_ann": (so[0] * 100 if so else float("nan"))}})
                            cnt += 1
    df = pd.DataFrame(rows).sort_values("ann", ascending=False)
    df.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"\n全部 {len(df)} 组合已存 {OUT}\n")

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