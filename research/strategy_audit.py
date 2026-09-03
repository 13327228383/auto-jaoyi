# -*- coding: utf-8 -*-
"""
strategy_audit.py —— 策略稳健性审计（敏感性 + 成本/滑点 + 基线校准）
================================================================================
复用 backtest 共享原语，对「当前实盘 auto_run.py CFG」做三组量化审计：
  A) 基线校准：实盘配置 = MIN_HOLD=1 + 跟踪4%(普通&防御统一) + 防御动量门10d，
     对比现有 backtest_current.py(6%跟踪、防御不加) —— 找出"回测脚本 vs 实盘"的脱节。
  B) 参数敏感性：SLOPE_WINDOW / MA_FILTER / TRAIL_PCT / DEF_MOM_DAYS / MIN_HOLD
     逐一扫描，看是否有更优且稳健(样本内+样本外同时改善)的参数。
  C) 成本/滑点敏感性：bse.FEE 扫描 {0.05%, 0.1%(默认), 0.2%, 0.3%, 0.5%}，评估换手成本现实度。

数据：真实ETF收盘价(2015-2026)，universe=CURRENT_UNI(原6+标普500)。
注意：simulate_daily 无"再入冷却"与"迟滞带"，这两项实盘已有但回测未建模（列为已知脱节）。
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import backtest_minhold as mh
import backtest_sr_enhance as bse
import backtest_universe_ext as bue
import backtest_current as bc

SPLIT = pd.Timestamp("2021-01-01")
OOS_LABEL = "样本外2021+"

# ---- 实盘对齐配置 ----
UNIVERSE = bc.CURRENT_UNI                        # 原6 + 513500
GOLD = dict(bue.GOLD)                            # DEFENSIVE 518880 / defensive / DEF_MOM_DAYS 10

def _fmt(r):
    return (f"累计{r['cum']:7.2f}x  年化{r['ann']*100:7.2f}%  回撤{r['maxdd']*100:6.2f}%  "
            f"Calmar{r['calmar']:5.2f}  Sharpe{r['sharpe']:4.2f}  换仓{r['switches']:4d}  "
            f"止损{r['stop_exits']:3d}  持仓%{r['time_in_mkt']*100:5.1f}")

def run(min_hold=1, trail=0.04, def_trail=0.04, def_mom=10, sr_params=None, universe=UNIVERSE, fee=bse.FEE):
    old = bse.FEE
    bse.FEE = fee
    try:
        return mh.simulate_daily(bc.load_prices7(), min_hold, sr_params=sr_params or GOLD,
                                 def_trail_pct=def_trail, def_mom_days=def_mom, universe=universe)
    finally:
        bse.FEE = old

def oos(r):
    eq = r["equity"].loc[SPLIT:]
    if len(eq) < 2:
        return None
    n = max(len(eq), 1)
    cum = float(eq.iloc[-1])
    ann = float(cum ** (252.0 / n) - 1) if cum > 0 else -1.0
    maxdd = float(((eq - eq.cummax()) / eq.cummax()).min())
    return ann, cum, maxdd

def show(name, r):
    so = oos(r)
    so_s = f"  {OOS_LABEL}: 年化{so[0]*100:6.2f}%  累计{so[1]:5.2f}x/回撤{so[2]*100:6.2f}%" if so else ""
    print(f"{name:34s} {_fmt(r)}{so_s}")

def main():
    pr = bc.load_prices7()
    print(f"数据 {pr.index[0].date()} ~ {pr.index[-1].date()}，{len(pr)} 交易日，标的 {list(pr.columns)}\n")

    print("=" * 104)
    print("A) 基线校准：现有 backtest_current(6%跟踪/防御不加) vs 实盘配置(4%统一)")
    print("=" * 104)
    base_stale = mh.simulate_daily(pr, 1, sr_params=GOLD, def_trail_pct=None, def_mom_days=10, universe=UNIVERSE)
    live = mh.simulate_daily(pr, 1, sr_params=GOLD, def_trail_pct=0.04, def_mom_days=10, universe=UNIVERSE)
    show("backtest_current口径(6%/防御免)", base_stale)
    show("实盘口径(4%/防御4%)       ", live)

    print("\n" + "=" * 104)
    print("B) 参数敏感性")
    print("=" * 104)

    print("\n-- B1 SLOPE_WINDOW 打分窗口(实盘60) --")
    for w in [40, 60, 90, 120, 150]:
        sp = dict(GOLD); sp["SLOPE_WINDOW"] = w
        show(f"   slope_win={w}", run(sr_params=sp))

    print("\n-- B2 MA_FILTER 长期均线剔除(实盘60,0=关) --")
    for w in [0, 30, 60, 90, 120]:
        sp = dict(GOLD)
        if w:
            sp["MA_FILTER"] = w
        else:
            sp["MA_FILTER"] = None
        show(f"   ma_filter={w if w else 'off'}", run(sr_params=sp))

    print("\n-- B3 TRAIL_PCT 跟踪止损(实盘4%) --")
    for t in [0.03, 0.04, 0.05, 0.06, 0.08]:
        tag = "  <-实盘" if t == 0.04 else ""
        show(f"   trail={t*100:.0f}%{tag}", run(trail=t, def_trail=t))

    print("\n-- B4 DEF_MOM_DAYS 防御动量门(实盘10) --")
    for d in [5, 10, 15, 20, 30]:
        tag = "  <-实盘" if d == 10 else ""
        show(f"   def_mom={d}d{tag}", run(def_mom=d))

    print("\n-- B5 MIN_HOLD 最小持有(实盘1) --")
    for d in [0, 1, 3, 5]:
        tag = "  <-实盘" if d == 1 else ""
        show(f"   min_hold={d}d{tag}", run(min_hold=d))

    print("\n" + "=" * 104)
    print("C) 交易成本 / 滑点敏感性（默认 0.10%/次 = 双边佣金+滑点）")
    print("=" * 104)
    for fee in [0.0005, 0.0010, 0.0020, 0.0030, 0.0050]:
        tag = "  <-默认" if fee == 0.0010 else ""
        show(f"   换仓成本{fee*100:.2f}%{tag}", run(fee=fee))

    live2 = mh.simulate_daily(pr, 1, sr_params=GOLD, def_trail_pct=0.04, def_mom_days=10, universe=UNIVERSE)
    print(f"\n换仓次数={live2['switches']}，每次{bse.FEE*100:.2f}% → 全期成本拖累≈{live2['switches']*bse.FEE*100:.1f}%累计")

if __name__ == "__main__":
    main()