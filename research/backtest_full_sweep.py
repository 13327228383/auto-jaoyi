# -*- coding: utf-8 -*-
"""
backtest_full_sweep.py —— 策略结构参数「全覆盖」分层寻优（复权干净数据）
================================================================================
背景：用户指出之前只扫了 trail/halt/mhold 三个交易参数，把大量"结构参数"当固定值，
      属偷懒。本轮把真正进入决策(strategy_rotator.decide/global_risk_off)的所有可调
      维度全部打开，用"分层筛选 + 最优区细扫"控制过拟合与计算量。

第 1 层（本脚本：结构粗扫）——固定已验证的交易参数(trail=2%/halt=5%/mhold=3/冷却2)：
  SCORE 分支 slope_r2：
    SLOPE_WINDOW  {20,30,40,60}
    MA_FILTER     {30,60,None}
    HOLD_N        {1,2}
    ABS_MOM_THRESHOLD {0.0,0.02}
    DEF_MOM_DAYS  {5,10}
  SCORE 分支 mom：
    MOM_WINDOW    {10,20,40}
    MA_FILTER     {30,60,None}
    HOLD_N        {1,2}
    ABS_MOM_THRESHOLD {0.0,0.02}
    DEF_MOM_DAYS  {5,10}
  风险-off 结构参数本轮用现役默认(MA_GLOBAL=200/MA_GOLD=60/GOLD_LOOK=20/
  RISK_CONFIRM=0/RISK_EXIT=0/DEFENSE_MODE=defensive)，在第2层单独细扫。

判据（防过拟合）：
  - 样本外(2021+) Calmar 为主，全期年化/回撤/换手为辅
  - 记录换手与止损次数，避免"高换手虚高收益"
输出：各分支 Top 组合 + 单维分布，供第2层聚焦。
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

CURRENT_UNI = bue.BASE_CODES + ["513500"]
SPLIT = pd.Timestamp("2021-01-01")

T1 = {"510300", "510500", "159915"}
SLIP = 0.001
REENTER_CD = 2
# 已验证固定（第1轮不动，第2轮与结构最优联合复核）
TRAIL = 0.02
HALT = 0.05
MHOLD = 3

# 现役风险-off 结构默认（第2层单独扫）
RISK_BASE = {
    "MA_GLOBAL": 200, "MA_GOLD": 60, "GOLD_LOOK": 20,
    "RISK_CONFIRM_DAYS": 0, "RISK_EXIT_DAYS": 0, "DEFENSE_MODE": "defensive",
}


def base_sr_params():
    p = dict(bue.GOLD)
    p.setdefault("DEF_MOM_ENTER", 0.005)
    p.setdefault("DEF_MOM_EXIT", -0.008)
    p.update(RISK_BASE)
    return p


def run(p, close, open_, high, low):
    r = mh.simulate_daily_open(close, MHOLD, trail_pct=TRAIL, sr_params=p,
                               def_trail_pct=TRAIL, def_mom_days=p.get("DEF_MOM_DAYS", 10),
                               universe=CURRENT_UNI, t1_codes=T1,
                               open_=open_, slip=SLIP,
                               reenter_cooldown=REENTER_CD,
                               daily_loss_halt_pct=HALT, high_=high, low_=low)
    so = mh.slice_metrics(r, SPLIT)
    oos_ann = so[0]*100 if so else float("nan")
    oos_dd = so[2]*100 if so else float("nan")
    oos_cal = so[0]/abs(so[2]) if (so and so[2] < 0) else float("nan")
    return {
        "全期年化": r["ann"]*100, "全期回撤": r["maxdd"]*100, "全期Cal": r["calmar"],
        "样本外年化": oos_ann, "样本外回撤": oos_dd, "样本外Cal": oos_cal,
        "换手": r["switches"], "止损": r["stop_exits"],
    }


def sweep(rows, close, open_, high, low):
    for row in rows:
        p = base_sr_params()
        p.update(row["params"])
        m = run(p, close, open_, high, low)
        rows_meta = {k: row[k] for k in row if k != "params"}
        row.update(m)
    return rows


def show(rows, title):
    print(f"\n========== {title} ==========")
    # 以样本外Calmar排序，换手过高(<高收益靠换手)提示
    rr = sorted(rows, key=lambda x: (x.get("样本外Cal") if not pd.isna(x.get("样本外Cal")) else -1), reverse=True)
    for r in rr[:8]:
        seg = " | ".join(f"{k}={r[k]}" for k in r if k not in
                         ("全期年化","全期回撤","全期Cal","样本外年化","样本外回撤","样本外Cal","换手","止损")
                         and not isinstance(r[k], dict))
        print(f"  {seg:40s} 全期 {r['全期年化']:6.2f}%/-{r['全期回撤']:5.2f}% Cal{r['全期Cal']:4.2f} | "
              f"样本外 {r['样本外年化']:6.2f}%/-{r['样本外回撤']:5.2f}% Cal{r['样本外Cal']:5.2f} | "
              f"换{r['换手']} 损{r['止损']}")


def main():
    close, open_, high, low = bue.load_ohlc(CURRENT_UNI)
    print(f"统一复权OHLC {close.index[0].date()} ~ {close.index[-1].date()}，{len(close)}交易日，{len(list(close.columns))}标的")
    print(f"固定 = trail{TRAIL*100:.0f}%/halt{HALT*100:.0f}%/mhold{MHOLD}/冷却{REENTER_CD}/滑点{SLIP*100:.1f}%/万2.38/T+1")
    print(f"风险off结构默认 = {RISK_BASE}（本轮扫决策打分维度，风险off参数在第2层扫）")

    # ---- 第1层a: slope_r2 分支 ----
    rows = []
    for sw, mf, hold, absm, dm in itertools.product(
            [20, 30, 40, 60], [30, 60, None], [1, 2], [0.0, 0.02], [5, 10]):
        rows.append({
            "SCORE": "slope_r2", "SW": sw, "MA_FILTER": mf, "HOLD_N": hold,
            "ABS_MOM": absm, "DEF_MOM": dm,
            "params": {"SCORE": "slope_r2", "SLOPE_WINDOW": sw,
                       "MA_FILTER": mf, "HOLD_N": hold,
                       "ABS_MOM_THRESHOLD": absm, "DEF_MOM_DAYS": dm},
        })
    rows = sweep(rows, close, open_, high, low)
    show(rows, "slope_r2 分支（4SW×3MF×2HOLD×2ABS×2DEFMOM=96）")
    print(f"  分支总数 {len(rows)}")

    # ---- 第1层b: mom 分支 ----
    rows2 = []
    for mw, mf, hold, absm, dm in itertools.product(
            [10, 20, 40], [30, 60, None], [1, 2], [0.0, 0.02], [5, 10]):
        rows2.append({
            "SCORE": "mom", "MOMW": mw, "MA_FILTER": mf, "HOLD_N": hold,
            "ABS_MOM": absm, "DEF_MOM": dm,
            "params": {"SCORE": "mom", "MOM_WINDOW": mw, "SLOPE_WINDOW": 60,
                       "MA_FILTER": mf, "HOLD_N": hold,
                       "ABS_MOM_THRESHOLD": absm, "DEF_MOM_DAYS": dm},
        })
    rows2 = sweep(rows2, close, open_, high, low)
    show(rows2, "mom 分支（3MOMW×3MF×2HOLD×2ABS×2DEFMOM=72）")

    allr = rows + rows2
    print("\n===== 全局 Top10（全维度） =====")
    allr_s = sorted(allr, key=lambda x: (x.get("样本外Cal") if not pd.isna(x.get("样本外Cal")) else -1), reverse=True)
    for r in allr_s[:10]:
        seg = f"SCORE={r['SCORE']} " + " ".join(f"{k}={r[k]}" for k in r
            if k not in ("SCORE","params","全期年化","全期回撤","全期Cal","样本外年化","样本外回撤","样本外Cal","换手","止损"))
        print(f"  {seg:52s} 样本外Cal{r['样本外Cal']:5.2f} 全期年化{r['全期年化']:6.2f}% 换{r['换手']} 损{r['止损']}")


if __name__ == "__main__":
    main()