# -*- coding: utf-8 -*-
"""
backtest_halt_threshold.py —— 当日熔断「阈值」扫描 (3%/4%/5%/无限)
================================================================================
目的：决定实盘 MAX_DAILY_LOSS_PCT 到底设 3% 还是 5%。用真实 high/low 数据
      (峰值→谷底振幅，能抓"插针又收回")，复用 backtest_minhold.simulate_daily_open
      的【次日开盘价成交 + 滑点 + T+1 + 冷却 + 5天锁】全套口径，确保与实盘完全对齐。

当前实盘参数（与 auto_run.py CFG 对齐）：
  - 标的池   : 原6只 + 513500 = 7 只真实ETF
  - MIN_HOLD_DAYS = 5   （5天硬闸门）
  - SR_TRAIL_PCT = 0.03 / DEF_PEAK_STOP = 0.03 （跟踪止损3%）
  - REENTER_COOLDOWN_DAYS = 2（再入冷却2交易日）
  - DEF_MOM_DAYS = 10 + 迟滞带 DEF_MOM_ENTER=0.005 / DEF_MOM_EXIT=-0.008
  - T1_CODES = {510300, 510500, 159915}
  - 换仓费 COM_RATE=0.000238(万2.38)、滑点0.1%单侧
  - 熔断 action：当日持仓盘中自日内峰值回撤超阈值 → 强制转防御黄金 518880，次日重置
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import backtest_minhold as mh
import backtest_sr_enhance as bse
import backtest_universe_ext as bue

CURRENT_UNI = bue.BASE_CODES + ["513500"]     # 当前实盘 7 只
SPLIT = pd.Timestamp("2021-01-01")

# ---- 当前实盘全套参数（严格对齐 auto_run.py CFG + DB 现役 active_param 权威值）----
MIN_HOLD = bue.GOLD.get("MIN_HOLD_DAYS") or 5   # DB现役 MIN_HOLD_DAYS=5
# DB 现役(active_param 表)权威值，而非代码默认 —— 2026-09 现有 tracker 热更下发 2%
TRAIL_PCT = 0.02          # SR_TRAIL_PCT（现役=0.02，DB active_param 权威；代码默认 0.03 已被热更覆盖）
DEF_TRAIL = 0.02          # DEF_PEAK_STOP（现役=0.02，与 SR_TRAIL_PCT 同源追踪）
DEF_MOM = bue.GOLD.get("DEF_MOM_DAYS") or 10
REENTER_CD = 2
T1 = {"510300", "510500", "159915"}
SLIP = 0.001              # 滑点0.1%（实盘口径，0.1%是保守上界）
# 实盘防御迟滞带（strategy_rotator DEF_MOM_ENTER=0.005 / EXIT=-0.008）
SR_PARAMS = dict(bue.GOLD)          # 含 DEFENSIVE/GLOBAL_US/GLOBAL_GOLD/BROAD 等
SR_PARAMS.setdefault("DEF_MOM_ENTER", 0.005)
SR_PARAMS.setdefault("DEF_MOM_EXIT", -0.008)
# DB 现役权威 SLOPE_WINDOW=30（strategy_rotator 代码默认 60，必须显式传 30 才对齐实盘打分窗口）
SR_PARAMS.setdefault("SLOPE_WINDOW", 30)

CACHE = bue.CACHE


def load_ohlc_lows_highs():
    """统一复权口径加载（复用 bue.load_ohlc 单源，历史除权已前复权）。
    返回 (close_df, open_df, high_df, low_df)。"""
    return bue.load_ohlc(CURRENT_UNI)


def run(name, halt_pct):
    r = mh.simulate_daily_open(close, MIN_HOLD, trail_pct=TRAIL_PCT, sr_params=SR_PARAMS,
                               def_trail_pct=DEF_TRAIL, def_mom_days=DEF_MOM,
                               universe=CURRENT_UNI, t1_codes=T1,
                               open_=open_, slip=SLIP,
                               reenter_cooldown=REENTER_CD,
                               daily_loss_halt_pct=halt_pct,
                               high_=high, low_=low)
    so = mh.slice_metrics(r, SPLIT)
    return r, so


def fmt(name, r, so, halt_note):
    line = (f"{name:>9s} halt={halt_note:>6s} | 年化 {r['ann']*100:7.2f}% | 回撤 {r['maxdd']*100:6.2f}% | "
            f"Calmar {r['calmar']:6.2f} | Sharpe {r['sharpe']:5.2f} | 换仓 {r['switches']:4d} | 止损 {r['stop_exits']:3d}")
    if so:
        line += f"  || 样本外(2021+): 年化 {so[0]*100:7.2f}% 回撤 {so[2]*100:6.2f}% Calmar {so[0]/abs(so[2]) if so[2]<0 else float('nan'):6.2f}"
    else:
        line += "  || 样本外: 数据不足"
    return line


if __name__ == "__main__":
    close, open_, high, low = load_ohlc_lows_highs()
    print(f"真实OHLC数据 {close.index[0].date()} ~ {close.index[-1].date()}，{len(close)} 交易日，标的 {list(close.columns)}")
    print(f"口径 = 1天锁? 否,MIN_HOLD={MIN_HOLD}d + 跟踪{TRAIL_PCT*100:.0f}% + 冷却{REENTER_CD}日 + T+1{len(T1)}只 "
          f"+ 滑点{SLIP*100:.2f}% + 换仓费{bse.COM_RATE*100:.4f}% + 防御迟滞带")
    print("当日熔断阈值对比（真实 high/low，峰值→谷底振幅）：")
    print("-" * 120)

    results = {}
    for name, halt in [("无限(off)", 0.0), ("3%", 0.03), ("4%", 0.04), ("5%", 0.05), ("6%", 0.06)]:
        r, so = run(name, halt)
        results[name] = r
        print(fmt(name, r, so, f"{halt*100:.0f}%" if halt else "off"))

    print("\n=== 逐年收益对比(%) ===")
    years = sorted(set(c.year for c in close.index))
    print(f"{'年份':>6s}", *(f"{n:>12s}" for n in results), sep="")
    for y in years:
        row = []
        for name, r in results.items():
            eq = r["equity"]
            eqs = eq[eq.index.year == y]
            if len(eqs) == 0:
                row.append(float("nan"))
            else:
                base = eq[eq.index < pd.Timestamp(f"{y}-01-01")]
                prev = base.iloc[-1] if len(base) else eqs.iloc[0]
                row.append(eqs.iloc[-1] / prev - 1.0 if prev else float("nan"))
        print(f"{y:>6d}", *(f"{v*100:12.2f}" if not np.isnan(v) else f"{'--':>12s}" for v in row), sep="")