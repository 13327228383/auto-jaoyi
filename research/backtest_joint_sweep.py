# -*- coding: utf-8 -*-
"""
backtest_joint_sweep.py —— 核心可调参数「联合网格寻优」（统一复权干净数据）
================================================================================
目的：一次性彻底重扫决定"写进 DB active_param 与 auto_run.py CFG 的值"，
      不再把 DB 现役当默认值。全部使用复权干净的真实 OHLC（bue.load_ohlc 单源）。

扫描（其余按实盘机制固定：冷却2日、DEF_MOM=10+迟滞带、T+1、滑点0.1%、万2.38）：
  - trail    ∈ {0.01, 0.02, 0.03, 0.04, 0.06}  跟踪止损（普通+防御统一）
  - halt     ∈ {0(off), 0.04, 0.05, 0.06}       当日熔断阈值
  - min_hold ∈ {0, 3, 5}                        最小持仓锁(交易日)

判据（防过拟合，双窗都看）：
  - 样本外(2021+)：年化 / 回撤 / Calmar 为主
  - 全期：年化 / 回撤 / 换手 / 止损次数 为辅（换手/止损相近才可比）
输出：Top 组合表 + 各维度分布摘要，供最终定 params。
"""
import os, sys
import itertools
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import backtest_universe_ext as bue
import backtest_minhold as mh

CURRENT_UNI = bue.BASE_CODES + ["513500"]     # 当前实盘 7 只
SPLIT = pd.Timestamp("2021-01-01")

T1 = {"510300", "510500", "159915"}
SLIP = 0.001
REENTER_CD = 2
DEF_MOM = 10

# SR_PARAMS：与现役对齐的决策参数（SLOPE_WINDOW=30 / 防御迟滞带）
SR_PARAMS = dict(bue.GOLD)
SR_PARAMS.setdefault("DEF_MOM_ENTER", 0.005)
SR_PARAMS.setdefault("DEF_MOM_EXIT", -0.008)
SR_PARAMS.setdefault("SLOPE_WINDOW", 30)

GRID = {
    "trail": [0.01, 0.02, 0.03, 0.04, 0.06],
    "halt":  [0.0, 0.04, 0.05, 0.06],
    "mhold": [0, 3, 5],
}


def run(trail, halt, mhold, close, open_, high, low):
    r = mh.simulate_daily_open(close, mhold, trail_pct=trail, sr_params=SR_PARAMS,
                               def_trail_pct=trail, def_mom_days=DEF_MOM,
                               universe=CURRENT_UNI, t1_codes=T1,
                               open_=open_, slip=SLIP,
                               reenter_cooldown=REENTER_CD,
                               daily_loss_halt_pct=halt,
                               high_=high, low_=low)
    so = mh.slice_metrics(r, SPLIT)   # (ann_oos, cum_oos, maxdd_oos) or None
    return r, so


def fmt_metric(r, so):
    oos_ann = so[0]*100 if so else float("nan")
    oos_dd = so[2]*100 if so else float("nan")
    oos_cal = so[0]/abs(so[2]) if (so and so[2] < 0) else float("nan")
    return {
        "全期年化": r["ann"]*100, "全期回撤": r["maxdd"]*100,
        "全期Calmar": r["calmar"], "Sharpe": r["sharpe"],
        "换手": r["switches"], "止损": r["stop_exits"],
        "样本外年化": oos_ann, "样本外回撤": oos_dd, "样本外Calmar": oos_cal,
    }


def main():
    close, open_, high, low = bue.load_ohlc(CURRENT_UNI)
    print(f"统一复权OHLC {close.index[0].date()} ~ {close.index[-1].date()}，{len(close)}交易日，{len(list(close.columns))}标的")
    print(f"机制固定 = 冷却{REENTER_CD}日 + DEF_MOM{DEF_MOM}d+迟滞带 + T+1{len(T1)}只 + 滑点{SLIP*100:.1f}% + 万2.38")
    ncomb = len(GRID["trail"])*len(GRID["halt"])*len(GRID["mhold"])
    print(f"联合网格 {len(GRID['trail'])}×{len(GRID['halt'])}×{len(GRID['mhold'])} = {ncomb} 组合\n")

    rows = []
    for trail, halt, mhold in itertools.product(GRID["trail"], GRID["halt"], GRID["mhold"]):
        r, so = run(trail, halt, mhold, close, open_, high, low)
        m = fmt_metric(r, so)
        rows.append({"trail": trail, "halt": halt if halt else 0.0, "mhold": mhold, **m})

    df = pd.DataFrame(rows)

    # 排序：样本外 Calmar 降序（存在则用，否则全期）
    df["_score"] = df["样本外Calmar"]
    df.loc[df["_score"].isna(), "_score"] = df["全期Calmar"]
    # 约束：全期回撤不过深（-40% 内），保证可用
    df_ok = df[df["全期回撤"] > -40.0].copy()
    if len(df_ok) == 0:
        df_ok = df
    top = df_ok.sort_values("_score", ascending=False).head(12)

    print("===== Top12 组合（按样本外Calmar 降序） =====")
    print(f"{'trail':>6}{'halt':>6}{'mhold':>6} | {'全期年化':>8}{'全期回撤':>8}{'全期Cal':>7} | {'样本外年化':>9}{'样本外回撤':>9}{'样本外Cal':>8} | {'换手':>4}{'止损':>4}")
    for _, x in top.iterrows():
        print(f"{x['trail']*100:>5.0f}%{x['halt']*100:>5.0f}%{int(x['mhold']):>6d} | "
              f"{x['全期年化']:>8.2f}{x['全期回撤']:>8.2f}{x['全期Calmar']:>7.2f} | "
              f"{x['样本外年化']:>9.2f}{x['样本外回撤']:>9.2f}{x['样本外Calmar']:>8.2f} | "
              f"{int(x['换手']):>4d}{int(x['止损']):>4d}")

    print("\n===== 单维分布（按 trail 分组，其余取平均） =====")
    for col in ["trail", "halt", "mhold"]:
        g = df.groupby(col).agg(全期年化=("全期年化", "mean"), 全期回撤=("全期回撤", "mean"),
                                样本外年化=("样本外年化", "mean"), 样本外Calmar=("样本外Calmar", "mean"))
        print(f"--- {col} ---")
        for idx, row in g.iterrows():
            unit = "%" if col in ("trail", "halt") else "d"
            print(f"  {col}={idx}{unit if col in ('trail','halt') else '':>4} : 全期年化{row['全期年化']:.2f}%  回撤{row['全期回撤']:.2f}%  "
                  f"样本外年化{row['样本外年化']:.2f}%  样本外Calmar{row['样本外Calmar']:.2f}")


if __name__ == "__main__":
    main()