# -*- coding: utf-8 -*-
"""
backtest_t1_constraint.py —— T+0 vs T+1 制度对轮动收益的真实影响
================================================================================
回答用户灵魂质疑："T+1 从最高回撤3%离场，当天买走不了，绕不开"。

对比两组（同一套验证过的 simulate_daily，含前视消除+万2.38摩擦）：
  A. t1=None    : 全部按 T+0，当日触发回撤当日离场（=当前生产脚本口径，高估）
  B. t1={3只}   : 沪深300/中证500/创业板 按真实 T+1，当日触发次日才离场（承受次日下探）

数据/配置同 backtest_current（当前实盘口径：原6+513500，MIN_HOLD=1，黄金防御，
SLOPE_R2 选标，SR止损+入场过滤+回撤3%跟踪止损）。
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import backtest_minhold as mh
import backtest_current as bc

GOLD = {
    "DEFENSIVE": "518880", "DEFENSE_MODE": "defensive", "DEF_MOM_DAYS": 10,
    "GLOBAL_US": "513100", "GLOBAL_GOLD": "518880", "BROAD": "510300",
}
CURRENT_UNI = ["510300", "510500", "159915", "518880", "513100", "511010", "513500"]
DEF_TRAIL = None
DEF_MOM = 10

T1 = {"510300", "510500", "159915"}          # 真实 T+1
MIN_HOLD = 1
SPLIT = pd.Timestamp("2021-01-01")


def main():
    price = bc.load_prices7()
    print(f"数据 {price.index[0].date()} ~ {price.index[-1].date()}  {len(price)}交易日, 标的 {list(price.columns)}\n")

    for name, t1 in [("全按T+0(当日离场,现状口径)", None),
                     ("T+1标的回撤延迟1日(真实制度)", T1)]:
        r = mh.simulate_daily(price, MIN_HOLD, sr_params=GOLD,
                              def_trail_pct=DEF_TRAIL, def_mom_days=DEF_MOM,
                              universe=CURRENT_UNI, t1_codes=t1)
        # 样本外
        oos = mh.slice_metrics(r, SPLIT)
        print(f"\n【{name}】")
        print(f"  全期: 累计{r['cum']:.2f}x  年化{r['ann']*100:.2f}%  回撤{r['maxdd']*100:.2f}%  "
              f"Calmar{r['calmar']:.2f}  换手{r['switches']}  持仓{r['time_in_mkt']*100:.1f}%")
        if oos:
            (ann, cum, maxdd) = oos
            import numpy as _np
            calmar = ann / abs(maxdd) if maxdd < 0 else float("nan")
            print(f"  样本外(2021+): 累计{cum:.2f}x  年化{ann*100:.2f}%  "
                  f"回撤{maxdd*100:.2f}%  Calmar{calmar:.2f}")


if __name__ == "__main__":
    main()