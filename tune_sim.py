# -*- coding: utf-8 -*-
"""
tune_sim.py —— 模拟近一年交易，量化"自动调参收益"（沙箱，绝不影响真钱）
================================================================================
目标：在"近一年"这段上，用真实回测引擎(实盘口径: 当日收盘+双边滑点0.1%+佣金)
把每个候选参数组重放一遍，得到其模拟年化/回撤/换手 → 存 sim_panel。

然后按与 auto_tuner 相同的选优逻辑（最高模拟年化、样本充足即采纳）给出：
  "若把近一年当实盘数据跑调参器，它会从现役切到哪组、模拟增益多少 pp"。

安全边界：
  - 模拟结果只写入 sim_panel（面板，供查看/报告），绝不写入 active_param——不会触发真钱换参。
  - 不往 trade_log 灌假成交（避免污染自动调参器的真实 round-trip 计算）。
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(HERE, "research"))

import backtest_universe_ext as bue
import backtest_minhold as mh
import tuning_db as db

CODES = sorted({'510300', '510500', '159915', '518880', '513100', '511010', '513500'})
T1 = {'510300', '510500', '159915'}
SLIP = 0.001
N_DAYS = 252          # 近一年


def simulate_win(close, sw, tr, moh, win_start):
    pr = close[close.index >= win_start]
    if len(pr) < 60:
        return None
    G = dict(bue.GOLD); G['SLOPE_WINDOW'] = sw; G['MOM_WINDOW'] = 20
    G['DEF_MOM_ENTER'] = 0.005; G['DEF_MOM_EXIT'] = -0.008
    r = mh.simulate_daily(pr, moh, trail_pct=tr, sr_params=G, def_trail_pct=tr,
                          def_mom_days=10, universe=CODES, t1_codes=T1, slip=SLIP)
    eq = r['equity']
    tail = eq.iloc[-N_DAYS:]
    if len(tail) < 40 or tail.iloc[-1] <= 0:
        return None
    n = len(tail)
    ann = (float(tail.iloc[-1] / tail.iloc[0]) ** (252.0 / n) - 1) * 100
    dd = float(((tail - tail.cummax()) / tail.cummax()).min()) * 100
    return ann, dd, r['switches']


def main():
    close, _ = bue.load_prices_open(CODES)
    close = close[close.index >= '2014-01-01']
    win_end = close.index[-1]
    # 取近 N_DAYS 之前约 400 日做预热(冷启动 + 让斜坡/移动均线有效)
    win_start = close.index[-min(N_DAYS + 400, len(close))]
    warm = close.index[-N_DAYS]
    sim_start = win_start   # 模拟窗口起点（含预热），年化只看近 N 天? 
    # 简化：模拟整段(含预热)得到 equity，但"等价年化"用近 N_DAYS 的几何收益
    warm_pr = close[close.index >= win_start]
    rw = None
    for (sid, cfg) in db.list_param_sets():
        sw, tr, moh = cfg["SW"], cfg["trail"], cfg["MOH"]
        res = simulate_win(close, sw, tr, moh, win_start)
        if res is None:
            continue
        ann, dd, swc = res
        db.save_sim_panel(sid, str(win_start.date()), str(warm.date()), ann, dd, swc,
                          "tune_sim 近一年重放(实盘口径)")
        print(f"  param#{sid}: SW{sw} trail{tr:.1%} moh{moh}  模拟年化 {ann:6.1f}%  "
              f"回撤 {dd:5.1f}%  换手 {swc}", flush=True)

    panel = db.get_sim_panel()
    act = db.get_active_params()
    cur_id = db.get_param_set_id(act)
    if not panel:
        print("无模拟结果"); return

    print("\n== 近一年模拟·候选榜（按模拟年化降序） ==")
    best_id, best_ann = None, -1e18
    for (pid, ann, dd, swc) in panel:
        mark = "  <- 现役" if pid == cur_id else ""
        if ann > best_ann:
            best_ann, best_id = ann, pid
        print(f"  param#{pid}: 模拟年化 {ann:6.1f}%  回撤{dd:5.1f}%  换手{swc}{mark}")

    cur_ann = next((a for (p, a, _, _) in panel if p == cur_id), None)
    print(f"\n== 调参器近一年会怎么选 == 最佳候选 param#{best_id} "
          f"({best_ann:.1f}%) vs 现役 param#{cur_id} ({cur_ann if cur_ann is not None else '—'})")
    if cur_ann is not None:
        print(f"   模拟增益 = {best_ann - cur_ann:+.2f}pp")
    print("\n（沙箱仿真：结果存 sim_panel，未改 active_param，真钱不受影响）")


if __name__ == "__main__":
    main()