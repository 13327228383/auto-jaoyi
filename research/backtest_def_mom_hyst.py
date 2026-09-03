# -*- coding: utf-8 -*-
"""
backtest_def_mom_hyst.py —— 防御资产动量门「迟滞带」的前后回测对比
================================================================================
问题：_def_momentum_ok 用 `m>0`（零阈值），518880 的 10 日动量常在 0% 附近抖动，
  会触发同一下午"清仓→又买回"的无谓 whipsaw（例：2026-09-02 14:19 卖→14:26 买）。
方案：加迟滞带 —— 已持有→动量须跌破 DEF_MOM_EXIT 才卖；未持有→动量须突破 DEF_MOM_ENTER 才买。
  EXIT < ENTER 构成"维持原状"区间，零轴附近不再来回切。默认 (0,0) 等效原版 `m>0`.

数据：cache/backtest/final_*.csv（同 backtest_sr_enhance，2016-2026 全历史，指数代理）
口径：2W 双周调仓，成本 0.1%/次；与基线一致。防御资产此处 = 518880|黄金(518880 是与实盘同标的)。
参数与实盘钉死一致：DEFENSE_MODE=defensive、RISK_CONFIRM_DAYS=0、RISK_EXIT_DAYS=0、DEF_MOM_DAYS=10。
【状态相关】逐决策日回传"上一决策日实际是否持有防御资产"，正确模拟迟滞的连续性。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import strategy_rotator as sr
sys.path.insert(0, HERE)
from backtest_sr_enhance import FREQ, load_prices, simulate

DEFENSIVE = sr.DEFENSIVE          # '518880'

# 迟滞参数组：name -> (DEF_MOM_ENTER, DEF_MOM_EXIT)  （单位：小数，%）
VARIANTS = [
    ("H0 基线(无迟滞 m>0)",  (0.0, 0.0)),
    ("H1 对称 ±0.5%",        (0.005, -0.005)),
    ("H2 进0.5% 退-0.8%",    (0.005, -0.008)),
    ("H3 较宽 进1% 退-1%",   (0.010, -0.010)),
    ("H4 进0.8% 退-0.8%",    (0.008, -0.008)),
    ("H5 只抬进 进0.5% 退0", (0.005, 0.0)),
]

# 与实盘钉死一致的组合参数（右除迟滞，逐组覆盖）
BASE_PARAMS = {
    "HOLD_N": 1, "SCORE": "slope_r2", "MA_FILTER": 60,
    "DEFENSIVE": DEFENSIVE, "DEF_MOM_DAYS": 10,
    "GLOBAL_US": "513100", "GLOBAL_GOLD": "518880", "BROAD": "510300",
    "MA_GLOBAL": sr.MA_GLOBAL, "MA_GOLD": sr.MA_GOLD, "GOLD_LOOK": sr.GOLD_LOOK,
    "MACRO_OFF": False,
    "DEFENSE_MODE": "defensive",
    "RISK_CONFIRM_DAYS": 0, "RISK_EXIT_DAYS": 0,
}


def decide_targets_stateful(price, dec_dates, params):
    """逐决策日决策，并回传上一日是否持有防御资产（迟滞状态连续性）。"""
    targets = {}
    prev_def_held = False          # 初始未持有防御资产
    for d in dec_dates:
        hist = price.loc[:d]
        if len(hist) < 200:
            targets[d] = "cash"
            continue
        prices = {c: hist[c].values for c in price.columns}
        res = sr.decide(prices, params, risk_state=None, prev_def_held=prev_def_held)
        t = res["target"]
        targets[d] = t
        # 更新"当前是否持有防御资产"状态，供下一决策日判定退出阈值
        prev_def_held = (t != "cash") and (DEFENSIVE in t)
    return targets


def main():
    price = load_prices()
    dec_dates = list(price.resample(FREQ).last().index)
    dec_dates = [d for d in dec_dates if len(price.loc[:d]) >= 200]
    print(f"数据 {price.index[0].date()} ~ {price.index[-1].date()}，共 {len(price)} 交易日，{len(dec_dates)} 个决策日")
    print(f"防御资产：{DEFENSIVE}（黄金ETF，与实盘同标的）  调仓频率={FREQ}  成本={0.1}%/次\n")

    results = {}
    for name, (enter, exit) in VARIANTS:
        params = dict(BASE_PARAMS)
        params["DEF_MOM_ENTER"] = enter
        params["DEF_MOM_EXIT"] = exit
        targets = decide_targets_stateful(price, dec_dates, params)
        results[name] = simulate(price, dec_dates, targets, {})
        r = results[name]
        print(f"{name:22s} 累计 {r['cum']:6.3f}x 年化 {r['ann']*100:6.2f}% "
              f"回撤 {r['maxdd']*100:7.2f}% 换仓 {r['switches']:4d} 持仓占比 {r['time_in_mkt']*100:5.1f}%")

    base = results["H0 基线(无迟滞 m>0)"]
    print("\n=== 对比基线 ===")
    for name in [v[0] for v in VARIANTS[1:]]:
        r = results[name]
        d_sw = r['switches'] - base['switches']
        print(f"{name:22s} 年化 {r['ann']*100:+6.2f}pp(基线{base['ann']*100:.2f})  "
              f"回撤 {r['maxdd']*100:+.2f}pp(基线{base['maxdd']*100:.2f})  "
              f"换仓 {d_sw:+d}(基线{base['switches']})")


if __name__ == "__main__":
    main()