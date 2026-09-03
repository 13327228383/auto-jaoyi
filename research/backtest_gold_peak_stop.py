# -*- coding: utf-8 -*-
"""
backtest_gold_peak_stop.py —— 「防御黄金(518880) 高点回落卖出」的前后回测对比
================================================================================
背景：实盘/回测目前对防御资产(DEFENSIVE=518880)不开跟踪止损（auto_run.enforce_hard_stop
  用 `code != DEFENSIVE` 明确排除；backtest_sr_enhance.simulate 对 DEF 也是 stop/tp=None），
  因此黄金冲高回落后利润守着不落袋（例：某日盘中盈利 900+ → 回落只剩 500+ 未锁定）。

本回测：决策序列固定为「已落地的实盘参数 + 迟滞带 H2(0.005,-0.008)」（即当前实盘决策层），
  只在中/长持仓期对 DEF 叠加「从持仓以来最高收盘价回撤 trail 即卖出」，对比不同阈值。
  因决策层不感知高位回落（这是执行层止损），故各阈值共用同一决策序列，聚焦单一变量。

口径：日线收盘近似。实盘用盘中实时价追踪峰值，日线用「收盘 cummax」作为追踪高位，
  更保守（盘中更高峰/更早触发不会高估）。触发转现金后段内不买回，下个决策日按策略重建。

数据：cache/backtest/final_*.csv（同 backtest_sr_enhance，2016-2026 全历史，指数代理）
输出：命令行对比表 + cache/backtest/gold_peak_stop_result.json
"""
import os
import sys
import json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import strategy_rotator as sr
sys.path.insert(0, HERE)
from backtest_sr_enhance import FREQ, load_prices

DEFENSIVE = sr.DEFENSIVE          # '518880'

# 决策参数：与实盘钉死一致 + 已落地的迟滞带 H2（本回测聚焦「高点回落」单一变量）
BASE_PARAMS = {
    "HOLD_N": 1, "SCORE": "slope_r2", "MA_FILTER": 60,
    "DEFENSIVE": DEFENSIVE, "DEF_MOM_DAYS": 10,
    "GLOBAL_US": "513100", "GLOBAL_GOLD": "518880", "BROAD": "510300",
    "MA_GLOBAL": sr.MA_GLOBAL, "MA_GOLD": sr.MA_GOLD, "GOLD_LOOK": sr.GOLD_LOOK,
    "MACRO_OFF": False,
    "DEFENSE_MODE": "defensive",
    "RISK_CONFIRM_DAYS": 0, "RISK_EXIT_DAYS": 0,
    "DEF_MOM_ENTER": 0.005, "DEF_MOM_EXIT": -0.008,   # H2 已落地
}

# 高点回落阈值变体（0.0=关闭=现行基线）
TRAILS = [0.0, 0.04, 0.06, 0.08, 0.10]


def decide_targets_stateful(price, dec_dates, params):
    targets = {}
    prev_def_held = False
    for d in dec_dates:
        hist = price.loc[:d]
        if len(hist) < 200:
            targets[d] = "cash"
            continue
        prices = {c: hist[c].values for c in price.columns}
        res = sr.decide(prices, params, risk_state=None, prev_def_held=prev_def_held)
        t = res["target"]
        targets[d] = t
        prev_def_held = (t != "cash") and (DEFENSIVE in t)
    return targets


def simulate_peak_stop(price, dec_dates, targets, trail_pct):
    """逐决策段仿真，DEF 持仓期内叠加高点回落卖出。返回 equity 及各指标。"""
    dates = price.index
    daily_ret = price.pct_change().fillna(0.0)
    port = pd.Series(0.0, index=dates)
    switches = 0
    peak_exits = 0
    invested_days = 0
    prev_key = None
    for i in range(1, len(dec_dates)):
        d0, d1 = dec_dates[i - 1], dec_dates[i]
        tgt = targets[d0]
        seg = dates[(dates > d0) & (dates <= d1)]
        if len(seg) == 0:
            continue
        _past = price.index[price.index <= d0]
        d0_t = _past[-1] if len(_past) else d0
        key = "cash" if tgt == "cash" else tuple(sorted(tgt))
        if prev_key is not None and key != prev_key:
            switches += 1
        prev_key = key
        if tgt == "cash":
            port.loc[seg] = 0.0
            continue
        held = list(tgt)
        weights = {c: 1.0 for c in tgt}
        # 每次建仓重置追踪峰值为建仓价
        peak = {c: float(price.loc[d0_t, c]) for c in tgt}
        for t in seg:
            for c in list(held):
                px = float(price.loc[t, c])
                if c == DEFENSIVE and trail_pct > 0:
                    peak[c] = max(peak[c], px)
                    if px <= peak[c] * (1 - trail_pct):
                        held.remove(c)
                        peak_exits += 1
                        switches += 1          # 卖→现金，产生换手
            if not held:
                port.loc[t] = 0.0
            else:
                port.loc[t] = sum(weights[c] * float(daily_ret.loc[t, c]) for c in held)
            if port.loc[t] != 0.0:
                invested_days += 1
    equity = (1 + port).cumprod()
    return {"equity": equity, "switches": switches, "peak_exits": peak_exits,
            "invested_days": invested_days}


def metrics(equity, start_ts=None):
    if start_ts is not None:
        equity = equity.loc[equity.index >= start_ts]
    if len(equity) == 0:
        return None
    cum = float(equity.iloc[-1])
    ann = float(cum ** (252.0 / max(len(equity), 1)) - 1) if cum > 0 else -1.0
    maxdd = float(((equity - equity.cummax()) / equity.cummax()).min())
    return {"cum": cum, "ann": ann, "maxdd": maxdd}


def main():
    price = load_prices()
    dec_dates = list(price.resample(FREQ).last().index)
    dec_dates = [d for d in dec_dates if len(price.loc[:d]) >= 200]
    targets = decide_targets_stateful(price, dec_dates, BASE_PARAMS)
    oos_ts = pd.Timestamp("2021-01-01")
    print(f"数据 {price.index[0].date()} ~ {price.index[-1].date()}，{len(price)} 交易日，{len(dec_dates)} 决策日")
    print(f"防御资产={DEFENSIVE}（黄金指数代理） 迟滞带 H2(+0.5%/-0.8%) 调仓={FREQ}\n")
    print(f"{'方案':26s} {'累计':>6s} {'年化':>7s} {'回撤':>7s} {'换手':>5s} {'高位落袋':>6s} | {'样本外年化':>8s} {'样本外回撤':>8s}")

    rows, star = {}, {}
    for tr in TRAILS:
        if tr > 0:
            schem = f"peak-{tr*100:.0f}%"
        else:
            schem = "基线(不开)"
        r = simulate_peak_stop(price, dec_dates, targets, tr)
        eq = r["equity"]
        m = metrics(eq)
        mo = metrics(eq, oos_ts)
        rows[schem] = {"trail": tr, "switches": r["switches"], "peak_exits": r["peak_exits"],
                       "invested_days": r["invested_days"], **m,
                       "oos_ann": mo["ann"], "oos_maxdd": mo["maxdd"]}
        print(f"{schem:26s} {m['cum']:6.3f}x {m['ann']*100:6.2f}% {m['maxdd']*100:7.2f}% "
              f"{r['switches']:5d} {r['peak_exits']:6d}ab | {mo['ann']*100:7.2f}% {mo['maxdd']*100:8.2f}%")

    base = rows["基线(不开)"]
    print("\n=== 对比基线 ===")
    for schem, r in rows.items():
        if schem == "基线(不开)":
            continue
        print(f"{schem:20s} 年化 {r['ann']*100-base['ann']*100:+6.2f}pp(基线{base['ann']*100:.2f}) "
              f"回撤 {r['maxdd']*100-base['maxdd']*100:+.2f}pp(基线{base['maxdd']*100:.2f}) "
              f"换手 {r['switches']-base['switches']:+3d} 高位落袋 {r['peak_exits']} 次")

    out = os.path.join(ROOT, "cache", "backtest", "gold_peak_stop_result.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"\n结果已写 {out}")


if __name__ == "__main__":
    main()