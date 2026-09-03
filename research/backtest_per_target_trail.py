# -*- coding: utf-8 -*-
"""
backtest_per_target_trail.py —— 7 标的 4% vs 6% 跟踪止损「逐只差异化」回测
================================================================================
背景：实盘当前只有防御黄金 518880 有独立阈值(4%)，普通标的统一用 6%(SR_TRAIL_PCT)。
  用户质疑：普通标的 6% 是从未逐只回测的"一刀切"，应像黄金一样逐只差异化。共 7 只，
  每只独立验证最合适的高点回落阈值。

口径：与 backtest_gold_peak_stop.py 一致 —— 决策序列钉死为实盘参数+迟滞带 H2，
  只在中/长持仓期对「被测试标的」叠加「从持仓以来最高收盘价回撤 trail 即卖出」。
  触发转现金后段内不买回，下个决策日按策略重建。日线收盘近似。

本脚本针对单只标的做隔离对比：基线(不开) vs 该标的 4% vs 该标的 6%，
  实盘其余标的保持 SR_TRAIL=6%（用标红标注与现行一致）。
  ——先做"只动一只"的隔离实验，避免多变量耦合，逐只找最优点。

数据：cache/backtest/etf_*.csv（7 只真实 ETF 日线）
输出：命令行对比表 + cache/backtest/per_target_trail_result.json
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
from backtest_sr_enhance import FREQ

DEFENSIVE = sr.DEFENSIVE          # '518880'

# 7 只真实 ETF（实盘标的池 = UNIVERSE）
UNIVERSE = ["510300", "510500", "159915", "518880", "513100", "511010", "513500"]

# 决策参数：与实盘一致 + 已落地迟滞带 H2（聚焦"高点回落"单一变量）
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

CACHE = os.path.join(ROOT, "cache", "backtest")


def load_prices():
    cols = {}
    for code in UNIVERSE:
        df = pd.read_csv(os.path.join(CACHE, f"etf_{code}.csv"), dtype={"date": str})
        s = df.set_index("date")["close"].astype(float)
        s.index = pd.to_datetime(s.index)
        cols[code] = s
    price = pd.DataFrame(cols).dropna()
    price.index = pd.to_datetime(price.index)
    return price.sort_index()


def decide_targets_stateful(price, dec_dates, params):
    """与 backtest_gold_peak_stop 完全相同：逐决策日调用 sr.decide。"""
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


def simulate_trail(price, dec_dates, targets, trail_map):
    """逐决策段仿真；trail_map={code: pct}，只对映射中的标的叠加高点回落，其余开着跟踪但阈值 None=不开。"""
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
        peak = {c: float(price.loc[d0_t, c]) for c in tgt}   # 建仓重置追踪峰值
        for t in seg:
            for c in list(held):
                tr = trail_map.get(c, 0) or 0
                if tr > 0:
                    px = float(price.loc[t, c])
                    peak[c] = max(peak[c], px)
                    if px <= peak[c] * (1 - tr):
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
    print(f"标的池({len(UNIVERSE)})：{'/'.join(UNIVERSE)}   调仓={FREQ}\n")

    def fmt(name, tm):
        r = simulate_trail(price, dec_dates, targets, tm)
        eq = r["equity"]
        m = metrics(eq); mo = metrics(eq, oos_ts)
        return r, m, mo, f"{name:<28s} 累计{m['cum']:6.3f}x 年化{m['ann']*100:6.2f}% 回撤{m['maxdd']*100:7.2f}% " \
                        f"换手{r['switches']:4d} 落袋{r['peak_exits']:3d} | OOS年化{mo['ann']*100:7.2f}% OOS回撤{mo['maxdd']*100:8.2f}%"

    # 全局基线
    rows = {}
    print("===== 全局基线 & 全局统一阈值 =====")
    base_r, base_m, base_mo, line = fmt("基线(全部不开)", {})
    print(line); base = base_m
    r6, m6, mo6, line = fmt("全部 6%(现行)", {c: 0.06 for c in UNIVERSE})
    print(line)
    r4, m4, mo4, line = fmt("全部 4%", {c: 0.04 for c in UNIVERSE})
    print(line)
    # 现行实盘映射：普通6%、黄金4%
    cur = {c: 0.06 for c in UNIVERSE}; cur[DEFENSIVE] = 0.04
    r_cur, m_cur, mo_cur, line = fmt("现行(普通6%+黄金4%)", cur)
    print(line)

    print("\n===== 逐标的隔离对比（只对该标的开 / 4% / 6%，其余不开） =====")
    # 表头
    print(f"{'标的':<8s} {'方案':<10s} {'累计':>7s} {'全样本年化':>10s} {'全样本回撤':>9s} {'换手':>5s} " \
          f"{'落袋':>4s} {'样本外年化':>9s} {'样本外回撤':>9s} | {'Δ年化(vs基线)':>12s}")
    per = {}
    for code in UNIVERSE:
        name = sr.ETF_UNIVERSE.get(code, code)
        per[code] = {}
        for tr_val, tag in ((0.0, "基线"), (0.04, "4%"), (0.06, "6%")):
            tm = {code: tr_val}
            r = simulate_trail(price, dec_dates, targets, tm)
            eq = r["equity"]
            m = metrics(eq); mo = metrics(eq, oos_ts)
            d_ann = m["ann"] - base["ann"] if base else 0
            star = " ◀-现行实盘" if (code == DEFENSIVE and tr_val == 0.04) or (code != DEFENSIVE and tr_val == 0.06) else ""
            print(f"{name:<8s} {tag:<10s} {m['cum']:7.3f}x {m['ann']*100:9.2f}% {m['maxdd']*100:8.2f}% "
                  f"{r['switches']:5d} {r['peak_exits']:4d} {mo['ann']*100:8.2f}% {mo['maxdd']*100:8.2f}% | "
                  f"{d_ann*100:+11.2f}pp{star}")
            per[code][tag] = {"ann": m["ann"], "maxdd": m["maxdd"], "cum": m["cum"],
                              "switches": r["switches"], "peak_exits": r["peak_exits"],
                              "oos_ann": mo["ann"], "oos_maxdd": mo["maxdd"], "d_ann": d_ann}
        print()

    out = os.path.join(CACHE, "per_target_trail_result.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"universe": UNIVERSE, "baseline": base, "global_all6": m6, "global_all4": m4,
                   "global_current": m_cur, "per_target": per}, f, ensure_ascii=False, indent=2)
    print(f"\n结果已写 {out}")


if __name__ == "__main__":
    main()