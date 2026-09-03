# -*- coding: utf-8 -*-
"""
backtest_intraday_trail.py —— A项「盘中快循环(高频读价判定)」 vs 「收盘判定」回测
================================================================================
背景：A 项 = 把峰值/止损判定从「每决策周期(120s)按收盘」改为「盘中 5~10s 高频读实时价」，
  好处是能更早捕捉盘中冲高后的回落(用盘中最高价更新峰、跌破止损价即离场)，
  而不是等到收盘。用户要求：A 要做，但先回测验证其价值与副作用(是否更多误杀)。

数据限制：只有日线 OHLC(新浪)，无真分钟线。用「日内 high/low/close」近似盘中：
  - 收盘判定(现状)：峰来源累计收盘价，触发用当日收盘 close
  - 盘中判定(快循环)：峰来源累计「日内高」high（更接近盘中冲高），触发用日内低 low
    （即盘中任何时刻跌破止损就离场，比收盘更早/更频繁）
对比二者在 全期/样本外 的年化、回撤、换手、落袋数 —— 验证"更早触发"是否值得。

口径：复用 backtest_per_target_trail 决策序列 + 逐段仿真，仅替换 peak 来源与触发价。
输出：命令行对比表
"""
import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)
from backtest_per_target_trail import (UNIVERSE, BASE_PARAMS, load_prices,
                                       decide_targets_stateful)

CACHE = os.path.join(ROOT, "cache", "backtest")
TRAIL = {c: 0.04 for c in UNIVERSE}   # 统一 4%


def load_ohlc():
    """OHLC 面板，索引日期，列 MultiIndex(code, OHLC)。"""
    frames = {}
    for code in UNIVERSE:
        df = pd.read_csv(os.path.join(CACHE, f"ohlc_{code}.csv"), dtype={"date": str})
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        frames[code] = df[["open", "high", "low", "close"]]
    merged = pd.concat(frames, axis=1)          # MultiIndex cols
    return merged.sort_index().dropna()


def simulate_intraday(price, ohlc, dec_dates, targets, trail_map, intraday):
    """intraday=False: 收盘判定(现状)；True: 盘中判定(high更新峰, low触发)。"""
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
        peak = {c: float(price.loc[d0_t, c]) for c in tgt}
        for t in seg:
            for c in list(held):
                tr = trail_map.get(c, 0) or 0
                if tr > 0:
                    # 峰值来源：收盘(现状) vs 盘中高(快循环更早捕捉冲高)
                    if intraday:
                        px_up = float(ohlc.loc[t, (c, "high")])
                        px_down = float(ohlc.loc[t, (c, "low")])
                    else:
                        px_up = float(price.loc[t, c])
                        px_down = px_up
                    peak[c] = max(peak[c], px_up)
                    if px_down <= peak[c] * (1 - tr):
                        held.remove(c)
                        peak_exits += 1
                        switches += 1
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
    ohlc = load_ohlc()
    idx = price.index.intersection(ohlc.index)
    price = price.loc[idx]
    ohlc = ohlc.loc[idx]
    dec_dates = list(price.resample("2W").last().index)
    dec_dates = [d for d in dec_dates if len(price.loc[:d]) >= 200]
    targets = decide_targets_stateful(price, dec_dates, BASE_PARAMS)
    oos_ts = pd.Timestamp("2021-01-01")
    print(f"数据 {price.index[0].date()} ~ {price.index[-1].date()}，{len(price)} 交易日，调仓 2W，跟踪统一 4%")
    print("对比：收盘判定(现状) vs 盘中判定(高更新峰/低触发=5-10s快循环)\n")

    rows = {}
    for name, intraday in (("收盘判定(现状)", False), ("盘中判定(快循环)", True)):
        r = simulate_intraday(price, ohlc, dec_dates, targets, TRAIL, intraday)
        eq = r["equity"]
        m = metrics(eq); mo = metrics(eq, oos_ts)
        calmar = m["ann"] / abs(m["maxdd"]) if m["maxdd"] < 0 else float("nan")
        rows[name] = (r, m, mo)
        print(f"{name:<16s} 累计{m['cum']:6.3f}x 年化{m['ann']*100:7.2f}% 回撤{m['maxdd']*100:7.2f}% "
              f"Calmar{calmar:5.2f} 换手{r['switches']:4d} 落袋{r['peak_exits']:3d} "
              f"持仓{r['invested_days']/max(len(eq)-1,1)*100:5.1f}% | OOS年化{mo['ann']*100:7.2f}% "
              f"OOS回撤{mo['maxdd']*100:7.2f}%")

    (r0, m0, mo0) = rows["收盘判定(现状)"]
    (r1, m1, mo1) = rows["盘中判定(快循环)"]
    print("\n=== 结论：盘中判定 vs 收盘判定 ===")
    print(f"  Δ年化 {(m1['ann']-m0['ann'])*100:+6.2f}pp | Δ回撤 {(m1['maxdd']-m0['maxdd'])*100:+6.2f}pp | "
          f"Δ换手 {r1['switches']-r0['switches']:+3d} | Δ落袋 {r1['peak_exits']-r0['peak_exits']:+3d} | "
          f"OOSΔ年化 {(mo1['ann']-mo0['ann'])*100:+6.2f}pp")


if __name__ == "__main__":
    main()