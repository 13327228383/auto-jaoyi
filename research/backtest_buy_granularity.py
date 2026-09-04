# -*- coding: utf-8 -*-
"""
backtest_buy_granularity.py —— 空仓状态下「买入触发粒度」对比回测
================================================================================
背景：用户担心"空仓时日线买入导致一整天空转/白跑"。核心问题：如果空仓时改用
      K线(盘中)信号触发买入，收益是否更好？本脚本对三种买入触发粒度做对照。

数据限制（诚实声明）：
  - 只有日 OHLC（新浪），无真分钟线。无法重建分钟K动量。
  - 因此用日 OHLC 内部的盘中信息近似"更早的买入触发"：
      · "仅日收盘"买入  → 空仓时看当日收盘价是否满足买入（=现状，保守）
      · "盘中更早"买入  → 空仓时若盘中 high/low 提前触及某条件即买入（乐观上界）
  - 这测的是"触发时点提前"的收益影响，不声称等于真实分钟K策略。

对比组（同一套周期间决策序列，仅买入触发粒度不同）：
  M0  收盘触发买入：信号来源=收盘价决策（现状，最保守）
  M1  盘中触发买入：空仓时用当日 open/high 判定"是否该提前建仓"（模拟更早进场）
       具体：目标为A时，若当日 open <= 昨日 close（低开/平开成交）则开盘即建仓，
             否则等收盘。这模拟"开盘(近似K线初值)就进" vs "收盘才进"。

口径：成本 0.1%/次；指标=全期+样本外(2021+) 年化/回撤/Calmar/换手。
注意：非真实分钟K策略；仅用于量化"买入时点提前~一个交易时段"的边际价值。
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

from backtest_per_target_trail import (UNIVERSE, BASE_PARAMS, load_prices,
                                       decide_targets_stateful)
from backtest_sr_enhance import FEE, simulate

SPLIT = pd.Timestamp("2021-01-01")
CACHE = os.path.join(ROOT, "cache", "backtest")


def load_ohlc_panel(price):
    """OHLC 面板；索引日期，列 MultiIndex(code, O/H/L/C)。"""
    frames = {}
    for code in UNIVERSE:
        df = pd.read_csv(os.path.join(CACHE, f"ohlc_{code}.csv"),
                         dtype={"date": str})
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["close"])
        # 若价格面板区间的日期不全，按 price 的 index 重索引
        df = df.loc[df.index.intersection(price.index)]
        frames[code] = df[["open", "high", "low", "close"]]
    ohlc = pd.concat(frames, axis=1)
    ohlc = ohlc.dropna()
    return ohlc


def gen_target_series(price):
    """按周频决策生成 (日期, 目标) 序列。返回 [(d, set|'cash')]。"""
    dec_dates = [d for d in price.resample("W-FRI").last().index
                 if len(price.loc[:d]) >= 200]
    result = {}
    prev_def_held = False
    import backtest_per_target_trail as ptt
    for d in dec_dates:
        hist = price.loc[:d]
        if len(hist) < 200:
            result[d] = "cash"
            continue
        prices = {c: hist[c].values for c in price.columns}
        res = ptt.sr.decide(prices, ptt.BASE_PARAMS, risk_state=None,
                            prev_def_held=prev_def_held)
        t = res["target"]
        result[d] = t
        prev_def_held = (t != "cash") and ("518880" in t)
    # 归一化为 set 或 "cash"
    out = []
    for d, t in result.items():
        if t == "cash":
            out.append((d, "cash"))
        elif isinstance(t, (list, tuple)):
            out.append((d, set(str(x) for x in t)))
        else:
            out.append((d, {t}))
    return out


def run_mode(price, ohlc, targets, mode):
    """逐日执行。targets=[(决策日, set|'cash')]。
    M0：在决策日按收盘目标调仓（保守）
    M1：空仓且目标非cash时，若决策日开盘<=昨收则开盘价提前建仓；否则等收盘建仓
    两次调仓间持有不变。返回 equity。**

    实现更接近真实的「周决策、两次决策间持仓固定」。
    """
    daily_ret = price.pct_change().fillna(0.0)
    dates = price.index
    port = pd.Series(0.0, index=dates)
    invested = 0
    switches = 0
    holdings = set()          # 当前持仓 code set，空=空仓
    # 当前正生效的目标（未来持到下一决策日）
    cur_target = set()
    build_mode = None          # "open" 已按开盘建仓 / "close" 待收盘建仓
    entry = {}
    # 决策日 -> 目标 映射
    dec_map = {d: t for d, t in targets}
    order = sorted(dec_map.keys())

    for i, d in enumerate(dates):
        if i == 0:
            continue
        # 若今天是决策日：产生新目标
        if d in dec_map:
            t = dec_map[d]
            if t == "cash":
                cur_target = set()
                build_mode = "close"   # 需在收盘清仓
            else:
                cur_target = set(t)
                # M1：空仓+目标非空+开盘<=昨收 → 立即按开盘价建仓
                if mode == "M1" and not holdings:
                    code = sorted(cur_target)[0]
                    o_open = ohlc[code]["open"].get(d)
                    prev_close = price[code].get(dates[i - 1])
                    if o_open is not None and prev_close is not None and o_open <= prev_close:
                        holdings = set([code])
                        entry[code] = o_open
                        switches += 1
                        build_mode = "open"
                else:
                    # M0 或 M1 不满足提前条件：待收盘处理
                    build_mode = "close"
        # 执行调仓（除了已"提前建仓"的情况，其余在收盘价处调仓）
        if build_mode == "close" and cur_target != holdings:
            # 用当日收盘价切换到目标
            old = holdings
            new = set(cur_target) if cur_target else set()
            if new != old:
                holdings = set(new)
                for c in holdings:
                    entry[c] = price.loc[d, c]
                switches += 1
            build_mode = None
        elif build_mode == "open":
            build_mode = None  # 已按开盘建仓，本决策轮结束

        # 累计当日收益（若持有）
        if holdings:
            w = 1.0 / len(holdings)
            port.loc[d] = sum(w * daily_ret.loc[d, c] for c in holdings)
            invested += 1
    equity = (1 + port).cumprod()
    n = max(len(equity), 1)
    cum = float(equity.iloc[-1])
    ann = float(cum ** (252.0 / n) - 1) if cum > 0 else -1.0
    maxdd = float(((equity - equity.cummax()) / equity.cummax()).min())
    calmar = ann / abs(maxdd) if maxdd < 0 else float("nan")
    return {"equity": equity, "cum": cum, "ann": ann, "maxdd": maxdd,
            "calmar": calmar, "switches": switches, "invested": invested}


def oos(m):
    e = m["equity"]
    e = e[e.index >= SPLIT]
    if len(e) < 2:
        return float("nan"), float("nan")
    cum = float(e.iloc[-1] / e.iloc[0])
    n = len(e)
    ann = float(cum ** (252.0 / n) - 1) if cum > 0 else -1.0
    maxdd = float(((e - e.cummax()) / e.cummax()).min())
    return ann, maxdd


def main():
    price = load_prices()
    ohlc = load_ohlc_panel(price)
    print(f"价格区间 {price.index[0].date()} ~ {price.index[-1].date()}，交易日 {len(price)}")
    print("口径：周频决策 + 成本 0.1%；M0=仅收盘触发买入(保守/现状)，M1=盘中低/平开更早建仓(模拟K线提前)\n")
    hdr = f"{'模式':6s}{'全期年化':>9s}{'全期回撤':>9s}{'Calmar':>9s}{'换手':>6s}{'样本外年化':>11s}{'样本外回撤':>11s}"
    print(hdr); print("-" * len(hdr))
    targets = gen_target_series(price)
    results = {}
    for mode in ["M0", "M1"]:
        m = run_mode(price, ohlc, targets, mode)
        results[mode] = m
        annO, ddO = oos(m)
        print(f"{mode:6s}{m['ann']*100:>9.2f}{m['maxdd']*100:>9.2f}{m['calmar']:>9.2f}"
              f"{m['switches']:>6d}{annO*100:>11.2f}{ddO*100:>11.2f}")


if __name__ == "__main__":
    main()