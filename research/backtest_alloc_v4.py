# -*- coding: utf-8 -*-
"""
backtest_alloc_v4.py —— 购买比例（目标内权重分配）两个方向打分回测 + 特征值
================================================================================
【注意】本脚本是早期 HOLD_N=2(同时持两只) 时代研究购买比例用的历史回测，仅供权重方法参考；
现实盘已落地 HOLD_N=1(只持一只最强)，此时不再有"标的内分配"，mom2/mom/等权退化为单只=100%。
下方背景/对比均针对 HOLD_N=2 的历史口径，勿据此误读当前策略目标为"两只"。

背景(HOLD_N=2 历史口径)：资金在两只目标 ETF 之间如何分配？这决定"买入比例"。
本脚本在【同一套保护】(V2d S/R动态止损 + 入场过滤 + 跟踪止损6%，继承 backtest_sr_v3 最优)
下对比 4 种权重分配，覆盖"两个方向"：
  · 收益/趋势方向（把更多权重给更强趋势的资产）：
      equal   等权         （基线，目前实盘用）
      mom     动量加权      （权重∝动量分）
      mom2    动量平方加权   （更激进地追强趋势）
  · 风险/波动方向（把更多权重给更稳的资产，逆波动率）：
      invol   逆波动率加权  （权重∝1/σ）

特征值（为后续实盘与市场动态判断）：
  累计(倍数) / 年化CAGR / 最大回撤 / Calmar / Sharpe / 年化波动 / 换仓次数 /
  止损次数 / 各资产在持期间平均权重（看组合如何倾斜，识别市场偏好的资产）。

输出：命令行对比表 + cache/backtest/alloc_v4_result.json
"""
import os, sys, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import backtest_sr_enhance as bse

CACHE = os.path.join(os.path.dirname(HERE), "cache", "backtest")
FREQ = "2W"
SPLIT = pd.Timestamp("2021-01-01")        # 样本外切点
TRAIL_PCT = 0.06                          # 跟踪止损 6%（v3 实测最优）
ALLOCS = ["equal", "mom", "mom2", "invol"]  # 权重分配方案

ENTRY_BUF = bse.ENTRY_BUF
DEF = bse.DEF


def decide_targets(price, dec_dates, hold_n=2):
    """返回 {date: (target, scores)}。复用 bse.sr.decide 拿每标的分。"""
    out = {}
    for d in dec_dates:
        hist = price.loc[:d]
        if len(hist) < 200:
            out[d] = ("cash", {})
            continue
        prices = {c: hist[c].values for c in price.columns}
        res = bse.sr.decide(prices, {"HOLD_N": hold_n})
        out[d] = (res["target"], res.get("scores", {}))
    return out


def _weights(alloc, tgt, scores, daily_ret, days):
    """算目标内权重，返回 dict（未归一化也可）。异常回退等权。"""
    if alloc == "equal":
        return {c: 1.0 for c in tgt}
    if alloc in ("mom", "mom2"):
        p = 1 if alloc == "mom" else 2
        s = {c: max(float(scores.get(c, 0) or 0), 0.0) ** p for c in tgt}
        tot = sum(s.values())
        if tot <= 1e-12:
            return {c: 1.0 for c in tgt}
        return {c: s[c] / tot for c in tgt}
    if alloc == "invol":
        w = {}
        for c in tgt:
            r = daily_ret[c].loc[days].dropna()
            v = float(r.std()) if len(r) >= 5 and r.std() > 0 else None
            w[c] = 1.0 / v if v else 1.0
        tot = sum(w.values())
        return {c: w[c] / tot for c in tgt}
    return {c: 1.0 for c in tgt}


def simulate(price, dec_dates, targets, alloc, trail_pct=TRAIL_PCT):
    """与 backtest_sr_v3.simulate 结构一致的逐区间模拟；仅权重分配可插拔。"""
    dates = price.index
    daily_ret = price.pct_change().fillna(0.0)
    port = pd.Series(0.0, index=dates)
    adj = pd.Series(1.0, index=dates)      # 换仓费用拖累
    switches = 0
    stop_exits = 0
    invested_days = 0
    wsum = {c: 0.0 for c in price.columns}   # 每资产累计权重
    inmarket = {c: 0 for c in price.columns} # 每资产在持天数
    prev_key = None

    for i in range(1, len(dec_dates)):
        d0, d1 = dec_dates[i - 1], dec_dates[i]
        tgt, scores = targets[d0]
        mask = (dates > d0) & (dates <= d1)
        seg = dates[mask]
        if len(seg) == 0:
            continue
        d0_t = price.index[price.index <= d0][-1]
        key = "cash" if tgt == "cash" else tuple(sorted(c for c in tgt))
        if prev_key is not None and key != prev_key:
            switches += 1
            adj.loc[dates >= d0] *= (1 - bse.FEE)
        prev_key = key

        if tgt == "cash":
            port.loc[seg] = 0.0
            continue

        # 历史天数（invol 用已发生数据算 σ，避免未来函数）
        hist_days = dates[dates <= d0_t]
        weights = dict(_weights(alloc, tgt, scores, daily_ret, hist_days))
        held = set(tgt)
        entry = {c: price.loc[d0_t, c] for c in tgt}
        stop, peak = {}, {}
        for c in list(tgt):
            if c == DEF:
                stop[c] = None
                continue
            nsup, nres = bse.compute_sr(price.loc[:d0_t, c], entry[c])
            # 入场过滤：贴近强阻力不进
            if nres is not None and entry[c] >= nres * (1 - ENTRY_BUF):
                weights.pop(c, None)
                held.discard(c)
                continue
            stop[c] = bse.sr_stop_price(entry[c], nsup)
            peak[c] = entry[c]
        if not held:
            port.loc[seg] = 0.0
            continue
        tot = sum(weights[c] for c in held)
        weights = {c: weights[c] / tot for c in held}

        for t in seg:
            rt = {c: daily_ret.loc[t, c] for c in held}
            for c in list(held):
                px = price.loc[t, c]
                # 跟踪止损：随创新高只升不降
                if peak.get(c) is not None and px > peak[c]:
                    peak[c] = px
                    ts = peak[c] * (1 - trail_pct)
                    if stop.get(c) is None or ts > stop[c]:
                        stop[c] = ts
                if stop.get(c) is not None and px <= stop[c]:
                    held.discard(c)
                    stop_exits += 1
            if not held:
                port.loc[t] = 0.0
            else:
                port.loc[t] = sum(weights[c] * rt[c] for c in held)
                for c in held:
                    wsum[c] += weights[c]
                    inmarket[c] += 1
            if port.loc[t] != 0.0:
                invested_days += 1

    equity = (1 + port).cumprod() * adj
    n = max(len(equity), 1)
    cum = float(equity.iloc[-1])
    ann = float(cum ** (252.0 / n) - 1) if cum > 0 else -1.0
    maxdd = float(((equity - equity.cummax()) / equity.cummax()).min())

    def feat(eq):
        eq = eq.dropna()
        r = eq.pct_change().dropna()
        sd = float(r.std())
        sharpe = float(r.mean() / sd * np.sqrt(252)) if sd > 0 else float("nan")
        vol = sd * np.sqrt(252)
        return sharpe, vol

    sharpe, vol = feat(equity)
    calmar = ann / abs(maxdd) if maxdd < 0 else float("nan")
    avg_w = {c: (wsum[c] / inmarket[c] if inmarket[c] else 0.0) for c in price.columns}

    return {"equity": equity, "cum": cum, "ann": ann, "maxdd": maxdd,
            "sharpe": sharpe, "vol": vol, "calmar": calmar,
            "switches": switches, "stop_exits": stop_exits,
            "avg_w": avg_w, "port": port, "adj": adj,
            "time_in_mkt": invested_days / max(n - 1, 1)}


def slice_metrics(r, start):
    eq = r["equity"].loc[start:]
    if len(eq) < 2:
        return None
    n = max(len(eq), 1)
    cum = float(eq.iloc[-1])
    ann = float(cum ** (252.0 / n) - 1) if cum > 0 else -1.0
    maxdd = float(((eq - eq.cummax()) / eq.cummax()).min())
    rr = eq.pct_change().dropna()
    sd = float(rr.std())
    sharpe = float(rr.mean() / sd * np.sqrt(252)) if sd > 0 else float("nan")
    calmar = ann / abs(maxdd) if maxdd < 0 else float("nan")
    return ann, maxdd, sharpe, calmar


def main():
    price = bse.load_prices()
    codes = list(price.columns)
    print(f"数据 {price.index[0].date()} ~ {price.index[-1].date()}，{len(price)} 交易日，标的 {codes}\n")
    dec_dates = [d for d in price.resample(FREQ).last().index if len(price.loc[:d]) >= 200]
    targets = decide_targets(price, dec_dates)

    print("购买比例（权重分配）对比  |  保护：S/R动态止损+入场过滤+跟踪止损6%\n")
    print(f"{'方案':28s}{'累计x':>7}{'年化':>7}{'回撤':>7}{'Calmar':>7}{'Sharpe':>7}"
          f"{'年化波动':>8}{'换仓':>5}{'止损':>5}{'持仓%':>7}")
    rows = {}
    for alloc in ALLOCS:
        r = simulate(price, dec_dates, targets, alloc)
        rows[alloc] = r
        name = {"equal": "equal 等权(基线)", "mom": "mom 动量加权+",
                "mom2": "mom2 动量平方加权+", "invol": "invol 逆波动率+"}[alloc]
        print(f"{name:28s}{r['cum']:>7.3f}{r['ann']*100:>7.2f}{r['maxdd']*100:>7.2f}"
              f"{r['calmar']:>7.2f}{r['sharpe']:>7.2f}{r['vol']*100:>8.2f}"
              f"{r['switches']:>5d}{r['stop_exits']:>5d}{r['time_in_mkt']*100:>7.1f}")

        # 平均权重（特征值：看组合向哪些资产倾斜）
        wstr = "  ".join(f"{c}:{r['avg_w'][c]*100:.0f}%" for c in codes if r['avg_w'][c] > 0)
        print(f"{'':28s}  平均权重>> {wstr}")

    print("\n样本外(2021+)校验：")
    for alloc in ALLOCS:
        so = slice_metrics(rows[alloc], SPLIT)
        if so:
            print(f"   {alloc:8s} 年化{so[0]*100:6.2f}% 回撤{so[1]*100:6.2f}% "
                  f"Sharpe{so[2]:5.2f} Calmar{so[3]:5.2f}")
        else:
            print(f"   {alloc:8s} 样本外数据不足")

    out = {}
    for alloc, r in rows.items():
        out[alloc] = {k: (v.tolist() if hasattr(v, "tolist") else v)
                      for k, v in r.items() if k not in ("equity", "port", "adj")}
    out["split"] = str(SPLIT.date())
    with open(os.path.join(CACHE, "alloc_v4_result.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n结果已写入 cache/backtest/alloc_v4_result.json")


if __name__ == "__main__":
    main()