# -*- coding: utf-8 -*-
"""
backtest_sr_v3.py —— 在 V2d 基础上找「多赚少亏」的改进
================================================================================
当前装机基线 = V2d(S/R动态止损+入场过滤)：回测 2.763x / 年化11.94% / 回撤-12.64%。
本脚本测「尚未测试过」的方向（同一套数据/成本/双周口径）：
  1) V2d 复现基线（确保可比）
  2) V3x 跟踪止损：随价格创新高把止损上移(trail_pct 6/8/10/12%)，锁住利润、少回吐
  3) HOLD_N 敏感性(1/2/3)（叠加最优 trail）
并做样本外(2021+)校验，防止过拟合。结果写 cache/backtest/sr_v3_result.json。
"""
import os, sys, json
import datetime
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import backtest_sr_enhance as bse   # 复用数据/选股/S-R 工具

CACHE = os.path.join(os.path.dirname(HERE), "cache", "backtest")
FREQ = "2W"
SPLIT = pd.Timestamp("2021-01-01")

# 从 bse 借用常量（保证与 V2d 基线完全一致）
ENTRY_BUF = bse.ENTRY_BUF
DEF = bse.DEF
STOP_CAP_LO = bse.STOP_CAP_LO
STOP_CAP_HI = bse.STOP_CAP_HI


def decide_targets(price, dec_dates, hold_n=2):
    targets = {}
    for d in dec_dates:
        hist = price.loc[:d]
        if len(hist) < 200:
            targets[d] = "cash"
            continue
        prices = {c: hist[c].values for c in price.columns}
        res = bse.sr.decide(prices, {"HOLD_N": hold_n})
        targets[d] = res["target"]
    return targets


def simulate(price, dec_dates, targets, trail_pct=None, defn=DEF):
    """V2d 结构 + 可选跟踪止损(trail_pct)。defn 防御资产。"""
    codes = list(price.columns)
    dates = price.index
    daily_ret = price.pct_change().fillna(0.0)
    port = pd.Series(0.0, index=dates)
    adj = pd.Series(1.0, index=dates)
    switches = 0
    stop_exits = 0
    invested_days = 0
    prev_key = None

    for i in range(1, len(dec_dates)):
        d0, d1 = dec_dates[i - 1], dec_dates[i]
        tgt = targets[d0]
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

        weights = {c: 1.0 for c in tgt}
        held = set(tgt)
        entry = {c: price.loc[d0_t, c] for c in tgt}
        stop, peak, tp = {}, {}, {}
        for c in list(tgt):
            if c == defn:
                stop[c] = None
                continue
            nsup, nres = bse.compute_sr(price.loc[:d0_t, c], entry[c])
            # 入场过滤：贴近阻力不进
            if nres is not None and entry[c] >= nres * (1 - ENTRY_BUF):
                weights.pop(c, None)
                held.discard(c)
                continue
            stop[c] = bse.sr_stop_price(entry[c], nsup)
            peak[c] = entry[c]
        if not weights:
            port.loc[seg] = 0.0
            continue
        tot = sum(weights.values())
        weights = {c: w / tot for c, w in weights.items()}

        for t in seg:
            rt = {c: daily_ret.loc[t, c] for c in held}
            for c in list(held):
                px = price.loc[t, c]
                # 跟踪止损：随创新高上移，只升不降
                if peak.get(c) is not None and px > peak[c]:
                    peak[c] = px
                    ts = peak[c] * (1 - trail_pct) if trail_pct else None
                    if ts is not None and (stop[c] is None or ts > stop[c]):
                        stop[c] = ts
                if stop.get(c) is not None and px <= stop[c]:
                    held.discard(c)
                    stop_exits += 1
            if not held:
                port.loc[t] = 0.0
            else:
                port.loc[t] = sum(weights[c] * rt[c] for c in held)
            if port.loc[t] != 0.0:
                invested_days += 1

    equity = (1 + port).cumprod() * adj
    n = max(len(equity), 1)
    cum = float(equity.iloc[-1])
    ann = float(cum ** (252.0 / n) - 1) if cum > 0 else -1.0
    maxdd = float(((equity - equity.cummax()) / equity.cummax()).min())
    return {"equity": equity, "cum": cum, "ann": ann, "maxdd": maxdd,
            "switches": switches, "stop_exits": stop_exits, "port": port, "adj": adj}


def report(price, dec_dates, targets, label):
    r = simulate(price, dec_dates, targets, trail_pct=None)
    print(f"{label:26s} 累计{r['cum']:6.3f}x 年化{r['ann']*100:6.2f}% "
          f"回撤{r['maxdd']*100:6.2f}% 止损{r['stop_exits']:3d}")
    return r


def slice_metrics(result, start):
    """对已算好的全期 result 的权益曲线取 start 起的样本外指标。"""
    eq = result["equity"].loc[start:]
    if len(eq) < 2:
        return None
    n = max(len(eq), 1)
    cum = float(eq.iloc[-1])
    ann = float(cum ** (252.0 / n) - 1) if cum > 0 else -1.0
    maxdd = float(((eq - eq.cummax()) / eq.cummax()).min())
    return ann, maxdd


def main():
    price = bse.load_prices()
    print(f"数据 {price.index[0].date()} ~ {price.index[-1].date()}，{len(price)} 交易日\n")

    dec_dates = [d for d in price.resample(FREQ).last().index if len(price.loc[:d]) >= 200]

    # —— 1) V2d 复现基线 ——
    tg2 = decide_targets(price, dec_dates, 2)
    base = report(price, dec_dates, tg2, "1) V2d 基线(HOLD_N=2)")
    expect_ann = 11.94
    print(f"   (复现目标≈年化{expect_ann}%，偏差{abs(base['ann']*100-expect_ann):.2f}pp)\n")

    # —— 2) 跟踪止损 sweep（HOLD_N=2）——
    print("2) 跟踪止损 sweep（HOLD_N=2）：")
    rows = {}
    for tp in [0.06, 0.08, 0.10, 0.12]:
        r = simulate(price, dec_dates, tg2, trail_pct=tp)
        rows[f"V3 trail{tp*100:.0f}%"] = r
        print(f"   trail {tp*100:.0f}% 累计{r['cum']:6.3f}x 年化{r['ann']*100:6.2f}% "
              f"回撤{r['maxdd']*100:6.2f}% 止损{r['stop_exits']:3d}")
    best_trail = max(rows, key=lambda k: rows[k]["ann"])
    combos = {**{"V2d基线": base}, **rows}

    # —— 3) HOLD_N 敏感性（叠加最优 trail） ——
    print("\n3) HOLD_N 敏感性（叠加最优 trail=" + best_trail[-4:].replace("%", "") + "%）：")
    best_tp = float(best_trail.split("trail")[1].rstrip("%")) / 100
    for hn in [1, 3]:
        tg = decide_targets(price, dec_dates, hn)
        r = simulate(price, dec_dates, tg, trail_pct=best_tp)
        combos[f"HOLD_N={hn}+trail{best_tp*100:.0f}%"] = r
        print(f"   HOLD_N={hn} 累计{r['cum']:6.3f}x 年化{r['ann']*100:6.2f}% "
              f"回撤{r['maxdd']*100:6.2f}% 止损{r['stop_exits']:3d}")

    # —— 4) 样本外校验（2021+）：V2d vs 组合方案 ——
    print("\n4) 样本外(2021+)校验：")
    combos2 = {"V2d基线": base, **rows}
    combos2.update({k: v for k, v in combos.items() if "HOLD_N" in k})
    for name, r in combos2.items():
        so = slice_metrics(r, SPLIT)
        if so:
            print(f"   {name:24s} 样本外: 年化{so[0]*100:6.2f}% 回撤{so[1]*100:6.2f}%")
        else:
            print(f"   {name:24s} 样本外: 数据不足")

    # 输出
    out = {}
    for name, r in combos.items():
        out[name] = {k: (v.tolist() if hasattr(v, "tolist") else v)
                     for k, v in r.items() if k not in ("equity", "port", "adj")}
    out["split"] = str(SPLIT.date())
    with open(os.path.join(CACHE, "sr_v3_result.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n结果已写入 cache/backtest/sr_v3_result.json")


if __name__ == "__main__":
    main()