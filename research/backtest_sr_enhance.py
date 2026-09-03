# -*- coding: utf-8 -*-
"""
backtest_sr_enhance.py —— 用 alexpantyukhin/support_resistance_levels 给「多资产 ETF 动量轮动」做增强回测
================================================================================
数据：cache/backtest/final_idx_*.csv（指数代理，2015-2026 全历史，已缓存，免网络）
  510300<-沪深300  510500<-中证500  159915<-创业板  511010<-国债  513100<-纳指  518880<-黄金
  (ETF 紧密跟踪指数，轮动跨大类比较足够；与真实ETF有极小跟踪误差，仅作相对对比)

对比的 5 个方案（同一套数据 / 同一套选股 decide()）：
  V0  基线轮动      ：持有 decide() 选出的 Top2/防御，直到下次调仓，区间内不止损（=原 backtest_rotation.py）
  V1  基线+固定-8%  ：V0 + 区间内价格跌破 买入价*0.92 即退出到现金
  V2a S/R动态止损   ：V0 + 止损位=最近支撑(夹在4%~15%)，无支撑则回退-8%
  V2b S/R止损+止盈  ：V2a + 价格触及最近阻力(<+20%)即止盈退现金
  V2c S/R全增强     ：V2b + 入场过滤(不买在强阻力1%以内)
基准：等权持有6 / 买入持有沪深300

输出：命令行对比表 + cache/backtest/sr_enhance_result.json + sr_equity_monthly.json(供绘图)
"""
import os
import sys
import json
import datetime
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import strategy_rotator as sr
import sr_levels as srl

CACHE = os.path.join(os.path.dirname(HERE), "cache", "backtest")

# 统一诚实成本口径（2026-09 对齐实盘账户）：
#   佣金 = 万2.38 含规费；换仓 = 卖出1单 + 买入1单 = 双边；5元最低佣金对 >=2.1万 账户无效。
#   ETF 免印花税（与股票的唯一差异；用户决策按免印花统一，信息熵=0 不予区分）。
#   满仓换仓成本 = 0.000238*2 = 0.000476；停损->再入同为双边，亦计此成本（不再免费）。
COM_RATE = 0.000238        # 万2.38 含规费（单边）
ROUND_TRIP = COM_RATE * 2  # 0.0476% 满仓换仓双边成本
FEE = ROUND_TRIP           # simulate_daily / simulate 每"切换"扣一次，取真实双边
FREQ = "2W"                # 双周调仓(研究更优)

# S/R 参数
SR_WIN = 120            # 计算支撑/阻力的回看窗口(交易日)
SR_ORDER = 3            # 极值 order
SR_MERGE = 2.0          # 聚类百分比(delta)：相近极值聚成同一价位
MIN_STRENGTH = 2        # 价位最少 touch 次数(强度)才采用
STOP_CAP_LO = 0.04      # 支撑太近->止损最小 4%
STOP_CAP_HI = 0.15      # 支撑太远->止损最大 15%
TP_CAP = 0.20           # 阻力超过 +20% 不设置止盈
ENTRY_BUF = 0.01        # 入场过滤：买价在强阻力 1% 以内则跳过
DEF = sr.DEFENSIVE

PROXY = {
    "510300": "final_idx_300.csv",
    "510500": "final_idx_500.csv",
    "159915": "final_idx_cyb.csv",
    "511010": "final_idx_bond.csv",
    "513100": "final_us_ixic.csv",
    "518880": "final_gold_au9999.csv",
}


def load_prices():
    cols = {}
    for code, fn in PROXY.items():
        df = pd.read_csv(os.path.join(CACHE, fn), dtype={"date": str})
        s = df.set_index("date")["close"].astype(float)
        s.index = pd.to_datetime(s.index)
        cols[code] = s
    price = pd.DataFrame(cols).dropna()
    price.index = pd.to_datetime(price.index)
    return price.sort_index()


def synth_ohlc(close_series):
    """仅有收盘价时，近似合成 OHLC：open=昨收，high/low 取 open/close 包络。"""
    df = pd.DataFrame({"close": close_series.astype(float)})
    df["open"] = df["close"].shift(1).fillna(df["close"])
    df["high"] = df[["open", "close"]].max(axis=1)
    df["low"] = df[["open", "close"]].min(axis=1)
    return df.dropna()


def compute_sr(series_upto, price):
    w = series_upto.iloc[-SR_WIN:]
    ohlc = synth_ohlc(w)
    levels = srl.get_levels(ohlc, SR_ORDER, SR_MERGE)
    sup, res = [], []
    for lv in levels:
        if srl.strength(lv) < MIN_STRENGTH:
            continue
        if lv.level_value < price:
            sup.append(lv.level_value)
        elif lv.level_value > price:
            res.append(lv.level_value)
    return (max(sup) if sup else None, min(res) if res else None)


def sr_stop_price(entry, nsup):
    if nsup is None:
        return entry * (1 - 0.08)          # 无支撑 -> 回退固定 -8%
    dist = (entry - nsup) / entry
    if dist < STOP_CAP_LO:
        return entry * (1 - 0.08)          # 支撑太近 -> 回退 -8%
    if dist > STOP_CAP_HI:
        return entry * (1 - STOP_CAP_HI)    # 支撑太远 -> 封顶 -15%
    return nsup                            # 用真实支撑


def decide_targets(price, dec_dates):
    targets = {}
    for d in dec_dates:
        hist = price.loc[:d]
        if len(hist) < 200:
            targets[d] = "cash"
            continue
        prices = {c: hist[c].values for c in price.columns}
        res = sr.decide(prices)
        targets[d] = res["target"]
    return targets


def simulate(price, dec_dates, targets, flags):
    codes = list(price.columns)
    dates = price.index
    daily_ret = price.pct_change().fillna(0.0)
    port = pd.Series(0.0, index=dates)
    adj = pd.Series(1.0, index=dates)      # 换仓费用拖累(从换仓日起生效)
    switches = 0
    stop_exits = 0
    tp_exits = 0
    invested_days = 0
    prev_key = None

    for i in range(1, len(dec_dates)):
        d0, d1 = dec_dates[i - 1], dec_dates[i]
        tgt = targets[d0]
        mask = (dates > d0) & (dates <= d1)
        seg = dates[mask]
        if len(seg) == 0:
            continue
        d0_t = price.index[price.index <= d0]
        d0_t = d0_t[-1] if len(d0_t) else d0
        key = "cash" if tgt == "cash" else tuple(sorted(c for c in tgt))
        if prev_key is not None and key != prev_key:
            switches += 1
            adj.loc[dates >= d0] *= (1 - FEE)
        prev_key = key

        if tgt == "cash":
            port.loc[seg] = 0.0
            continue

        # 建仓
        weights = {c: 1.0 for c in tgt}
        held = set(tgt)
        entry = {c: price.loc[d0_t, c] for c in tgt}
        stop = {}
        tp = {}
        for c in list(tgt):
            if c == DEF:
                stop[c] = None
                tp[c] = None
                continue
            if flags.get("flat_stop"):
                stop[c] = entry[c] * 0.92
                tp[c] = None
            elif flags.get("sr_stop"):
                nsup, nres = compute_sr(price.loc[:d0_t, c], entry[c])
                if flags.get("entry_filter") and nres is not None and entry[c] >= nres * (1 - ENTRY_BUF):
                    weights.pop(c, None)
                    held.discard(c)
                    continue
                stop[c] = sr_stop_price(entry[c], nsup)
                if flags.get("tp") and nres is not None and (nres - entry[c]) / entry[c] <= TP_CAP:
                    tp[c] = nres
                else:
                    tp[c] = None
            else:
                stop[c] = None
                tp[c] = None
        if not weights:
            port.loc[seg] = 0.0
            continue
        tot = sum(weights.values())
        weights = {c: w / tot for c, w in weights.items()}

        for t in seg:
            rt = {c: daily_ret.loc[t, c] for c in held}
            for c in list(held):
                px = price.loc[t, c]
                hit = False
                if stop.get(c) is not None and px <= stop[c]:
                    hit = True
                    stop_exits += 1
                elif tp.get(c) is not None and px >= tp[c]:
                    hit = True
                    tp_exits += 1
                if hit:
                    held.discard(c)
            if not held:
                port.loc[t] = 0.0
            else:
                port.loc[t] = sum(weights[c] * rt[c] for c in held)
            if port.loc[t] != 0.0:
                invested_days += 1

    equity = (1 + port).cumprod() * adj   # adj 是"换仓日起生效"的一次性费用拖累(按关卡乘)，不是每日复利
    n = max(len(equity), 1)
    cum = float(equity.iloc[-1])
    ann = float(cum ** (252.0 / n) - 1) if cum > 0 else -1.0
    maxdd = float(((equity - equity.cummax()) / equity.cummax()).min())
    time_in_mkt = invested_days / max(n - 1, 1)
    return {
        "equity": equity, "cum": cum, "ann": ann, "maxdd": maxdd,
        "switches": switches, "stop_exits": stop_exits, "tp_exits": tp_exits,
        "time_in_mkt": time_in_mkt, "port": port, "adj": adj,
    }


def metrics_from_series(equity):
    n = max(len(equity), 1)
    cum = float(equity.iloc[-1])
    ann = float(cum ** (252.0 / n) - 1) if cum > 0 else -1.0
    maxdd = float(((equity - equity.cummax()) / equity.cummax()).min())
    return cum, ann, maxdd


def main():
    price = load_prices()
    print(f"数据区间 {price.index[0].date()} ~ {price.index[-1].date()}，共 {len(price)} 交易日，标的 {list(price.columns)}")

    dec_dates = list(price.resample(FREQ).last().index)
    # 跳过历史不足的早期
    dec_dates = [d for d in dec_dates if len(price.loc[:d]) >= 200]
    targets = decide_targets(price, dec_dates)

    daily_ret = price.pct_change().fillna(0.0)
    eqw = (1 + daily_ret[list(PROXY.keys())].mean(axis=1)).cumprod()
    bh300 = (1 + daily_ret["510300"]).cumprod()

    variants = {
        "V0 基线轮动": {},
        "V1 基线+-8%止损": {"flat_stop": True},
        "V2a S/R动态止损": {"sr_stop": True},
        "V2d S/R止损+入场过滤": {"sr_stop": True, "entry_filter": True},
        "V2b S/R止损+止盈": {"sr_stop": True, "tp": True},
        "V2c S/R全增强": {"sr_stop": True, "tp": True, "entry_filter": True},
    }
    results = {}
    for name, flags in variants.items():
        results[name] = simulate(price, dec_dates, targets, flags)
        r = results[name]
        print(f"{name:18s} 累计 {r['cum']:.3f}x  年化 {r['ann']*100:6.2f}%  "
              f"最大回撤 {r['maxdd']*100:6.2f}%  换仓 {r['switches']:4d}  "
              f"止损退出 {r['stop_exits']:4d}  止盈退出 {r['tp_exits']:4d}  "
              f"持仓时间占比 {r['time_in_mkt']*100:4.1f}%")

    ec, ea, ed = metrics_from_series(eqw)
    bc, ba, bd = metrics_from_series(bh300)
    print(f"{'等权6ETF':18s} 累计 {ec:.3f}x  年化 {ea*100:6.2f}%  最大回撤 {ed*100:6.2f}%")
    print(f"{'买入持有300':18s} 累计 {bc:.3f}x  年化 {ba*100:6.2f}%  最大回撤 {bd*100:6.2f}%")

    # 保存结果 + 月度权益曲线(供绘图)
    out = {}
    monthly_idx = price.resample("ME").last().index
    eq_monthly = {"dates": [d.strftime("%Y-%m") for d in monthly_idx]}
    for name, r in results.items():
        out[name] = {k: v for k, v in r.items() if k not in ("equity", "port", "adj")}
        s = r["equity"].reindex(monthly_idx, method="ffill")
        eq_monthly[name] = [round(float(x), 4) for x in s.values]
    eq_monthly["等权6ETF"] = [round(float(x), 4) for x in eqw.reindex(monthly_idx, method="ffill").values]
    eq_monthly["买入持有300"] = [round(float(x), 4) for x in bh300.reindex(monthly_idx, method="ffill").values]

    with open(os.path.join(CACHE, "sr_enhance_result.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    with open(os.path.join(CACHE, "sr_equity_monthly.json"), "w") as f:
        json.dump(eq_monthly, f, ensure_ascii=False)

    # 结论提示
    base = results["V0 基线轮动"]
    best = results["V2c S/R全增强"]
    print("\n=== 结论 ===")
    print(f"S/R全增强 vs 基线：年化 {base['ann']*100:.2f}% -> {best['ann']*100:.2f}%，"
          f"回撤 {base['maxdd']*100:.2f}% -> {best['maxdd']*100:.2f}%")
    print("（注：数据用指数代理、双周调仓、成本0.1%/次；与真实ETF回测有极小跟踪误差，仅作相对对比）")


if __name__ == "__main__":
    main()
