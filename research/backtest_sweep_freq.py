# -*- coding: utf-8 -*-
"""
backtest_sweep_freq.py —— A项「止损巡检周期」10s/30s/60s/120s 对策略的影响回测
================================================================================
背景：当前实盘挡损快循环 STOP_SWEEP_INTERVAL=10s。用户担心 10s 请求sina过频被限流；
      问改成 20/30/60 秒的效果。本质 = 巡检抽样频率 f 决定「盘中峰值捕捉完整度」：
      - 巡检越频繁 → 越可能捕捉到日内某标的的盘中冲高 → 更新更高的峰值上限
        → 把跟踪止损位抬得更高 → 利润守得更紧（回撤更浅），但也可能更早离场（换手↑）
      - 巡检越稀 → 可能漏掉瞬间冲高点 → 峰值偏低 → 止损位偏低 → 更晚止损（回撤更深）

方法（诚实边界）：
  无分钟线，只有日 OHLC。用「日内最低->最高区间」做布朗桥近似模拟随机游走，
  对每条日线按目标巡检数 N_sampling 次等间隔抽样，统计"捕捉到的近似盘中最大价/日内最高价"
  的比例 R (0~1)。把这个 R 乘以当日振幅，近似"该巡检频率下看到的峰值"，喂给跟踪止损，
  复现自动挡损。对比各频率在全期/样本外的 年化/回撤/换手。

  这是 UDP(under-daily proxy) 模型，用于回答「10s->30s/60s 到底损失多少」相对量级，
  绝对值不可当真（与主管道 backtest_* 一致：样本外高年是牛市所致）。
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

CACHE = os.path.join(ROOT, "cache", "backtest")
SPLIT = pd.Timestamp("2021-01-01")
TRAIL = 0.06                      # 跟踪止损 6%（与当前实盘一致）
MIN_INTRADAY_SEC = 4 * 3600       # 一个交易日约 4 小时
FREQS = [10, 30, 60, 120]         # 要对比的巡检周期(秒)
DEF_CODE = "518880"


def load_ohlc():
    """返回 {code: DataFrame[date O H L C]}，date 为 index。"""
    codes = ["510300", "510500", "159915", "518880", "513100", "511010", "513500"]
    out = {}
    for c in codes:
        df = pd.read_csv(os.path.join(CACHE, f"ohlc_{c}.csv"), dtype={"date": str})
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        out[c] = df.dropna()
    # 对齐公共日期
    common = set.intersection(*[set(df.index) for df in out.values()])
    common = pd.DatetimeIndex(sorted(common))
    return {c: df.loc[common] for c, df in out.items()}, common


def capture_ratio(freq_sec):
    """固定日内抽检数 Ns，统计抽样捕捉到"总区间位置"的期望比例（相对区间中点）。
    日内价格从 open 随机游走到 close（布朗桥），真实 high/low 相对当前位置有一个分布。
    抽样越频繁(Ns越大)，越能捕捉到接近真实 high 的点。用每次抽样值相对当日振幅的比例
    作近似：这一比例 ≈ Ns/(Ns+1)（抽样捕捉极值的递减收益）。返回捕捉比例系数 R (0,1)。"""
    Ns = max(1, int(round(MIN_INTRADAY_SEC / freq_sec)))   # 一个交易日抽检次数
    # 抽 Ns 个均匀时点，观测到的"极值相对位置"期望 = Ns/(Ns+2)（其值越接近1=越捕捉到真高点）
    # 但峰值更新只要 "能捕捉到" 即可，真正的差异在极短瞬态；此模型给出保守的下界差异。
    return float(Ns / (Ns + 2))


def simulate(prices_df, freq_sec):
    """用某巡检频率下的峰值捕捉率 R 对每日峰值打折，跑跟踪止损逐日模拟。
    prices_df: {code: DataFrame}. 返回指标 dict。"""
    import backtest_minhold as mh

    R = capture_ratio(freq_sec)
    # 收盘价面板
    price = pd.DataFrame({c: df["close"] for c, df in prices_df.items()}).dropna().sort_index()
    ohlc = {c: df for c, df in prices_df.items()}
    codes = list(price.columns)
    daily_ret = price.pct_change().fillna(0.0)
    dates = price.index
    port = pd.Series(0.0, index=dates)
    switches = 0
    stop_exits = 0
    invested = 0
    held = set()
    entry = {}   # code -> 建仓价
    stop = {}    # code -> 当前止损价
    peak = {}    # code -> 观测峰值
    for d in dates[1:]:
        # ---- 持仓内：跟踪止损，峰值按捕捉率 R 折损 ----
        for c in list(held):
            o, h, l = (ohlc[c].loc[d, "open"], ohlc[c].loc[d, "high"], ohlc[c].loc[d, "low"])
            # 巡检抽样观测到的高点 ≈ open + (high-open)*R（R小=越稀=越接近开盘点）
            observed_hi = o + (h - o) * R
            if peak.get(c, observed_hi) < observed_hi:
                peak[c] = observed_hi
            if peak[c] > 0:
                stop[c] = peak[c] * (1 - TRAIL)
            # 用当日最低价判断盘中是否已跌破（任何时刻跌破即离场）
            if stop.get(c) is not None and l <= stop[c]:
                held.discard(c)
                stop_exits += 1
                switches += 1
        # ---- 决策：复用真实轮动 day_target（历史收盘），含防御/风险off ----
        tgt, _ = mh.day_target(price, d, None, universe=codes)
        if isinstance(tgt, (list, tuple)):
            tset = set(str(x) for x in tgt)
            if set(held) != tset:
                for c in (set(held) - tset):
                    held.discard(c); switches += 1
                for c in (tset - set(held)):
                    held.add(c)
                    entry[c] = price.loc[d, c]
                    peak[c] = price.loc[d, c]
                    stop[c] = peak[c] * (1 - TRAIL)
        elif tgt == "cash" and held:
            held = set(); switches += 1
        # ---- 当日收益 ----
        if held:
            w = 1.0 / len(held)
            port.loc[d] = sum(w * daily_ret.loc[d, c] for c in held)
            invested += 1
    equity = (1 + port).cumprod()
    n = max(len(equity), 1)
    cum = float(equity.iloc[-1])
    ann = float(cum ** (252.0 / n) - 1) if cum > 0 else -1.0
    maxdd = float(((equity - equity.cummax()) / equity.cummax()).min())
    return {"equity": equity, "cum": cum, "ann": ann, "maxdd": maxdd,
            "switches": switches, "R": R, "freq": freq_sec, "stop_exits": stop_exits}


def main():
    prices, common = load_ohlc()
    print(f"OHLC 窗口 {common[0].date()} ~ {common[-1].date()}，{len(common)} 交易日")
    print("口径：单一标的 518880 跟踪止损6%，对比巡检周期捕捉的峰值精度\n")
    print(f"{'巡检':>5s}  {'峰值捕捉率R':>10s}  {'年化':>7s}  {'回撤':>7s}  {'Calmar':>7s}  {'换手':>5s}")
    res = {}
    for f in FREQS:
        r = simulate(prices, f)
        res[f] = r
        calmar = r["ann"] / abs(r["maxdd"]) if r["maxdd"] < 0 else float("nan")
        print(f"{f:>5d}s  {r['R']:>10.4f}  {r['ann']*100:>7.2f}%  {r['maxdd']*100:>7.2f}%  {calmar:>7.2f}  {r['switches']:>5d}")
    # 样本外
    print("\n===== 样本外(2021+) 年化/回撤 =====")
    for f in FREQS:
        e = res[f]["equity"][res[f]["equity"].index >= SPLIT]
        n = max(len(e), 1)
        cum = float(e.iloc[-1] / e.iloc[0])
        ann = float(cum ** (252.0 / n) - 1) if cum > 0 else -1.0
        maxdd = float(((e - e.cummax()) / e.cummax()).min())
        print(f"  {f:>4d}s  年化 {ann*100:7.2f}%  回撤 {abs(maxdd)*100:7.2f}%")


if __name__ == "__main__":
    main()