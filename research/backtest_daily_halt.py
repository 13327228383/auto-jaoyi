# -*- coding: utf-8 -*-
"""
backtest_daily_halt.py —— 单日组合亏损熔断：「触发时机」对照回测
================================================================================
用户问题：策略只买一只 ETF（单标），「当日组合亏损 -3% → 转防御」的熔断，
到底该「维持 14:50 尾盘判」还是「提前到全天盘中触发」？哪个更优？

三组对照（口径与当前实盘策略严格一致，复用 day_target/bse/bue 同批原语）：
  base       : 无当日熔断（现状无熔断层 baseline）
  halt_close : 当日组合收益 <= -3%（用收盘价判，≈实盘 14:50）→ 当日转防御
  halt_intra : 当日持仓标的最坏盘中触及 -3%（用当日最低价近似）→ 当日即转防御

熔断动作（对齐 auto_run.daily_loss_halt）：触发当日把目标强制为防御资产 518880(黄金)，
次日重置后恢复正常 day_target 决策（熔断仅作用于触发当日，不跨日）。

方法学边界（诚实声明）：
  - 日线粒度下，"盘中 vs 尾盘"的差异只体现在「当日 -3% 之后/触及当日的剩余时段」。
  - 用「当日最低价」近似「盘中触及 -3%」，有一定近似误差；close 熔断近似为「当日按原持仓走完、
    动作落于收盘」→ 在日线下几乎不会产生当日避险收益（这正是要展示的信息点）。
  - 转防御仅按防御资产当日收盘收益结算触发当日，不模拟日内分时路径。
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import backtest_minhold as mh          # day_target（单一决策逻辑源）
import backtest_sr_enhance as bse      # compute_sr / sr_stop_price / FEE / ENTRY_BUF
import backtest_universe_ext as bue    # 真实ETF数据 + GOLD 配置
import data_center as dc

MIN_HOLD = 1
DEF_TRAIL = None
DEF_MOM = 10
TRAIL_PCT = 0.06
MAX_DLY_LOSS = 0.03          # 单日组合亏损熔断阈值 -3%（与 auto_run MAX_DAILY_LOSS_PCT 一致）
CURRENT_UNI = bue.BASE_CODES + ["513500"]
DEF_CODE = str(bse.DEF)


def load_prices():
    """7 只真实ETF收盘价（当前实盘标的池）。"""
    p = bue.load_prices()
    return p.reindex(columns=CURRENT_UNI).dropna().sort_index()


def load_lows(prices):
    """拉取各标的『最低价』(日K OHLC)，对齐收盘价日期；优先新浪JSON，失败退化为当日收盘(即收盘判定)。"""
    import json
    from urllib.request import Request, urlopen

    lows = {}
    for code in CURRENT_UNI:
        sym = bue.ALL_CODES.get(code)
        cf = os.path.join(bue.CACHE, f"etf_low_{code}.csv")
        s = None
        if os.path.exists(cf):                       # 本地缓存
            try:
                x = pd.read_csv(cf, dtype={"date": str}).set_index("date")["low"]
                if len(x) > 300:
                    s = x
            except Exception:
                pass
        if s is None and sym:
            try:                                     # akshare 拉日K OHLC（含 low），缓存
                import akshare as ak
                import time as _t
                for _ in range(3):                   # 网络偶断，重试
                    try:
                        df = ak.fund_etf_hist_sina(symbol=sym)
                        df = df[["date", "low"]].copy()
                        df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
                        df["low"] = pd.to_numeric(df["low"], errors="coerce")
                        df = df.dropna()
                        df.to_csv(cf, index=False)
                        s = df.set_index("date")["low"]
                        break
                    except Exception as _e:
                        s = None
                        _t.sleep(2)
                if s is None:
                    print(f"[warn] {code} akshare拉low失败，退化为收盘判定")
            except Exception as _e:
                print(f"[warn] {code} akshare不可用({_e})，退化为收盘判定")
        if s is None:
            s = prices[code]
        lows[code] = pd.Series(s.values, index=pd.to_datetime(s.index)).reindex(prices.index).ffill()
    return pd.DataFrame(lows)


def simulate(prices, low_df=None, timing=None):
    """逐日模拟 + 单日熔断。timing: None/close/intra。返回 (metrics, closed, open_trades, melt_counter)。"""
    daily_ret = prices.pct_change().fillna(0.0)
    dates = prices.index
    port = pd.Series(0.0, index=dates)
    adj = pd.Series(1.0, index=dates)
    switches = stop_exits = invested = 0
    melt_count = 0

    held = set()
    entry, stop, peak = {}, {}, {}
    firstday = {}
    last_switch = None
    cur_key = None
    entered = False
    closed = []
    open_trades = []

    def _has_seg(c):
        return c in firstday

    def _settle(c, reason, day=None):
        if day is None:
            day = d  # 默认取当前迭代日（自由变量）
        px = prices.loc[day, c] if hasattr(day, "date") else prices.iloc[-1][c]
        prev = entry.get(c)
        ret = px / prev - 1.0 if prev else 0.0
        closed.append({"code": c, "ret": ret, "days": (day - firstday[c]).days, "reason": reason})
        firstday.pop(c, None)

    def _worst_daily(c, d):
        """当日 c 的『最坏盘中收益』= low[d]/前收 - 1（未隔日跳空处理）。"""
        if d == dates[0]:
            return 0.0
        prev_c = prices.loc[dates[dates.get_loc(d) - 1], c]
        lo = low_df.loc[d, c] if low_df is not None else prices.loc[d, c]
        return lo / prev_c - 1.0 if prev_c else 0.0

    for i, d in enumerate(dates):
        if i == 0:
            continue
        # ---- 1) 持仓内部：S/R / 跟踪止损 ----
        if held:
            changed = False
            for c in list(held):
                px = prices.loc[d, c]
                is_def = (c == DEF_CODE)
                this_trail = DEF_TRAIL if is_def else TRAIL_PCT
                if this_trail is None:
                    continue
                if peak.get(c) is not None and px > peak[c]:
                    peak[c] = px
                    ts = peak[c] * (1 - this_trail)
                    if stop.get(c) is None or ts > stop[c]:
                        stop[c] = ts
                if stop.get(c) is not None and px <= stop[c]:
                    held.discard(c)
                    stop_exits += 1
                    if _has_seg(c):
                        _settle(c, "止损·S/R/跟踪", d)
                    changed = True
            if changed:
                last_switch = None

        # ---- 2) 单日熔断【盘中即时】预判：持仓标的已最坏触及 -3% → 当日直接转防御 ----
        intra_melt = False
        if (timing == "intra") and held:
            intra_melt = any(_worst_daily(c, d) <= -MAX_DLY_LOSS for c in held)
            if intra_melt:
                melt_count += 1

        # ---- 3) 决策 + 换仓 ----
        can = (not held) or (last_switch is None) or ((d - last_switch).days >= MIN_HOLD)
        target_override = intra_melt  # 当日盘中熔断 → 目标强制防御
        if can:
            if target_override:
                tgt = [DEF_CODE]
            else:
                tgt, _ = mh.day_target(prices, d, {"DEFENSIVE": DEF_CODE}, universe=CURRENT_UNI)
            if DEF_MOM and isinstance(tgt, (list, tuple)) and tgt and str(tgt[0]) == DEF_CODE:
                dm = bse.sr.momentum(pd.Series(prices.loc[:d, DEF_CODE]), DEF_MOM)
                if dm is None or np.isnan(dm) or dm <= 0:
                    tgt = "cash"
            key = "cash" if tgt == "cash" else tuple(sorted(str(c) for c in tgt))
            if entered and key != cur_key and not target_override:
                switches += 1
                adj.loc[dates >= d] *= (1 - bse.FEE)
                last_switch = d
            elif not entered:
                entered = True
            if (not held) or (key != cur_key):
                old, new = set(held), (set(key) if key != "cash" else set())
                for c in (old - new):
                    if _has_seg(c):
                        _settle(c, "换仓卖出", d)
                held = set()
                entry, stop, peak, firstday = {}, {}, {}, {}
                if tgt != "cash":
                    for c in tgt:
                        c = str(c)
                        px = prices.loc[d, c]
                        if c == DEF_CODE:
                            held.add(c); entry[c] = px; stop[c] = None; firstday[c] = d
                        else:
                            nsup, nres = bse.compute_sr(prices.loc[:d, c], px)
                            if nres is not None and px >= nres * (1 - bse.ENTRY_BUF):
                                continue
                            sp = bse.sr_stop_price(px, nsup)
                            held.add(c); entry[c] = px; stop[c] = sp; firstday[c] = d
                        peak[c] = px
                cur_key = key if held or key == "cash" else "cash"

        # ---- 4) 当日收益 ----
        if held:
            w = 1.0 / len(held)
            port.loc[d] = sum(w * daily_ret.loc[d, c] for c in held)
            invested += 1

        # ---- 5) 【尾盘】单日熔断：当日组合收益 <= -3% → 记入（当日已按原持仓走完，次日恢复） ----
        if timing == "close" and (not intra_melt) and port.loc[d] <= -MAX_DLY_LOSS:
            melt_count += 1

    for c in list(held):
        px = prices.iloc[-1][c]
        ret = px / entry[c] - 1.0 if entry.get(c) else 0.0
        open_trades.append({"code": c, "ret": ret, "days": (dates[-1] - firstday[c]).days})

    equity = (1 + port).cumprod() * adj
    n = max(len(equity), 1)
    cum = float(equity.iloc[-1])
    ann = float(cum ** (252.0 / n) - 1) if cum > 0 else -1.0
    maxdd = float(((equity - equity.cummax()) / equity.cummax()).min())
    rr = equity.pct_change().dropna()
    sd = float(rr.std())
    sharpe = float(rr.mean() / sd * np.sqrt(252)) if sd > 0 else float("nan")
    calmar = ann / abs(maxdd) if maxdd < 0 else float("nan")
    metrics = {"equity": equity, "port": port, "cum": cum, "ann": ann, "maxdd": maxdd,
               "sharpe": sharpe, "calmar": calmar, "switches": switches,
               "stop_exits": stop_exits, "time_in_mkt": invested / max(n - 1, 1),
               "melt_count": melt_count}
    return metrics, closed, open_trades


def report(name, m):
    print(f"\n===== {name} =====")
    print(f"  累计倍数 : {m['cum']:7.4f}")
    print(f"  年化     : {m['ann']*100:6.2f}%   最大回撤 {abs(m['maxdd'])*100:5.2f}%   "
          f"Calmar {m['calmar']:5.2f}   Sharpe {m['sharpe']:5.2f}")
    print(f"  换仓次数 : {m['switches']}   止损次数 {m['stop_exits']}   在市占比 {m['time_in_mkt']*100:5.1f}%"
          f"   熔断触发 {m['melt_count']} 天")


def main():
    prices = load_prices()
    print(f"数据窗口: {prices.index[0].date()} ~ {prices.index[-1].date()}  共 {len(prices)} 个交易日 / {len(prices.columns)} 标的")
    low_df = load_lows(prices)

    base, _, _ = simulate(prices, low_df, timing=None)
    hc, _, _ = simulate(prices, low_df, timing="close")
    hi, _, _ = simulate(prices, low_df, timing="intra")

    report("base(无当日熔断)", base)
    report("halt_close(14:50收盘触发)", hc)
    report("halt_intra(盘中-3%即触发)", hi)

    # 样本外(2021+)口径：换手/熔断幅度是否在近年同样成立
    sp = pd.Timestamp("2021-01-01")
    print("\n===== 样本外(2021+) =====")
    for name, m in (("base", base), ("close", hc), ("intra", hi)):
        e = m["equity"][m["equity"].index >= sp]
        n = max(len(e), 1)
        cum = float(e.iloc[-1] / e.iloc[0])
        ann = float(cum ** (252.0 / n) - 1) if cum > 0 else -1.0
        maxdd = float(((e - e.cummax()) / e.cummax()).min())
        calmar = ann / abs(maxdd) if maxdd < 0 else float("nan")
        print(f"  {name:5s} 年化 {ann*100:6.2f}%   累计 {cum:5.2f}x   回撤 {abs(maxdd)*100:5.2f}%   Calmar {calmar:5.2f}")

    # 逐年收益（1 + 日收益）连乘，按自然年 groupby
    yearly = pd.DataFrame({name: (1 + m["equity"].pct_change().fillna(0.0))
                                  .groupby(m["equity"].index.to_period("Y"), observed=True).prod() - 1
                           for name, m in (("base", base), ("close", hc), ("intra", hi))})
    print("\n===== 逐年收益(%) =====")
    print((yearly * 100).round(2).to_string())


if __name__ == "__main__":
    main()