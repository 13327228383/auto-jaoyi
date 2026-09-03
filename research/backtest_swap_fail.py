# -*- coding: utf-8 -*-
"""
backtest_swap_fail.py —— 换仓「滑点 + 未成交」敏感性回测
================================================================================
动机：对手价下单(买=卖一/卖=买一)实测滑点仅 0.001%~0.045%。本脚本量化两点：
  1) 换仓成本(滑点/手续费合计)从 0.05% 扫到 1.00%，收益率/回撤/保本率如何退化
     —— 检验当前回测假设 0.10% 是否够保守、实盘对手价损耗是否更小。
  2) 换仓「未成交→延后一天」按概率 fail_p 注入：当一次换仓失败(挂单未成交)，
     保留旧持仓到下一交易日再尝试。量化频繁未成交对收益/回撤/保本率的影响。
标的池/决策与 backtest_current 完全一致(7只 + 黄金防御 + 1天锁 + 动量门10)。
A股 ETF 无涨跌停限制，故不模拟涨跌停无法成交；未成交主因≈流动性/停机。
"""
import os, sys, random
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import backtest_minhold as mh
import backtest_sr_enhance as bse
import backtest_universe_ext as bue

MIN_HOLD = 1
DEF_TRAIL = None
DEF_MOM = 10
TRAIL_PCT = 0.06
CURRENT_UNI = bue.BASE_CODES + ["513500"]


def load_prices7():
    p11 = bue.load_prices()
    return p11.reindex(columns=CURRENT_UNI).dropna().sort_index()


def simulate(price, fee=0.001, fail_p=0.0, seed=0):
    """逐日模拟。fee=换仓成本；fail_p=每次换仓触发时未成交(延后一天)概率。"""
    rng = random.Random(seed)
    daily_ret = price.pct_change().fillna(0.0)
    dates = price.index
    port = pd.Series(0.0, index=dates)
    adj = pd.Series(1.0, index=dates)
    switches = stop_exits = invested = fail_skips = 0
    def_code = str(bse.DEF)

    held = set()
    entry, stop, peak = {}, {}, {}
    firstday = {}
    last_switch = None
    cur_key = None
    entered = False
    closed = []
    open_trades = []

    def _has_seg(code):
        return code in firstday

    def _settle(code, reason):
        px = price.loc[d, code]
        ret = px / entry[code] - 1.0 if entry.get(code) else 0.0
        closed.append({"code": code, "ret": ret,
                       "days": (d - firstday[code]).days, "reason": reason})
        firstday.pop(code, None)

    for i, d in enumerate(dates):
        if i == 0:
            continue
        if held:
            changed = False
            for c in list(held):
                px = price.loc[d, c]
                is_def = (c == def_code)
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
                        _settle(c, "止损·S/R/跟踪")
                    changed = True
            if changed:
                last_switch = None

        can = (not held) or (last_switch is None) or ((d - last_switch).days >= MIN_HOLD)
        if can:
            tgt, _ = mh.day_target(price, d, None, universe=CURRENT_UNI)
            if DEF_MOM and isinstance(tgt, (list, tuple)) and tgt and str(tgt[0]) == def_code:
                dm = bse.sr.momentum(pd.Series(price.loc[:d, def_code]), DEF_MOM)
                if dm is None or np.isnan(dm) or dm <= 0:
                    tgt = "cash"
            key = "cash" if tgt == "cash" else tuple(sorted(str(c) for c in tgt))

            switch_now = entered and key != cur_key
            changed_key = (not held) or (key != cur_key)

            skip_now = False
            if switch_now and fail_p > 0 and rng.random() < fail_p:
                skip_now = True
                fail_skips += 1

            if changed_key and not skip_now:
                if switch_now:
                    switches += 1
                    adj.loc[dates >= d] *= (1 - fee)
                    last_switch = d
                elif not entered:
                    entered = True
                old, new = set(held), (set(key) if key != "cash" else set())
                for c in (old - new):
                    if _has_seg(c):
                        _settle(c, "换仓卖出")
                held = set()
                entry, stop, peak, firstday = {}, {}, {}, {}
                if tgt != "cash":
                    for c in tgt:
                        c = str(c)
                        px = price.loc[d, c]
                        if c == def_code:
                            held.add(c); entry[c] = px; stop[c] = None; firstday[c] = d
                        else:
                            nsup, nres = bse.compute_sr(price.loc[:d, c], px)
                            if nres is not None and px >= nres * (1 - bse.ENTRY_BUF):
                                continue
                            sp = bse.sr_stop_price(px, nsup)
                            held.add(c); entry[c] = px; stop[c] = sp; firstday[c] = d
                        peak[c] = px
                cur_key = key if held or key == "cash" else "cash"

        if held:
            w = 1.0 / len(held)
            port.loc[d] = sum(w * daily_ret.loc[d, c] for c in held)
            invested += 1

    for c in list(held):
        px = price.loc[dates[-1], c]
        ret = px / entry[c] - 1.0 if entry.get(c) else 0.0
        open_trades.append({"code": c, "ret": ret,
                            "days": (dates[-1] - firstday[c]).days if c in firstday else 0})

    equity = (1 + port).cumprod() * adj
    n = max(len(equity), 1)
    cum = float(equity.iloc[-1])
    ann = float(cum ** (252.0 / n) - 1) if cum > 0 else -1.0
    maxdd = float(((equity - equity.cummax()) / equity.cummax()).min())
    rr = equity.pct_change().dropna()
    sd = float(rr.std())
    sharpe = float(rr.mean() / sd * np.sqrt(252)) if sd > 0 else float("nan")
    calmar = ann / abs(maxdd) if maxdd < 0 else float("nan")
    if closed:
        rets = [t["ret"] for t in closed]
        breakeven = sum(r >= 0 for r in rets) / len(rets)
    else:
        breakeven = float("nan")
    metrics = {"cum": cum, "ann": ann, "maxdd": maxdd, "sharpe": sharpe,
               "calmar": calmar, "switches": switches, "stop_exits": stop_exits,
               "time_in_mkt": invested / max(n - 1, 1), "fail_skips": fail_skips,
               "breakeven": breakeven, "n_trades": len(closed)}
    return metrics


def main():
    price = load_prices7()
    print(f"数据 {price.index[0].date()} ~ {price.index[-1].date()}，{len(price)} 交易日，{len(CURRENT_UNI)} 只")
    print("=" * 86)

    base = simulate(price, fee=0.001, fail_p=0.0, seed=1)
    print("基准(换仓费0.10%, 未成交0%): "
          f"年化{base['ann']*100:+.2f}% 最大回撤{base['maxdd']*100:.2f}% "
          f"Calmar{base['calmar']:.2f} 累计{base['cum']:.2f}x 保本率{base['breakeven']*100:.1f}%\n")

    print("\n[1] 换仓成本(滑点+费用)敏感性, 未成交0%:")
    print(f"{'成本':>8} {'年化':>10} {'最大回撤':>10} {'Calmar':>8} {'累计':>8} {'保本率':>8} {'换仓':>6}")
    for fee in (0.0005, 0.001, 0.002, 0.005, 0.01):
        m = simulate(price, fee=fee, fail_p=0.0, seed=1)
        print(f"{fee*100:7.2f}% {m['ann']*100:+9.2f}% {m['maxdd']*100:+9.2f}% "
              f"{m['calmar']:8.2f} {m['cum']:7.2f}x {m['breakeven']*100:7.1f}% {m['switches']:6d}")

    print("\n[2] 换仓未成交概率敏感性, 换仓费0.10%:")
    print(f"{'fail_p':>8} {'年化':>10} {'最大回撤':>10} {'Calmar':>8} {'累计':>8} {'保本率':>8} {'跳过':>6}")
    for fp in (0.0, 0.01, 0.05, 0.10, 0.25):
        m = simulate(price, fee=0.001, fail_p=fp, seed=1)
        print(f"{fp*100:7.1f}% {m['ann']*100:+9.2f}% {m['maxdd']*100:+9.2f}% "
              f"{m['calmar']:8.2f} {m['cum']:7.2f}x {m['breakeven']*100:7.1f}% {m['fail_skips']:6d}")


if __name__ == "__main__":
    main()