# -*- coding: utf-8 -*-
"""
backtest_current.py —— 当前实盘策略的「收益率 + 保本率」回测
================================================================================
目的：量化当前实盘(auto_run.py)落地策略的长期表现，重点回答"收益率 + 保本率"。

当前策略(与 auto_run.py CFG 完全对齐):
  - 标的池：原 6 只 + 513500(标普500) = 7 只真实ETF（sina 缓存数据，2015~2026）
  - MIN_HOLD_DAYS = 1（1天锁）
  - 防御资产 518880(黄金)，DEFENSE_MODE=defensive
  - 防御自身动量门 DEF_MOM_DAYS=10（动量转负→空仓等待）
  - 保护：HOLD_N=1 + S/R动态止损(-8%兜底) + 跟踪止损6%(防御不加) + 入场过滤(贴阻力跳过) + 换仓费0.10%
  - day_target → strategy_rotator.decide()（含全局风险-off判定，与实盘同一逻辑源）

指标口径：
  1) 收益率：累计倍数 / 年化 / 最大回撤 / Calmar / Sharpe / 换仓次数 / 持仓占比
  2) 交易日胜率：持仓日组合收益 >0 / =0 / <0 占比（不亏 = >=0）
  3) 交易级保本率：已了结"每段持仓(买入→清仓)"中收益>=0(不亏本金) 占比，
     并分 盈利(>0)/保本(==0)/亏损(<0)；期末未了结持仓按期末价单列，不计入已了结。
  4) 年度收益表：逐年收益，看"常年正收益"稳健性。

模拟与 research/backtest_minhold.simulate_daily 使用同一批原语（bse.compute_sr /
sr_stop_price / FEE / ENTRY_BUF / mh.day_target），确保不偏离"当前策略"口径。
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

MIN_HOLD = 1                 # 与实盘一致：1天锁
DEF_TRAIL = None             # 防御黄金不加跟踪止损（当前落地）
DEF_MOM = 10                 # 防御动量门
TRAIL_PCT = 0.06             # 持仓跟踪止损 6%
CURRENT_UNI = bue.BASE_CODES + ["513500"]   # 当前实盘标的池 = 原6只 + 标普500


def load_prices7():
    """7 只真实 ETF 收盘价（同 bue 数据口径，过滤出当前标的池）。"""
    p11 = bue.load_prices()
    return p11.reindex(columns=CURRENT_UNI).dropna().sort_index()


def simulate_episode(price, min_hold=MIN_HOLD, trail_pct=TRAIL_PCT, sr_params=None,
                     def_trail_pct=DEF_TRAIL, def_mom_days=DEF_MOM, universe=CURRENT_UNI):
    """逐日模拟 + 交易段切分。返回 (指标dict, 已了结交易list, 未了结持仓list)。
    逻辑严格对齐 backtest_minhold.simulate_daily，额外在标的进出场时结算每段收益。"""
    daily_ret = price.pct_change().fillna(0.0)
    dates = price.index
    port = pd.Series(0.0, index=dates)
    adj = pd.Series(1.0, index=dates)
    switches = stop_exits = invested = 0
    def_code = str((sr_params or {}).get("DEFENSIVE", bse.DEF))

    held = set()
    entry, stop, peak = {}, {}, {}
    firstday = {}                 # code -> 当前未结算建仓日
    last_switch = None
    cur_key = None
    entered = False
    closed = []                   # 已了结：{code, ret, days, reason}
    open_trades = []              # 未了结：{code, ret, days}

    def _has_seg(code):
        return code in firstday

    def _settle(code, reason):
        """已了结一段持仓：ret = 期价/建仓价-1（不含手续费，仅做保本判定）。"""
        px = price.loc[d, code]
        ret = px / entry[code] - 1.0 if entry.get(code) else 0.0
        closed.append({"code": code, "ret": ret,
                       "days": (d - firstday[code]).days, "reason": reason})
        firstday.pop(code, None)   # 标记该段已结算，防重复

    for i, d in enumerate(dates):
        if i == 0:
            continue
        # ---- 1) 持仓内部：S/R / 跟踪止损 ----
        if held:
            changed = False
            for c in list(held):
                px = price.loc[d, c]
                is_def = (c == def_code)
                this_trail = def_trail_pct if is_def else trail_pct
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
                last_switch = None   # 止损当日解锁（对齐实盘 _stop_today）
        # ---- 2) 每日决策 + 换仓 ----
        can = (not held) or (last_switch is None) or ((d - last_switch).days >= min_hold)
        if can:
            tgt, _ = mh.day_target(price, d, sr_params, universe=universe)
            # 防御动量门：防御资产自身动量转负 → 视同空仓（不买回）
            if def_mom_days and isinstance(tgt, (list, tuple)) and tgt and str(tgt[0]) == def_code:
                dm = bse.sr.momentum(pd.Series(price.loc[:d, def_code]), def_mom_days)
                if dm is None or np.isnan(dm) or dm <= 0:
                    tgt = "cash"
            key = "cash" if tgt == "cash" else tuple(sorted(str(c) for c in tgt))
            if entered and key != cur_key:
                switches += 1
                adj.loc[dates >= d] *= (1 - bse.FEE)
                last_switch = d
            elif not entered:
                entered = True
            if (not held) or (key != cur_key):
                # 结算被撤出的旧持仓（换仓卖出），再重建
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
                            held.add(c); entry[c] = px; stop[c] = None
                            firstday[c] = d
                        else:
                            nsup, nres = bse.compute_sr(price.loc[:d, c], px)
                            if nres is not None and px >= nres * (1 - bse.ENTRY_BUF):
                                continue          # 入场过滤：贴强阻力不进
                            sp = bse.sr_stop_price(px, nsup)
                            held.add(c); entry[c] = px; stop[c] = sp; firstday[c] = d
                        peak[c] = px
                cur_key = key if held or key == "cash" else "cash"
        # ---- 3) 当日收益（等权，HOLD_N=1 基本单目标） ----
        if held:
            w = 1.0 / len(held)
            port.loc[d] = sum(w * daily_ret.loc[d, c] for c in held)
            invested += 1

    # ---- 期末未了结持仓：按期末价结算（单列，不计入已了结保本率） ----
    for c in list(held):
        px = price.loc[dates[-1], c]
        ret = px / entry[c] - 1.0 if entry.get(c) else 0.0
        open_trades.append({"code": c, "ret": ret, "days": (dates[-1] - firstday[c]).days})

    # ---- 指标 ----
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
               "stop_exits": stop_exits, "time_in_mkt": invested / max(n - 1, 1)}
    return metrics, closed, open_trades


def pct(x, n):
    return (x / n * 100) if n else 0.0


def main():
    price = load_prices7()
    print(f"真实ETF数据 {price.index[0].date()} ~ {price.index[-1].date()}，{len(price)} 交易日，标的 {CURRENT_UNI}")
    print(f"配置 = 7只 + MIN_HOLD={MIN_HOLD}d + 黄金防御 + 动量门{DEF_MOM}d + 跟踪{TRAIL_PCT*100:.0f}% + S/R止损 + 入场过滤 + 换仓费{bse.FEE*100:.2f}%\n")

    m, closed, openx = simulate_episode(price, sr_params=bue.GOLD)

    print("=== 收益率 ===")
    print(f"  累计倍数 : {m['cum']:.3f}x")
    print(f"  年化     : {m['ann']*100:+.2f}%")
    print(f"  最大回撤 : {m['maxdd']*100:.2f}%")
    print(f"  Calmar   : {m['calmar']:.2f}   Sharpe: {m['sharpe']:.2f}")
    print(f"  换仓次数 : {m['switches']}  止损次数: {m['stop_exits']}  持仓占比: {m['time_in_mkt']*100:.1f}%")

    pos = m["port"][m["port"] != 0]
    up, eq, dn = float((pos > 0).mean()), float((pos == 0).mean()), float((pos < 0).mean())
    print("\n=== 交易日胜率（持仓日口径）===")
    print(f"  盈利日 {up*100:.2f}% | 保本日 {eq*100:.2f}% | 亏损日 {dn*100:.2f}% | 不亏(>=0) {(up+eq)*100:.2f}%")

    if closed:
        rets = [t["ret"] for t in closed]
        prof, flat, loss = sum(r > 0 for r in rets), sum(r == 0 for r in rets), sum(r < 0 for r in rets)
        n = len(rets)
        print("\n=== 交易级保本率（已了结持仓段）===")
        print(f"  总交易段 : {n}")
        print(f"  盈利 {prof} ({pct(prof,n):.1f}%) | 保本 {flat} ({pct(flat,n):.1f}%) | 亏损 {loss} ({pct(loss,n):.1f}%)")
        print(f"  保本率(收益>=0) : {pct(prof+flat, n):.2f}%")
        g = [r for r in rets if r > 0]; l = [r for r in rets if r < 0]
        print(f"  盈亏比 : 平均盈利 {np.mean(g)*100:+.2f}% vs 平均亏损 {np.mean(l)*100:+.2f}% (合计 {len(g)+len(l)} 段)")
    else:
        print("\n（无已了结交易）")

    if openx:
        print("\n  期末未了结持仓（按期末价，不计入已了结保本率）: "
              + "; ".join(f"{t['code']} {t['ret']*100:+.2f}% ({t['days']}d)" for t in openx))

    eq_s = m["equity"]
    year_rows, prev = [], None
    for y in sorted(eq_s.index.year.unique()):
        ys = eq_s[eq_s.index.year == y]
        base = prev if prev is not None else _year_start(eq_s, y)
        year_rows.append((y, ys.iloc[-1] / base - 1.0))
        prev = ys.iloc[-1]
    pos_y = sum(1 for _, r in year_rows if r > 0)
    print("\n=== 年度收益 ===")
    for y, r in year_rows:
        print(f"  {y}: {r*100:+7.2f}%")
    print(f"  有数据 {len(year_rows)} 个年度，正收益 {pos_y} 个（{pct(pos_y, len(year_rows)):.0f}%）")


def _year_start(eq, y):
    idx = eq.index[eq.index.year == y]
    return eq.loc[idx[0]]


if __name__ == "__main__":
    main()