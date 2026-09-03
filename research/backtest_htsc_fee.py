# -*- coding: utf-8 -*-
"""
backtest_htsc_fee.py —— 国泰海通真实佣金口径的「净额回测」
================================================================================
实盘成本（用户提供）：
  佣金 = 万2.38 含规费，单笔最低 5 元（成交额 <=2.1万 恒收5元 → 单边费率被5元抬升）。
  但这是股票口径；ETF 是否同费率、是否免印花税，需建模验证。

本脚本用「订单边 + 绝对资金」计费，逐日推进现金/持仓，回答四个问题：
  1) 真实 ETF 佣金（万2.38、双向、无印花税、含5元门槛）下，策略长期净年化是多少？
  2) 与当前回测假设 0.10%/次切换 相比，真实成本是更保守还是更乐观？
  3) 5 元最低门槛在多小资金时开始拖累收益（扫描不同本金）？
  4) ETF vs 股票（股票卖出加 0.05% 印花税）成本差多少？

口径：
  - HOLD_N=1 全仓单标的：每次换仓 = 卖出旧(1边) + 买入新(1边)；止损离场 = 卖出(1边)。
  - 佣金 fee(amount) = max(amount*0.000238, 5)。ETF 买卖均无印花税；股票卖出另收 amount*0.0005。
  - 决策/候选/动量门 与 backtest_minhold.simulate_daily 完全一致（同一 decide() 源）。
  - 数据：真实ETF(原6 + 513500)，2015-2026。
注意：此处未建模实盘已有的「再入冷却」与「动量迟滞」（回测简化，净费略被高估，偏保守）。
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import backtest_minhold as mh
import backtest_sr_enhance as bse
import backtest_universe_ext as bue
import backtest_current as bc

SPLIT = pd.Timestamp("2021-01-01")
UNIVERSE = bc.CURRENT_UNI
GOLD = dict(bue.GOLD)

COMM_RATE = 0.000238        # 万2.38 含规费
MIN_FEE = 5.0               # 单边最低佣金 5 元
STAMP_STOCK_SELL = 0.0005   # 股票卖出印花税(ETF 免)

def _fee(amount, stamp=False):
    f = max(amount * COMM_RATE, MIN_FEE)
    if stamp:
        f += amount * STAMP_STOCK_SELL
    return f

def simulate_net(price, principal, stamp=False, trail=0.04, def_trail=0.04,
                 def_mom=10, min_hold=1, sr_params=None, universe=UNIVERSE,
                 lookahead=False):
    """净额逐日模拟：按订单边扣真实佣金。
    lookahead=True 时在建仓当天额外记入该标的 daily_ret[d](d-1→d 的涨幅)，
    用于复现/剂量化标准 simulate_daily 的"建仓当日提前记一天"前视特性的影响。"""
    dates = price.index
    daily_ret = price.pct_change().fillna(0.0)
    equity = pd.Series(0.0, index=dates)
    switches = orders = stop_exits = invested = 0
    def_code = str((sr_params or GOLD).get("DEFENSIVE", "518880"))
    lookahead_carry = {}            # 建仓日要补记的前视收益 code->ret

    cash = float(principal)
    held = {}                    # code -> {sh, entry, stop, peak}
    last_switch = None
    cur_key = None
    entered = False
    prev_total = float(principal)

    def _sell(c, px):
        nonlocal cash
        v = held[c]["sh"] * px
        cash += v - _fee(v, stamp=stamp)
        held.pop(c, None)

    def _buy(c, px):
        nonlocal cash
        amount = cash                       # 全仓
        fee = _fee(amount, stamp=False)
        buy_notional = amount - fee
        if buy_notional <= 0:
            return
        held[c] = {"sh": buy_notional / px, "entry": px, "stop": None, "peak": px}
        cash = 0.0

    for i, d in enumerate(dates):
        if i == 0:
            continue
        switched_today = False
        new_asset = None
        # ---- 1) 持仓内部止损（S/R 或 跟踪） ----
        if held:
            changed = False
            for c in list(held.keys()):
                px = float(price.loc[d, c])
                is_def = (c == def_code)
                this_trail = def_trail if is_def else trail
                if this_trail:
                    if px > held[c]["peak"]:
                        held[c]["peak"] = px
                        ts = px * (1 - this_trail)
                        if held[c]["stop"] is None or ts > held[c]["stop"]:
                            held[c]["stop"] = ts
                st = held[c]["stop"]
                if st is not None and px <= st:
                    _sell(c, px)             # 止损卖出：1 边
                    orders += 1
                    stop_exits += 1
                    changed = True
            if changed:
                last_switch = None
        # ---- 2) 决策 + 换仓 ----
        can = (not held) or (last_switch is None) or ((d - last_switch).days >= min_hold)
        if can:
            tgt, _ = mh.day_target(price, d, sr_params or GOLD, universe=universe)
            if def_mom and isinstance(tgt, (list, tuple)) and tgt and str(tgt[0]) == def_code:
                dm = bse.sr.momentum(pd.Series(price.loc[:d, def_code]), def_mom)
                if dm is None or np.isnan(dm) or dm <= 0:
                    tgt = "cash"
            key = "cash" if tgt == "cash" else tuple(sorted(str(c) for c in tgt))
            if entered and key != cur_key:
                switches += 1
                switched_today = True
                last_switch = d
            elif not entered:
                entered = True
            if (not held) or (key != cur_key):
                for c in list(held.keys()):
                    if c in held:
                        _sell(c, float(price.loc[d, c]))
                        orders += 1
                held = {}
                if tgt != "cash":
                    for c in tgt:
                        c = str(c)
                        px = float(price.loc[d, c])
                        if c == def_code:
                            _buy(c, px); orders += 1
                        else:
                            nsup, nres = bse.compute_sr(price.loc[:d, c], px)
                            if nres is not None and px >= nres * (1 - bse.ENTRY_BUF):
                                continue            # 入场过滤
                            _buy(c, px); orders += 1
                            held[c]["stop"] = bse.sr_stop_price(px, nsup)
                        new_asset = c
                cur_key = key if held or key == "cash" else "cash"
        # ---- 3) 当日收益（现金+持仓市值） ----
        total = cash + sum(h["sh"] * float(price.loc[d, cc]) for cc, h in held.items())
        ret = total / prev_total - 1.0
        # lookahead: 复刻标准 simulate_daily 的建仓日前视——建仓/换仓当天按「新资产当日涨幅」记账
        if lookahead and switched_today and new_asset is not None and new_asset in held:
            ret = float(daily_ret.loc[d, new_asset])
        equity.loc[d] = ret
        prev_total = total
        lookahead_carry = {}
        if held:
            invested += 1

    eq = (1 + equity).cumprod()
    n = max(len(eq), 1)
    cum = float(eq.iloc[-1])
    ann = float(cum ** (252.0 / n) - 1) if cum > 0 else -1.0
    maxdd = float(((eq - eq.cummax()) / eq.cummax()).min())
    rr = eq.pct_change().dropna()
    sd = float(rr.std())
    sharpe = float(rr.mean() / sd * np.sqrt(252)) if sd else float("nan")
    calmar = ann / abs(maxdd) if maxdd < 0 else float("nan")
    return {"equity": eq, "cum": cum, "ann": ann, "maxdd": maxdd, "calmar": calmar,
            "sharpe": sharpe, "switches": switches, "orders": orders,
            "stop_exits": stop_exits, "time_in_mkt": invested / max(n - 1, 1)}

def oos(r):
    eq = r["equity"].loc[SPLIT:]
    if len(eq) < 2:
        return None
    n = max(len(eq), 1)
    cum = float(eq.iloc[-1])
    ann = float(cum ** (252.0 / n) - 1) if cum > 0 else -1.0
    return ann

def main():
    global COMM_RATE, MIN_FEE
    price = bc.load_prices7()
    print(f"数据 {price.index[0].date()} ~ {price.index[-1].date()}，{len(price)} 交易日，标的 {list(price.columns)}\n")
    print(f"佣金=万{COMM_RATE*10000:.2f} 含规费，单边最低{MIN_FEE:.0f}元（单边成交额<=约{MIN_FEE/COMM_RATE/10000:.1f}万时最低佣金抬升费率）")
    print(f"ETF 卖无印花税（若属实）；股票卖出另收 {STAMP_STOCK_SELL*100:.1f}% 印花税\n")

    print("=" * 100)
    print("0) 关键：标准 simulate_daily 存在「建仓/换仓当天记新资产当日涨幅」前视")
    print("   → 逐年虚增收益。本脚本用「当日收盘实际成交、次日开始计收益」的诚实时序净值。")
    print("   （诚实口径年化与 auto_run.py 自带声明 ~6.7% 一致）")
    print("=" * 100)
    honest0 = simulate_net(price, 500_000.0, stamp=False, sr_params=GOLD)            # 真实费
    ref = mh.simulate_daily(price, 1, trail_pct=0.04, sr_params=GOLD, def_trail_pct=0.04, def_mom_days=10, universe=UNIVERSE)
    print(f"  {'口径':<30s} {'净年化':>8s} {'回撤':>8s} {'Calmar':>7s} {'换仓':>5s} {'订单边':>6s}")
    print(f"  {'标准simulate_daily(含前视)':<27s} {ref['ann']*100:7.2f}% {ref['maxdd']*100:7.2f}% {ref['calmar']:7.2f} {ref['switches']:5d} {'--':>6s}")
    print(f"  {'诚实时序+真实佣金(50万)':<25s} {honest0['ann']*100:7.2f}% {honest0['maxdd']*100:7.2f}% {honest0['calmar']:7.2f} {honest0['switches']:5d} {honest0['orders']:6d}")
    print(f"  前视口径 → 诚实口径 年化差 = {(ref['ann']-honest0['ann'])*100:+.2f}pp（其中大部分为建仓日前视，非真实收益）")

    print("\n" + "=" * 100)
    print("1) 诚实口径下：真实 ETF 佣金(万2.38) 对净年化的拖累")
    print("=" * 100)
    # 零费版（毛收益基准）
    old_r, old_m = COMM_RATE, MIN_FEE
    h1 = simulate_net(price, 500_000.0, stamp=False, sr_params=GOLD, lookahead=False)
    COMM_RATE, MIN_FEE = 0.0, 0.0
    h1z = simulate_net(price, 500_000.0, stamp=False, sr_params=GOLD, lookahead=False)
    COMM_RATE, MIN_FEE = old_r, old_m
    print(f"  零佣金(毛)             : 年化{h1z['ann']*100:6.2f}%  回撤{h1z['maxdd']*100:6.2f}%")
    print(f"  万2.38含规费(50万,无门槛): 年化{h1['ann']*100:6.2f}%  回撤{h1['maxdd']*100:6.2f}%")
    print(f"  → 真实佣金年化拖累 = {(h1z['ann']-h1['ann'])*100:+.2f}pp（订单 {h1['orders']} 边）")

    print("\n2) 5 元最低佣金门槛：不同本金下的诚实净年化（小资金被最低佣金抬高有效费率）")
    print(f"   {'本金':>8s} {'诚实净年化':>10s} {'回撤':>8s} {'Calmar':>7s} {'订单边':>6s}")
    for p in [5000, 10000, 15000, 21000, 50000, 200000, 500000]:
        rr = simulate_net(price, float(p), stamp=False, sr_params=GOLD)
        print(f"   {p:>8d} {rr['ann']*100:10.2f}% {rr['maxdd']*100:8.2f}% {rr['calmar']:7.2f} {rr['orders']:6d}")

    print("\n3) ETF(免印花) vs 股票(卖0.05%印花)，同一佣金下成本差（诚实口径，本金50万）")
    e = simulate_net(price, 500_000.0, stamp=False, sr_params=GOLD)
    s = simulate_net(price, 500_000.0, stamp=True, sr_params=GOLD)
    print(f"   ETF买卖(免印花) : 年化{e['ann']*100:6.2f}%  Calmar{e['calmar']:5.2f}  订单{e['orders']}边")
    print(f"   股票买卖(0.05%印花) : 年化{s['ann']*100:6.2f}%  Calmar{s['calmar']:5.2f}  订单{s['orders']}边")
    print(f"   → ETF 因免印花税每年省 ≈ {(e['ann']-s['ann'])*100:+.2f}pp（若券商确有该政策）")

    print("\n4) 结论速览")
    print(f"   · 真实 ETF 佣金(万2.38/边)≈ 0.0476%/满仓换仓，拖累约 {(h1z['ann']-h1['ann'])*100:.1f}pp/年 —— 影响小")
    print(f"   · 5元最低门槛：账户≥约2.1万时无效；越小于2.1万拖累越重（5k→最低佣金主导）")
    print(f"   · ETF vs 股票：若ETF免印花税，ETF比股票每年省约 {(e['ann']-s['ann'])*100:.2f}pp")
    print(f"   · 但更重要的是：诚实口径年化仅约 {h1z['ann']*100:.1f}%(毛)/{h1['ann']*100:.1f}%(净)，"
          f"而 rebalance/backtest 脚本因建仓日前视虚报到 {ref['ann']*100:.1f}% —— 优化方向应放在『真实 alpha + 降换手』，而非累计费率数字。")

if __name__ == "__main__":
    main()