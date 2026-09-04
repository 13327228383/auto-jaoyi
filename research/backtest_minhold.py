# -*- coding: utf-8 -*-
"""
backtest_minhold.py —— 「最小调仓间隔(5天持仓锁)」回测：会错过新强势ETF吗？
================================================================================
实盘(auto_run.py)是「每日14:50决策 + MIN_HOLD_DAYS=5 自然日持仓锁」：
  仅当 目标变化 且 距上次主动换仓>=5天 才换；止损触发当日解锁可立即再配置。
本脚本还原该实盘逻辑做逐日模拟，对比不同最小间隔(自然日)：
  MIN_HOLD ∈ {0,1,3,5,7,10,14}
  - 0  = 无锁（新强势一到当天就换）→ 理论上限
  - 5  = 当前实盘值
  若 MIN_HOLD=5 相对 0 收益接近、操作明显更少 → 5天锁不会因“等”而错过新强势，
  反而省了换手费；若 5 明显更差 → 锁确实挡住了该换的票，需缩到更小。

逐日模拟保留实盘同一套保护：HOLD_N=1 + S/R动态止损 + 跟踪止损6% + 入场过滤。
止损独立于 min_hold 每日判断（保护性砍仓不限）；min_hold 只约束“主动换仓”。
输出：全期 + 样本外(2021+)对比表。
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import backtest_sr_enhance as bse

SPLIT = pd.Timestamp("2021-01-01")
TRAIL_PCT = 0.06           # 跟踪止损 6%（与当前落地一致）
MIN_HOLDS = [0, 1, 3, 5, 7, 10, 14]     # 最小调仓间隔（自然日）
TAIL = 250                 # 每次决策传入的历史长度（够 MA200/slope60/风险off）


def day_target(price, d, sr_params=None, universe=None):
    """取 d 日收盘后决策目标(+入场过滤交由 simulate 执行)。HOLD_N=1(默认)。
    sr_params 可覆盖 DEFENSIVE / DEFENSE_MODE / 等（供防御资产/防御方式对比）。
    universe: 参与轮动的标的集合（扩展标的池用，None=ETF_UNIVERSE 6 只）。"""
    hist = price.loc[:d]
    if len(hist) < 200:
        return "cash", {}
    tail = hist.tail(TAIL)
    prices = {c: tail[c].values for c in tail.columns}
    res = bse.sr.decide(prices, sr_params, universe=universe)
    return res["target"], res.get("scores", {})


def simulate_daily(price, min_hold, trail_pct=TRAIL_PCT, sr_params=None, def_trail_pct=None,
                   def_mom_days=None, universe=None, invest_ratio=1.0,
                   t1_codes=None):
    """逐日模拟：每日决策 + 最小调仓间隔 min_hold(自然日)。返回指标 dict。
    def_trail_pct: 防御资产(如黄金)的跟踪止损百分比。为 None 时防御资产不设止损(原版)；
                   传数值则为防御资产也启用跟踪止损(用于压缩回撤，见 backtest_def_stop.py)。
    def_mom_days: 防御资产自身动量门。>0 时要求防御资产 N 日动量>0 才持有，
                  否则空仓等待(不会次日因 risk-off 仍触发而立即买回)——真正压缩防御资产下跌回撤。
    invest_ratio: 持仓日投入资金比例(0,1]。<1 模拟"多余闲置现金不动、只投一部分进目标"（仅投入
                  该比例，剩余现金闲置=0收益）——用于量化"买(满仓) vs 不买(闲置现金)"的差异。
    t1_codes: T+1 标的集合。这些标的当日触发回撤止损时【当日无法卖出】，延迟到次一交易日
              才离场（承受次日继续下探的代价）——真实反映 A股股票ETF (沪深300/中证500/创业板)
              的 T+1 卖约束。None=全部按 T+0 当日可离场（当前生产脚本口径）。"""
    dates = price.index
    daily_ret = price.pct_change().fillna(0.0)
    port = pd.Series(0.0, index=dates)
    adj = pd.Series(1.0, index=dates)
    switches = 0
    stop_exits = 0
    invested = 0
    def_code = str((sr_params or {}).get("DEFENSIVE", bse.DEF))

    held = set()
    entry, stop, peak = {}, {}, {}
    pending_sell = set()        # T+1 标的今日触发止损、但当日卖不了 -> 明日才清仓
    last_switch = None          # 上次「主动换仓」日期；止损清空→立即解锁
    cur_key = None
    entered = False             # 是否已建过仓（首次入场不另收换仓费）

    for i, d in enumerate(dates):
        if i == 0:
            continue
        rebuild_override = None      # 建仓日应记「旧持仓当日涨幅」，而非新标的当日涨幅（消除前视）
        stop_today = False           # 今日是否触发止损（用于：停损→再入 同样计双边成本）

        # ---- 1) 持仓内部：S/R/跟踪止损（独立于 min_hold） ----
        # 先执行 T+1 标的「昨日触发、今日可卖」的离场（今日开盘价成交，承受前日触发口径）
        if pending_sell:
            changed = False
            for c in list(pending_sell):
                if c in held:
                    px = price.loc[d, c]
                    # 以今日执行离场：这是T+1的真实代价（比昨日触发晚1日）
                    held.discard(c)
                    entry.pop(c, None)
                    stop.pop(c, None)
                    peak.pop(c, None)
                    stop_exits += 1
                    stop_today = True
                    changed = True
            pending_sell = set()
            if changed:
                last_switch = None     # 止损当日解锁：可立即再配置（对应实盘 _stop_today）
        if held:
            changed = False
            for c in list(held):
                px = price.loc[d, c]
                # 防御资产：仅当显式给了 def_trail_pct 才启用跟踪止损；否则豁免(原版)
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
                    if t1_codes and c in t1_codes:
                        # T+1: 当日触发但卖不了，记 pending 次日离场（不 discard）
                        pending_sell.add(c)
                        changed = True
                    else:
                        held.discard(c)
                        entry.pop(c, None)
                        stop.pop(c, None)
                        peak.pop(c, None)
                        stop_exits += 1
                        stop_today = True
                        changed = True
            if changed:
                last_switch = None     # 止损当日解锁：可立即再配置（对应实盘 _stop_today）

        # ---- 2) 每日决策（受 min_hold 约束主动换仓） ----
        can = (not held) or (last_switch is None) or ((d - last_switch).days >= min_hold)
        if can:
            tgt, _ = day_target(price, d, sr_params, universe=universe)
            # 防御动量门：防御资产自身 N 日动量转负 → 视同空仓等待（不买回，压缩防御回撤）
            if def_mom_days and isinstance(tgt, (list, tuple)) and tgt and str(tgt[0]) == def_code:
                dm = bse.sr.momentum(pd.Series(price.loc[:d, def_code]), def_mom_days)
                if dm is None or np.isnan(dm) or dm <= 0:
                    tgt = "cash"
            key = "cash" if tgt == "cash" else tuple(sorted(str(c) for c in tgt))

            if entered and key != cur_key:      # 主动换仓：收手续费、计次数、重置间隔
                switches += 1
                adj.loc[dates >= d] *= (1 - bse.FEE)
                last_switch = d
            elif not entered:
                entered = True

            # 仅当 空仓 或 目标变化 时才重建持仓（避免重置跟踪止损/止损位）
            if (not held) or (key != cur_key):
                # 建仓日当天，收益应记「此刻仍持有的旧持仓」当日涨幅(你吃到的那段)，
                # 而不是新标的当日涨幅(d-1→d)——否则即前视，会白拿新强票确认日跳涨。
                if held:
                    w = 1.0 / len(held)
                    rebuild_override = invest_ratio * sum(w * daily_ret.loc[d, c] for c in held)
                else:
                    rebuild_override = 0.0
                # 停损→再入：实盘是「先卖止损仓、再买入」，两头都收佣金 → 计一次双边
                if stop_today and tgt != "cash":
                    adj.loc[dates >= d] *= (1 - bse.ROUND_TRIP)
                held = set()
                entry, stop, peak = {}, {}, {}
                if tgt != "cash":
                    for c in tgt:
                        c = str(c)
                        px = price.loc[d, c]
                        if c == def_code:
                            # 防御资产：不给 S/R 止损；仅当启用防御跟踪止损时初始化 peak
                            held.add(c); entry[c] = px; stop[c] = None
                            peak[c] = px if def_trail_pct is not None else None
                            continue
                        nsup, nres = bse.compute_sr(price.loc[:d, c], px)
                        if nres is not None and px >= nres * (1 - bse.ENTRY_BUF):
                            continue                       # 入场过滤：贴强阻力不进
                        sp = bse.sr_stop_price(px, nsup)
                        held.add(c); entry[c] = px; stop[c] = sp; peak[c] = px
                cur_key = key if held or key == "cash" else "cash"

        # ---- 3) 当日收益 ----
        if held:
            w = 1.0 / len(held)
            if rebuild_override is not None:
                port.loc[d] = rebuild_override       # 建仓日：记旧持仓走势，而非新标的
            else:
                port.loc[d] = invest_ratio * sum(w * daily_ret.loc[d, c] for c in held)
            invested += 1
        else:
            port.loc[d] = 0.0

    equity = (1 + port).cumprod() * adj
    n = max(len(equity), 1)
    cum = float(equity.iloc[-1])
    ann = float(cum ** (252.0 / n) - 1) if cum > 0 else -1.0
    maxdd = float(((equity - equity.cummax()) / equity.cummax()).min())
    rr = equity.pct_change().dropna()
    sd = float(rr.std())
    sharpe = float(rr.mean() / sd * np.sqrt(252)) if sd > 0 else float("nan")
    calmar = ann / abs(maxdd) if maxdd < 0 else float("nan")
    return {"equity": equity, "cum": cum, "ann": ann, "maxdd": maxdd,
            "sharpe": sharpe, "calmar": calmar, "switches": switches,
            "stop_exits": stop_exits, "time_in_mkt": invested / max(n - 1, 1)}


def slice_metrics(r, start):
    eq = r["equity"].loc[start:]
    if len(eq) < 2:
        return None
    n = max(len(eq), 1)
    cum = float(eq.iloc[-1])
    ann = float(cum ** (252.0 / n) - 1) if cum > 0 else -1.0
    maxdd = float(((eq - eq.cummax()) / eq.cummax()).min())
    return ann, cum, maxdd


def main():
    price = bse.load_prices()
    print(f"数据 {price.index[0].date()} ~ {price.index[-1].date()}，{len(price)} 交易日，标的 {list(price.columns)}")
    print(f"保护 = HOLD_N=1 + S/R动态止损 + 跟踪止损{TRAIL_PCT*100:.0f}% + 入场过滤；止损当日解锁\n")
    print("逐日决策 + 最小调仓间隔(自然日) 对比：")
    hdr = f"{'MIN_HOLD':>9s}{'累计x':>8}{'年化':>8}{'回撤':>8}{'Calmar':>8}{'Sharpe':>8}{'换仓':>6}{'止损':>6}{'持仓%':>8}"
    print(hdr)
    print("-" * len(hdr))

    rows = {}
    for m in MIN_HOLDS:
        r = simulate_daily(price, m)
        rows[m] = r
        tag = "  <- 当前实盘" if m == 5 else ""
        print(f"{m:>7d}d {r['cum']:>8.2f}{r['ann']*100:>8.2f}{r['maxdd']*100:>8.2f}"
              f"{r['calmar']:>8.2f}{r['sharpe']:>8.2f}{r['switches']:>6d}"
              f"{r['stop_exits']:>6d}{r['time_in_mkt']*100:>8.1f}{tag}")

    print("\n样本外(2021+)校验：")
    for m in MIN_HOLDS:
        so = slice_metrics(rows[m], SPLIT)
        if so:
            tag = "  <- 当前实盘" if m == 5 else ""
            print(f"   MIN_HOLD={m:>2d}d: 年化{so[0]*100:6.2f}%  累计{so[1]:5.2f}x  回撤{so[2]*100:6.2f}%{tag}")
        else:
            print(f"   MIN_HOLD={m:>2d}d: 样本外数据不足")

    # 结论
    b = rows[5]
    z = rows[0]
    print("\n=== 关键对比：当前5天锁 vs 无锁(0天) ===")
    print(f"  5天: 年化{b['ann']*100:.2f}%  回撤{b['maxdd']*100:.2f}%  换仓{b['switches']}次")
    print(f"  0天: 年化{z['ann']*100:.2f}%  回撤{z['maxdd']*100:.2f}%  换仓{z['switches']}次")
    print(f"  年化差 = {(b['ann']-z['ann'])*100:+.2f}pp  |  操作省 = {z['switches']-b['switches']} 次")


if __name__ == "__main__":
    main()