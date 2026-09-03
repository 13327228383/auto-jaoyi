# -*- coding: utf-8 -*-
"""
backtest_defense_assets.py —— B：风险-off 防御资产再细分回测
================================================================================
当前实盘：风险-off 时全仓国债 511010。之前只对比过 防御/保留强势/空仓 三种「方式」，
却从未在「防御资产内部」比选——国债 vs 黄金 到底谁更能避跌、回场更快？

复用 backtest_minhold 的逐日模拟（min_hold=1=当前落地，HOLD_N=1，同套 S/R+跟踪6%+入场过滤，
止损仍独立于 risk_off，只在风险-off 时防御资产不同）：
  def_bond  : 风险-off 全仓 511010 国债（基线，当前实盘）
  def_gold  : 风险-off 全仓 518880 黄金
  def_cash  : 风险-off 空仓持有现金（DEFENSE_MODE=cash）
输出：全期 + 样本外(2021+)对比。
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import backtest_sr_enhance as bse
import backtest_minhold as mh

SPLIT = pd.Timestamp("2021-01-01")
MIN_HOLD = 1            # 当前落地值

VARIANTS = [
    ("def_bond 国债(基线)", {"DEFENSIVE": "511010"}),
    ("def_gold 黄金",        {"DEFENSIVE": "518880"}),
    ("def_cash 空仓",        {"DEFENSIVE": "511010", "DEFENSE_MODE": "cash"}),
]


def main():
    price = bse.load_prices()
    print(f"数据 {price.index[0].date()} ~ {price.index[-1].date()}，{len(price)} 交易日，标的 {list(price.columns)}")
    print(f"逐日决策 + MIN_HOLD={MIN_HOLD}d + S/R止损+跟踪6%+入场过滤；仅「风险-off 防御资产」不同\n")

    hdr = f"{'方案':16s}{'累计x':>8}{'年化':>8}{'回撤':>8}{'Calmar':>8}{'Sharpe':>8}{'换仓':>6}{'止损':>6}{'持仓%':>8}"
    print(hdr)
    print("-" * len(hdr))
    rows = {}
    for name, params in VARIANTS:
        r = mh.simulate_daily(price, MIN_HOLD, sr_params=params)
        rows[name] = r
        print(f"{name:16s}{r['cum']:>8.2f}{r['ann']*100:>8.2f}{r['maxdd']*100:>8.2f}"
              f"{r['calmar']:>8.2f}{r['sharpe']:>8.2f}{r['switches']:>6d}"
              f"{r['stop_exits']:>6d}{r['time_in_mkt']*100:>8.1f}")

    print("\n样本外(2021+)校验：")
    for name, _ in VARIANTS:
        eq = rows[name]["equity"].loc[SPLIT:]
        n = max(len(eq), 1)
        cum = float(eq.iloc[-1])
        ann = float(cum ** (252.0 / n) - 1) if cum > 0 else -1.0
        maxdd = float(((eq - eq.cummax()) / eq.cummax()).min())
        print(f"   {name:16s} 年化{ann*100:6.2f}%  累计{cum:5.2f}x  回撤{maxdd*100:6.2f}%")


if __name__ == "__main__":
    main()