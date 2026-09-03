# -*- coding: utf-8 -*-
"""
backtest_def_sz_gold.py —— 防御资产深市黄金 159934 vs 现行 518880 对比回测
================================================================================
背景：风险-off 时当前实盘全仓 518880(沪市黄金ETF)。用户关注"有没有深市的ETF、要不要回测"。
     深市同类防御资产是 159934(易方达黄金ETF, sz)。两者都跟踪黄金，这里做"防御资产换深市黄金
     是否更优"的公平对比。

口径：与 backtest_defense_assets.py 相同的逐日模拟（MIN_HOLD=1，HOLD_N=1，
     S/R止损 + 跟踪6% + 入场过滤），仅「风险-off 防御资产」不同。数据用真实 ETF
     收盘价（backtest_universe_ext 口径），基池 = 当前实盘 7 只（原6只 + 513500）。

方案：
  def_sh518880 : 风险-off 全仓 518880（现行，沪市黄金）
  def_sz159934 : 风险-off 全仓 159934（深市黄金）
输出：全期 + 样本外(2021+) 对比。
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import backtest_universe_ext as bue
import backtest_minhold as mh

SPLIT = pd.Timestamp("2021-01-01")
MIN_HOLD = 1            # 当前落地值
UNIVERSE = bue.BASE_CODES + ["513500"]   # 当前实盘标的池 = 原6只 + 标普500

VARIANTS = [
    ("def_sh518880 沪金(现行)", {"DEFENSIVE": "518880"}),
    ("def_sz159934 深金",       {"DEFENSIVE": "159934"}),
]


def load_prices8():
    """7 只实盘基池 + 159934 防御候选 = 8 只真实ETF收盘价。"""
    p = bue.load_prices()
    gold = bue.fetch_sina("159934", "sz159934").copy()
    gold.index = pd.to_datetime(gold.index)
    return p.join(gold.rename("159934"), how="left").sort_index()


def main():
    price = load_prices8()
    print(f"数据 {price.index[0].date()} ~ {price.index[-1].date()}，{len(price)} 交易日，标的 {list(price.columns)}")
    print(f"逐日决策 + MIN_HOLD={MIN_HOLD}d + S/R止损+跟踪6%+入场过滤；仅「风险-off 防御资产」不同\n")

    hdr = f"{'方案':19s}{'累计x':>8}{'年化':>8}{'回撤':>8}{'Calmar':>8}{'Sharpe':>8}{'换仓':>6}{'止损':>6}{'持仓%':>8}"
    print(hdr)
    print("-" * len(hdr))
    rows = {}
    for name, params in VARIANTS:
        r = mh.simulate_daily(price, MIN_HOLD, sr_params=params, universe=list(UNIVERSE))
        rows[name] = r
        print(f"{name:19s}{r['cum']:>8.2f}{r['ann']*100:>8.2f}{r['maxdd']*100:>8.2f}"
              f"{r['calmar']:>8.2f}{r['sharpe']:>8.2f}{r['switches']:>6d}"
              f"{r['stop_exits']:>6d}{r['time_in_mkt']*100:>8.1f}")

    print("\n样本外(2021+)校验：")
    for name, _ in VARIANTS:
        eq = rows[name]["equity"].loc[SPLIT:]
        n = max(len(eq), 1)
        cum = float(eq.iloc[-1])
        ann = float(cum ** (252.0 / n) - 1) if cum > 0 else -1.0
        maxdd = float(((eq - eq.cummax()) / eq.cummax()).min())
        print(f"   {name:19s} 年化{ann*100:6.2f}%  累计{cum:5.2f}x  回撤{maxdd*100:6.2f}%")


if __name__ == "__main__":
    main()