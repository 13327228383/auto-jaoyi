# -*- coding: utf-8 -*-
"""A) 拉取候选扩展标的的真实 ETF 历史，存 cache/backtest/etf_<code>.csv。
候选均为 2013 前上市、跨大类、低流动风险，尽量覆盖 2016-12 起的回测全期。"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import backtest_rotation as br

# 候选扩展（主线）：510050 上证50(国内蓝筹) / 510880 红利(价值) /
# 159920 恒生(港股) / 513500 标普500(美股宽基) / 512010 医药(行业)
EXTRA = ["510050", "510880", "159920", "513500", "512010"]

START = "2016-06-01"          # 略早于回测起点，含建仓前 250 日历史

if __name__ == "__main__":
    for code in EXTRA:
        s = br.fetch_close(code, START, br.END)
        if len(s):
            print(f"{code}: {s.index.min().date()} ~ {s.index.max().date()}  n={len(s)}  last={float(s.iloc[-1]):.3f}")
        else:
            print(f"{code}: 拉取失败/空")