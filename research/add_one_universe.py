# -*- coding: utf-8 -*-
"""add-one 隔离：BASE6 + 单只扩展，看新增哪只贡献/拖累。"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
import backtest_universe_ext as bue
import backtest_minhold as mh

p = bue.load_prices()
BASE = bue.BASE_CODES
EXTRA = bue.EXTRA_CODE

print(f"{'方案':18s}{'全期年化':>9}{'全期回撤':>9}{'全期Calmar':>11}{'OOS年化':>9}{'OOS回撤':>9}{'换仓':>6}")
print("-" * 68)

def row(tag, uni):
    r = mh.simulate_daily(p, 1, sr_params=bue.GOLD, def_trail_pct=None, def_mom_days=10, universe=uni)
    so = mh.slice_metrics(r, mh.SPLIT)
    oa = so[0]*100 if so else float('nan')
    od = so[2]*100 if so else float('nan')
    print(f"{tag:18s}{r['ann']*100:>9.2f}{r['maxdd']*100:>9.2f}{r['calmar']:>11.2f}{oa:>9.2f}{od:>9.2f}{r['switches']:>6d}")
    return r

row("BASE(6只)", BASE)
for c in EXTRA:
    row(f"BASE+{c}", BASE + [c])
row("EXT(全11只)", list(bue.ALL_CODES))
# 精选组合（剔除拖累的 159920/512010）
row("BASE+513500", BASE + ["513500"])
row("BASE+513500+510880", BASE + ["513500", "510880"])
row("BASE+513500+510880+510050", BASE + ["513500", "510880", "510050"])