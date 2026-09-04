# -*- coding: utf-8 -*-
"""
big_universe_test.py —— 「更多标的=更稳?」大池 vs 7只 诚实回测
================================================================================
回答用户："7只难免是个例，扩到50只/100只是不是更稳定(更能赚钱)"。

诚实口径修正的关键：回测要可比，标的的历史必须足够长。多数跨市场/商品/行业ETF
2019年才上市 —— 若把它们全进场再 dropna，整体窗口会被拖到 2019+，样本太短、
OOS 失效、与 7 只不可比。故：
  1) 尽可能纳入「上市足够早(≤2018)」的跨类别 ETF（覆盖 A股宽基/风格/行业、港股、
     美股、黄金、债券、货币等）。
  2) 标注：哪些候选因上市晚(2019+)无法公平验证 —— 这正是"50~100只"在历史上
     根本凑不齐/验不了的直接证据。
  3) 在【公共可验证窗口】内，用同一套 simulate_daily（SW20/trail3%/hold1/real_t1
     为 7 只最优参数，防御动量门10d），对比 7只 vs 大池 的 年化/回撤/换手/OOS。

指标口径与 backtest_universe_ext 一致，保证可比、不另行美化。
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CACHE = os.path.join(ROOT, "cache", "backtest")
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import backtest_universe_ext as bue
import backtest_minhold as mh

SPLIT = pd.Timestamp("2021-01-01")
BASE7 = ["510300", "510500", "159915", "518880", "513100", "511010", "513500"]
# 兜底公共窗口起点——再晚就不认为"early enough"可公平验证
MAX_START = "2018-01-01"

# 跨类别大池候选：代码 -> 新浪symbol。全部为「我确信映射正确」的老牌高流动性ETF。
BIG_ALL = {
    # --- A股宽基/风格 ---
    "510050": "sh510050",   # 上证50
    "510300": "sh510300",   # 沪深300
    "510500": "sh510500",   # 中证500
    "159915": "sz159915",   # 创业板
    "510880": "sh510880",   # 红利
    "510900": "sh510900",   # 恒生H股ETF
    "159901": "sz159901",   # 深100
    "159902": "sz159902",   # 中小板
    # --- 行业(早年上市) ---
    "510230": "sh510230",   # 金融
    "512010": "sh512010",   # 医药
    # --- 跨境 ---
    "513100": "sh513100",   # 纳指
    "513500": "sh513500",   # 标普500
    "159920": "sz159920",   # 恒生
    "159941": "sz159941",   # 纳斯达克100
    # --- 商品 / 债券 / 货币 ---
    "518880": "sh518880",   # 黄金
    "518800": "sh518800",   # 黄金(暂不确认则跳过)
    "511010": "sh511010",   # 国债
    "511260": "sh511260",   # 十年国债ETF
    "511880": "sh511880",   # 银华日利(货币)
}
# 境内股票ETF = T+1；跨境/债/货币/商品 = T+0
T1_BIG = {"510050", "510300", "510500", "159915", "510880", "510900",
          "159901", "159902", "510230", "512010"}
T1_BASE = {"510300", "510500", "159915"}


def fetch(code, symbol):
    """拉真实全历史收盘(带缓存)，统一 datetime 索引，返回 (Series, 起点str)。"""
    try:
        s = bue.fetch_sina(code, symbol)
        s = s.copy()
        s.index = pd.to_datetime(s.index)
        return s, str(s.index.min().date())
    except Exception as e:
        return None, f"ERR:{e}"


def main():
    print("== 第一步：逐候选拉真实数据并记录上市起点 ==", flush=True)
    rows = []
    for code, sym in BIG_ALL.items():
        s, start = fetch(code, sym)
        if s is None or isinstance(start, str) and start.startswith("ERR"):
            print(f"  {code} 拉取失败/跳过: {start}", flush=True)
            rows.append((code, None, None))
            continue
        rows.append((code, s, start))
        print(f"  {code}  {sym}  起点 {start}  共{len(s)}日", flush=True)

    # 上市晚说明（诚实呈现")
    late = [code for code, s, start in rows
            if s is not None and start and start > MAX_START]
    included = [r for r in rows if r[2] is not None and not str(r[2]).startswith("ERR")
                and r[2] <= MAX_START]
    print(f"\n上市晚(>{MAX_START})、无法公平验证者：{late or '无'}")
    print(f"纳入公共池 {len(included)} 只：{[c for c,_,_ in included]}")

    if len(included) < len(BASE7):
        print("纳入池子不足7只，无法对比 —— 说明早期可验证的跨类ETF少于7只。")
        return

    # 公共窗口起点 = 纳入池里最晚的起点 (公平对齐)
    starts = [s.index.min() for _, s, _ in included]
    common_start = max(starts)
    if common_start > pd.Timestamp(MAX_START):
        print(f"公共起点 {common_start.date()} 晚于容限，样本太短，放弃。")
        return

    # 构造大池价格表（各列自身起点≤common_start，dropna 对齐从 common_start 起）
    cols = {code: s for code, s, _ in included}
    big = pd.DataFrame(cols)
    big = big[big.index >= common_start]
    big = big.dropna().sort_index()

    # 7只基准用同一窗口（公平）——simulate 直接用 big 价格表即可
    price = big

    print(f"\n== 公共可验证窗口: {price.index[0].date()} ~ {price.index[-1].date()}，{len(price)}交易日 ==")
    print(f"大池 {len(price.columns)} 只 vs 7只基准（同一窗口），参数固定 "
          f"SW=20 / trail=3% / hold=1d / real_t1 / 防御动量门10d")

    GOLD = dict(bue.GOLD)   # 用扩展脚本同一GOLD，避免口径漂移
    GOLD["SLOPE_WINDOW"] = 20
    GOLD["MOM_WINDOW"] = 20

    hdr = f"{'方案':>5}{'标的数':>5}{'年化%':>8}{'回撤%':>8}{'Calmar':>7}{'Sharpe':>7}{'换仓':>5}{'止损':>5}{'持仓%':>7}{'OOS年化%':>9}"
    print(hdr)
    print("-" * len(hdr))
    results = {}
    for name, uni, t1 in [
        ("BASE7", BASE7, T1_BASE),
        ("BIG", sorted(price.columns), T1_BIG & set(price.columns)),
    ]:
        r = mh.simulate_daily(price, 1, trail_pct=0.03, sr_params=GOLD,
                              def_trail_pct=None, def_mom_days=10,
                              universe=list(uni), t1_codes=set(t1))
        results[name] = r
        so = mh.slice_metrics(r, SPLIT)
        print(f"{name:>5}{len(uni):>5}{r['ann']*100:>8.2f}{r['maxdd']*100:>8.2f}"
              f"{r['calmar']:>7.2f}{r['sharpe']:>7.2f}{r['switches']:>5}"
              f"{r['stop_exits']:>5}{r['time_in_mkt']*100:>7.1f}"
              f"{(so[0]*100 if so else float('nan')):>9.2f}")

    b, x = results["BASE7"], results["BIG"]
    print(f"\n=== 结论：更多标的是否更稳/更赚 ===")
    print(f"  年化: {x['ann']*100:+.2f}pp  (7只 {b['ann']*100:.2f}% → 大池 {x['ann']*100:.2f}%)")
    print(f"  回撤: {x['maxdd']*100:+.2f}pp  (7只 {b['maxdd']*100:.2f}% → 大池 {x['maxdd']*100:.2f}%)")
    print(f"  Calmar: {b['calmar']:.2f} → {x['calmar']:.2f}")
    print(f"  换仓: 7只 {b['switches']}次 → 大池 {x['switches']}次")
    print(f"\n  ★ 注意：真正在2019年前上市、可公平验证的跨类ETF仅 {len(price.columns)} 只；")
    print(f"    想凑到50/100只，绝大多数是2019+次新，历史样本太短，无法回测验证'更稳'。")


if __name__ == "__main__":
    main()