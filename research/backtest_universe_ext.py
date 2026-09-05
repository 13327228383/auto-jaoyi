# -*- coding: utf-8 -*-
"""
backtest_universe_ext.py —— 标的池横向扩展(A) 统一数据回测
================================================================================
背景：用户要求"先继续标的池扩展"（方向 A）。原 6 只 ETF 回测用指数代理
      (backtest_sr_enhance.load_prices → cache/backtest/final_idx_*.csv)，
      原因是最早 akshare(东财) 拉不到全历史 ETF。现东财断连、但新浪 sina
      `fund_etf_hist_sina` 可拉真实 ETF 全历史 → 本次统一改用"真实 ETF 收盘价"，
      让 6 基 + 5 扩展在同一数据口径下公平对比，隔离"加标的是否真带来增益"。

扩展候选（均为 2015 前上市、跨大类、流动性良好，尽量覆盖回测全期）：
  510050 上证50(国内蓝筹) / 510880 红利(价值) / 159920 恒生(港股) /
  513500 标普500(美股宽基) / 512010 医药(行业)

对比：同一套逐日模拟(backtest_minhold.simulate_daily)，
      MIN_HOLD=1 + 黄金防御 + 防御动量门10d + S/R/跟踪止损6%，
      仅 universe 不同：
        BASE = 原 6 只            （真实ETF数据口径，对齐后基准）
        EXT  = 6+5 只(11只)        （横向扩展）
输出：全期 + 样本外(2021+) 对比表。
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

import backtest_minhold as mh
import strategy_rotator as sr

CACHE = os.path.join(ROOT, "cache", "backtest")
SPLIT = pd.Timestamp("2021-01-01")
MIN_HOLD = 1                  # 与当前落地一致
# 与当前实盘一致的核心配置
GOLD = {
    "DEFENSIVE": "518880", "DEFENSE_MODE": "defensive", "DEF_MOM_DAYS": 10,
    "GLOBAL_US": "513100", "GLOBAL_GOLD": "518880", "BROAD": "510300",
}
DEF_TRAIL = None              # 防御黄金不上跟踪止损(当前落地，动量门已接管)
DEF_MOM = 10

# 真实 ETF 代码（sina 前缀：5→sh, 1→sz）
ALL_CODES = {
    # 原 6 只
    "510300": "sh510300", "510500": "sh510500", "159915": "sz159915",
    "518880": "sh518880", "513100": "sh513100", "511010": "sh511010",
    # 扩展 5 只
    "510050": "sh510050", "510880": "sh510880", "159920": "sz159920",
    "513500": "sh513500", "512010": "sh512010",
}
EXTRA_CODE = "510050,510880,159920,513500,512010".split(",")
BASE_CODES = [c for c in ALL_CODES if c not in EXTRA_CODE]
# 当前落地 7 只 = 原 6 只 + 513500（标普500，记忆结论唯一有价值的扩展）
CUR7 = BASE_CODES + ["513500"]


def fetch_sina(code, symbol):
    """新浪真实 ETF 全历史收盘价（缓存到 etf_<code>.csv）。返回 Series。"""
    cf = os.path.join(CACHE, "etf_" + code + ".csv")
    if os.path.exists(cf):
        try:
            s = pd.read_csv(cf, dtype={"date": str}).set_index("date")["close"]
            if len(s) > 500:
                return s
        except Exception:
            pass
    import akshare as ak
    df = ak.fund_etf_hist_sina(symbol=symbol)
    df = df[["date", "close"]].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna()
    df.to_csv(cf, index=False)
    return df.set_index("date")["close"]


def load_prices():
    """全部 11 只真实 ETF 收盘价（同一数据口径）。"""
    cols = {}
    for code, sym in ALL_CODES.items():
        cols[code] = fetch_sina(code, sym)
    price = pd.DataFrame(cols).dropna()
    price.index = pd.to_datetime(price.index)
    return price.sort_index()


def fetch_open(code, symbol):
    """新浪真实 ETF 全历史开盘价（缓存到 etf_<code>_open.csv）。返回 Series。
    与 fetch_sina 同一数据源、同日期口径，供「次日开盘价成交」回测使用。"""
    cf = os.path.join(CACHE, "etf_" + code + "_open.csv")
    if os.path.exists(cf):
        try:
            s = pd.read_csv(cf, dtype={"date": str}).set_index("date")["open"]
            if len(s) > 500:
                return s
        except Exception:
            pass
    import akshare as ak
    df = ak.fund_etf_hist_sina(symbol=symbol)
    df = df[["date", "open"]].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["open"] = pd.to_numeric(df["open"], errors="coerce")
    df = df.dropna()
    df.to_csv(cf, index=False)
    return df.set_index("date")["open"]


def load_prices_open(codes=None):
    """加载真实 ETF 收盘价 + 开盘价（对齐同一公共交易日）。
    返回 (close_df, open_df)。codes=None 用全部 ALL_CODES。"""
    if codes is None:
        codes = list(ALL_CODES)
    ccols, ocols = {}, {}
    for code in codes:
        sym = ALL_CODES[code]
        ccols[code] = fetch_sina(code, sym)
        ocols[code] = fetch_open(code, sym)
    close = pd.DataFrame(ccols).dropna()
    open_ = pd.DataFrame(ocols)
    common = close.index.intersection(open_.index)
    close = close.reindex(common).sort_index()
    open_ = open_.reindex(common).sort_index()
    close.index = pd.to_datetime(close.index)
    open_.index = pd.to_datetime(open_.index)
    return close.astype(float), open_.astype(float)


def main():
    price = load_prices()
    print(f"真实ETF数据 {price.index[0].date()} ~ {price.index[-1].date()}，{len(price)} 交易日，标的 {list(price.columns)}")
    print(f"保护 = HOLD_N=1 + S/R止损 + 跟踪6% + 黄金防御 + 防御动量门{DEF_MOM}d + MIN_HOLD={MIN_HOLD}d\n")

    hdr = f"{'方案':7s}{'标的':>22s}{'累计x':>8}{'年化':>8}{'回撤':>8}{'Calmar':>8}{'Sharpe':>8}{'换仓':>6}{'止损':>6}{'持仓%':>8}"
    print(hdr)
    print("-" * len(hdr))

    rows = {}
    for name, uni in [("BASE", BASE_CODES), ("CUR7", CUR7), ("EXT", list(ALL_CODES))]:
        r = mh.simulate_daily(price, MIN_HOLD, sr_params=GOLD, def_trail_pct=DEF_TRAIL,
                              def_mom_days=DEF_MOM, universe=uni)
        rows[name] = r
        tag = "  <- 6只基准" if name == "BASE" else ("  <- 当前7只" if name == "CUR7" else "  <- +5扩展=11只")
        print(f"{name:7s}{','.join(uni):>22s}{r['cum']:>8.2f}{r['ann']*100:>8.2f}{r['maxdd']*100:>8.2f}"
              f"{r['calmar']:>8.2f}{r['sharpe']:>8.2f}{r['switches']:>6d}"
              f"{r['stop_exits']:>6d}{r['time_in_mkt']*100:>8.1f}{tag}")

    for name in rows:
        so = mh.slice_metrics(rows[name], SPLIT)
        if so:
            print(f"\n样本外(2021+) {name}: 年化{so[0]*100:6.2f}%  累计{so[1]:5.2f}x  回撤{so[2]*100:6.2f}%")
        else:
            print(f"\n样本外(2021+) {name}: 数据不足")

    b, c, x = rows["BASE"], rows["CUR7"], rows["EXT"]
    print("\n=== 结论：当前7只 vs 进一步扩展（真实ETF统一口径，全期） ===")
    print(f"  6只以外再扩展的累计增量:")
    print(f"    6→ 7(+513500): 年化 {b['ann']*100:.2f}% → {c['ann']*100:.2f}% ({(c['ann']-b['ann'])*100:+.2f}pp)  "
          f"回撤 {b['maxdd']*100:.2f}% → {c['maxdd']*100:.2f}%  换仓 {b['switches']}→{c['switches']}")
    print(f"    7→11(+4只):   年化 {c['ann']*100:.2f}% → {x['ann']*100:.2f}% ({(x['ann']-c['ann'])*100:+.2f}pp)  "
          f"回撤 {c['maxdd']*100:.2f}% → {x['maxdd']*100:.2f}%  换仓 {c['switches']}→{x['switches']}")
    for name in ("BASE", "CUR7", "EXT"):
        so = mh.slice_metrics(rows[name], SPLIT)
        if so:
            print(f"  样本外(2021+) {name}: 年化 {so[0]*100:.2f}%  回撤 {so[2]*100:.2f}%")


if __name__ == "__main__":
    main()