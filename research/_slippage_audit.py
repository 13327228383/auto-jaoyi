# -*- coding: utf-8 -*-
"""_slippage_audit.py —— 用实盘真实成交流水( journal )校核 0.1% 滑点假设
================================================================================
口径说明(诚实)：
  A. '盘中对手价 vs 当日日线收盘' 的偏移 = 盘中→收盘的时间漂移 + 真实滑点，混在一起，
     只能给数量级参考，不能当纯滑点。
  B. '相邻 buy→sell 对手价价差'：同标的相邻一买一卖(短时间)的对手价之差 ≈ 报价价差上限，
     这更接近"真实滑点"的量级(买用卖一、卖用买一，落点之间的宽度)。
样本仅15条(2天、基本是 518880)，结论只能作量级观测 + 判定 0.1% 是否偏保守/偏乐观。
"""
import os, sys, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
JOURNAL = os.path.join(ROOT, "cache", "trade_journal.jsonl")
CACHE = os.path.join(ROOT, "cache", "backtest")


def load_actions():
    acts = []
    if not os.path.exists(JOURNAL):
        return acts
    with open(JOURNAL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            acts.append(r)
    return acts


def close_series(code):
    fn = os.path.join(CACHE, f"etf_{code}.csv")
    if not os.path.exists(fn):
        return None
    df = pd.read_csv(fn, dtype={"date": str})
    df = df.set_index("date")
    return df["close"].astype(float)


def main():
    acts = load_actions()
    print(f"真实成交流水：{len(acts)} 条")
    if not acts:
        print("无记录，无法校准。"); return
    ts = sorted(a["ts"] for a in acts)
    codes = sorted({a["code"] for a in acts})
    print(f"时间范围 {ts[0]} ~ {ts[-1]}；标的 {codes}")

    # ---- A) 每笔盘中对手价 vs 当日该标的日线收盘 的偏离 ----
    rows = []
    for a in acts:
        cs = close_series(a["code"])
        if cs is None:
            continue
        d = a["ts"][:10]
        if d not in cs.index:
            continue
        close = float(cs.loc[d])
        rows.append({"side": a["side"], "bias_pct": (float(a["price"]) / close - 1) * 100})
    if rows:
        df = pd.DataFrame(rows)
        print("\n== A) 盘中对手价 vs 当日收盘 偏离(%)= (含盘中趋势+滑点，量级参考) ==")
        for side in ["buy", "sell"]:
            s = df[df["side"] == side]["bias_pct"]
            if len(s):
                print(f"  {side:5s} n={len(s):2d}  均值{np.mean(s):+6.3f}% 中位{np.median(s):+6.3f}% "
                      f"min{np.min(s):+6.3f}% max{np.max(s):+6.3f}%")

    # ---- B) 相邻 buy→sell 对手价价差（≈真实滑点量级） ----
    pads = []
    acts2 = sorted(acts, key=lambda x: (x["code"], x["ts"]))
    for i in range(len(acts2) - 1):
        a, b = acts2[i], acts2[i + 1]
        if a["side"] == "buy" and b["side"] == "sell" and a["code"] == b["code"]:
            spread = (float(a["price"]) - float(b["price"])) / float(a["price"]) * 100
            pads.append({"code": a["code"], "spread_pct": spread, "pair": f"{a['ts'][11:]}买→{b['ts'][11:]}卖",
                         "px_buy": a["price"], "px_sell": b["price"]})
    if pads:
        pd_ = pd.DataFrame(pads)
        print("\n== B) 相邻 买(卖一)→卖(买一) 对手价价差(%)= 更接近真实滑点 ==")
        for _, r in pd_.iterrows():
            print(f"  {r['code']} {r['pair']} 买价{r['px_buy']:.3f} 卖价{r['px_sell']:.3f} "
                  f"价差{r['spread_pct']:+.3f}%")
        print(f"  → {len(pd_)} 对，价差 中位{np.median(pd_['spread_pct']):+.3f}% 平均{np.mean(pd_['spread_pct']):+.3f}%")
    else:
        print("\n== B) 无相邻 买→卖 配对，无法直接估报价价差 ==")

    print("\n结论性提示：样本极少(几天、单一标的)，不能就此定滑点数值；"
          "0.1% 作为稳妥上界。真正校准需积累 券商交割单逐笔 对账 .")


if __name__ == "__main__":
    main()