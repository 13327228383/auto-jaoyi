# -*- coding: utf-8 -*-
"""
backtest_daily_close.py —— 验证「日收盘决策、盘中不即时砍仓」vs「盘中风险-off 翻转日即时防守」
================================================================================
背景：auto_run 每 ~120s 心跳重估，且"风险-off 新触发"会立即执行防守切换（可突破5天锁）。
本回测在日线粒度测：该"即时防守"相对"仅在双周决策点行动"是提收益还是增加摩擦。

对比触发时机（同一套 decide / 同一套防御=全仓国债）：
  M0  仅双周决策（现状回测口径≈"日收盘/定时行动"，即拟采用的保守方案）
  M1a 双周 + 风险-off 翻转日(off→on)即时切防御（模拟现盘"立即防守切换"，无退抖）
  M1b 双周 + 风险-off 翻转日即时切防御 + 连续2天确认退抖
指标：累计/年化/回撤/实际换手次数(=摩擦)/持仓占比。成本0.1%/次，口径与基线一致。
说明：数据为日收盘价(close)，无法区分"盘中低点 vs 收盘"的执行价；此处测的是触发时点差异，
      硬止损的"盘中 vs 收盘"不可仅凭 close 判定，属本测覆盖不到的部分。
"""
import os, sys, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import strategy_rotator as sr
sys.path.insert(0, HERE)
from backtest_sr_enhance import CACHE, FREQ, load_prices, simulate

BASE_PARAMS = {
    "GLOBAL_US": "513100", "GLOBAL_GOLD": "518880", "BROAD": "510300",
    "DEFENSIVE": sr.DEFENSIVE, "ETF_UNIVERSE": sr.ETF_UNIVERSE,
}


def state_at(state, d):
    """取日期 d 的风险状态：优先当日，否则回溯最近一日（状态向前延续，避免非交易日 KeyError）。"""
    if state is None:
        return None
    if d in state.index:
        return bool(state.loc[d])
    before = state[state.index <= d]
    return bool(before.iloc[-1]) if len(before) else False


def decide_at_date(price, d, params, risk_state):
    hist = price.loc[:d]
    if len(hist) < 200:
        return "cash"
    prices = {c: hist[c].values for c in price.columns}
    rs = state_at(risk_state, d)
    return sr.decide(prices, params, risk_state=rs)["target"]


def build_dec_dates(price, mode, risk_state=None):
    """返回 (dec_dates 递增列表)。"""
    base = [d for d in price.resample(FREQ).last().index if len(price.loc[:d]) >= 200]
    if mode == "M0":
        return sorted(set(base))
    # M1a/M1b：双周 + 风险-off 翻转日(False->True)，只保留数据充足的交易日
    if risk_state is None:
        return sorted(set(base))
    flip_days = []
    prev = False
    for ts in risk_state.index:
        if ts not in price.index:
            continue
        cur = bool(risk_state.loc[ts])
        if len(price.loc[:ts]) >= 200 and cur and not prev:
            flip_days.append(ts)
        prev = cur
    return sorted(set(list(base) + flip_days))


def main():
    price = load_prices()
    print(f"数据区间 {price.index[0].date()} ~ {price.index[-1].date()}")
    pmap = {c: price[c] for c in price.columns}

    # 原始/确认 风险状态轴（统一对齐到 trading index，避免节假日/对齐差异）
    raw_state = sr.risk_off_raw_axis(pmap, BASE_PARAMS).reindex(price.index).fillna(False).astype(bool)
    conf_state = sr.confirmed_risk_off_axis(pmap, dict(BASE_PARAMS,
                                                       RISK_CONFIRM_DAYS=2, RISK_EXIT_DAYS=2)
                                            ).reindex(price.index).fillna(False).astype(bool)

    variants = [
        ("M0 仅双周决策(保守日收盘)",       "M0", None),
        ("M1a 双周+翻转日即时防守",         "M1", raw_state),
        ("M1b 双周+翻转日即时防守+退抖2天", "M1", conf_state),
    ]
    print(f"\n{'方案':34s} {'累计':>7s} {'年化':>7s} {'回撤':>8s} {'换手':>5s} {'持仓占比':>8s} {'决策点':>6s}")
    results = {}
    for name, mode, state in variants:
        dec_dates = build_dec_dates(price, mode, state)
        targets = {d: decide_at_date(price, d, BASE_PARAMS, state) for d in dec_dates}
        r = simulate(price, sorted(dec_dates), targets, {})
        results[name] = r
        print(f"{name:34s} {r['cum']:7.3f}x {r['ann']*100:6.2f}% {r['maxdd']*100:7.2f}% "
              f"{r['switches']:5d} {r['time_in_mkt']*100:7.1f}% {len(dec_dates):6d}")

    raw_on = int(raw_state.sum()); conf_on = int(conf_state.sum())
    print(f"\n风险-off 原始信号 {raw_on} 日；连续2天确认后 {conf_on} 日（削掉 {raw_on-conf_on} 日抖动）")

    out = {name: {k: v for k, v in results[name].items() if k not in ('equity', 'port', 'adj')}
           for name, _, _ in variants}
    with open(os.path.join(CACHE, "daily_close_result.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()