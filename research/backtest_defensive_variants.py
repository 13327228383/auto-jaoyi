# -*- coding: utf-8 -*-
"""
backtest_defensive_variants.py —— 风险-off「退抖 + 防御模式」改造的前后回测对比
================================================================================
数据：cache/backtest/final_idx_*.csv（同 backtest_sr_enhance，2016-2026 全历史）
对比（同一套数据 / 同一套选股 decide()，仅风险-off 处理方式不同）：
  V0       原版   ：风险-off 立即触发，全仓国债(511010)
  C2-防御  退抖   ：风险-off 需连续 2 天才触发，解除需连续 2 天；触发后仍全仓国债
  C2+趋势  退抖+保留强势档：触发后「国债 + 保留最强正动量风险档」，不全仓锁死
  C2+空仓  退抖+空仓等待   ：风险-off 确认后不持国债，直接空仓等待机会
  C3+空仓  更强退抖(3天)+空仓  ：敏感性
指标：累计/年化/回撤/换仓次数(=摩擦)/持仓时间占比。成本 0.1%/次，口径与基线一致。
"""
import os, sys, json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import strategy_rotator as sr
sys.path.insert(0, HERE)  # 使 backtest_sr_enhance 的 reset/resample 等相对导入正常工作
from backtest_sr_enhance import CACHE, FEE, PROXY, FREQ, load_prices, simulate

# 变体定义：name -> (params_override, use_risk_state: bool)
# use_risk_state=True 时用 confirmed_risk_off_axis 预计算的状态驱动；False(原版)直接 sr.decide 默认
VARIANTS = [
    ("V0 原版(立即全仓国债)",  {"RISK_CONFIRM_DAYS": 0, "DEFENSE_MODE": "defensive"}, False),
    ("C2 退抖+全仓国债",       {"RISK_CONFIRM_DAYS": 2, "RISK_EXIT_DAYS": 2, "DEFENSE_MODE": "defensive"}, True),
    ("C2+保留强势档",          {"RISK_CONFIRM_DAYS": 2, "RISK_EXIT_DAYS": 2, "DEFENSE_MODE": "keep_trend"}, True),
    ("C2+空仓等待",            {"RISK_CONFIRM_DAYS": 2, "RISK_EXIT_DAYS": 2, "DEFENSE_MODE": "cash"}, True),
    ("C3+空仓等待",            {"RISK_CONFIRM_DAYS": 3, "RISK_EXIT_DAYS": 3, "DEFENSE_MODE": "cash"}, True),
]


def decide_targets(price, dec_dates, params, risk_state=None):
    targets = {}
    for d in dec_dates:
        hist = price.loc[:d]
        if len(hist) < 200:
            targets[d] = "cash"
            continue
        prices = {c: hist[c].values for c in price.columns}
        rs = None
        if risk_state is not None and d in risk_state.index:
            rs = bool(risk_state.loc[d])
        res = sr.decide(prices, params, risk_state=rs)
        targets[d] = res["target"]
    return targets


def main():
    price = load_prices()
    print(f"数据区间 {price.index[0].date()} ~ {price.index[-1].date()}，共 {len(price)} 交易日")
    dec_dates = [d for d in price.resample(FREQ).last().index if len(price.loc[:d]) >= 200]

    base_params = {
        "GLOBAL_US": "513100", "GLOBAL_GOLD": "518880", "BROAD": "510300",
        "DEFENSIVE": sr.DEFENSIVE, "ETF_UNIVERSE": sr.ETF_UNIVERSE,
    }

    # 预计算各退抖档位的风险状态轴（按 params 去重缓存）
    state_cache = {}
    results = {}
    for name, ov, use_state in VARIANTS:
        params = dict(base_params)
        params.update(ov)
        conf = params.get("RISK_CONFIRM_DAYS", 0)
        if use_state:
            key = conf
            if key not in state_cache:
                state_cache[key] = sr.confirmed_risk_off_axis({c: price[c] for c in price.columns}, params)
            risk_state = state_cache[key]
        else:
            risk_state = None
        targets = decide_targets(price, dec_dates, params, risk_state)
        results[name] = simulate(price, dec_dates, targets, {})

    # 打印
    print(f"\n{'方案':20s} {'累计':>7s} {'年化':>7s} {'回撤':>8s} {'换仓':>5s} {'持仓占比':>8s}")
    for name in [v[0] for v in VARIANTS]:
        r = results[name]
        print(f"{name:20s} {r['cum']:7.3f}x {r['ann']*100:6.2f}% {r['maxdd']*100:7.2f}% "
              f"{r['switches']:5d} {r['time_in_mkt']*100:7.1f}%")

    # 风险-off 状态统计（用 C2 的状态轴）
    conf2_state = state_cache.get(2)
    if conf2_state is not None:
        days_off = int(conf2_state.sum())
        raw = sr.risk_off_raw_axis({c: price[c] for c in price.columns},
                                   dict(base_params, **{"RISK_CONFIRM_DAYS": 0}))
        days_raw = int(raw.sum())
        print(f"\n风险信号统计(全区间): 原始信号={days_raw}日  C2确认后防御={days_off}日(减少{raw.sum()-conf2_state.sum():.0f}日抖动)")

    # 债券防御期收益：国债指数年化供"空仓vs国债"参考已单列，此处不再重复
    
    # 保存
    out = {name: {k: v for k, v in results[name].items() if k not in ("equity", "port", "adj")}
           for name in [v[0] for v in VARIANTS]}
    with open(os.path.join(CACHE, "defensive_variants_result.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()