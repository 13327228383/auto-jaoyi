# -*- coding: utf-8 -*-
"""
monte_carlo_compare.py —— 旧(-8%止损) vs 新组合(2W-N1) 蒙特卡洛对照
========================================================================
新组合口径：从「2W-HOLD_N1-mom2」回测的真实月度收益提取年化几何增长/波动，
用同一套 lognormal 月度复利方法投影 5万，与旧口径(-8%止损: 几何7.44%/波动9.85%)并排对比。
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import backtest_alloc_v4 as alloc

START = 50000.0
N_PATHS = 20000
SEED = 42
HORIZONS = [1, 2, 5, 10, 20, 40]
THRESH_WAN = [0, 1, 2, 3, 4, 5]

OLD_DRIFT, OLD_VOL = 0.0744, 0.0985   # 旧(-8%止损)校准值


def new_params():
    """从 2W-N1 回测真实月度收益提年化几何增长/波动。"""
    price = alloc.bse.load_prices()
    dec = [d for d in price.resample("2W").last().index if len(price.loc[:d]) >= 200]
    targets = alloc.decide_targets(price, dec, hold_n=1)
    r = alloc.simulate(price, dec, targets, "mom2")
    port = r["port"]
    mon = port.resample("ME").apply(lambda x: (1 + x).prod() - 1).dropna()
    mean_m = float(mon.mean())
    std_m = float(mon.std())
    ann_vol = std_m * np.sqrt(12.0)
    geo_log_ann = float(np.log1p(mon).mean() * 12.0)   # 年化几何(对数)
    ann_arith = mean_m * 12.0
    n_mon = len(mon)
    years = n_mon / 12.0
    cagr = float(r["cum"] ** (1.0 / years) - 1.0)
    return dict(ann_arith=ann_arith, ann_vol=ann_vol, geo_log_ann=geo_log_ann,
                cagr=cagr, mon_n=n_mon, maxdd=r["maxdd"], switches=r["switches"],
                stop_exits=r["stop_exits"])


def simulate(drift, vol, N_years):
    rng = np.random.default_rng(SEED)
    m = 12 * N_years
    mu = drift / 12.0
    sig = vol / np.sqrt(12.0)
    log_ret = mu + sig * rng.standard_normal((N_PATHS, m))
    return START * np.exp(log_ret.sum(axis=1))


def report(tag, drift, vol, extra=""):
    print(f"\n========== {tag} ==========  {extra}")
    print(f"  年化几何(对数)增长={drift*100:.2f}% / 年化波动={vol*100:.2f}%")
    print(f"  中位值(万) → {'  '.join(f'{np.median(simulate(drift,vol,y))/1e4:5.1f}' for y in HORIZONS)}")
    print("  终值<=X万概率(%)")
    print(f"{'年数':>4}" + "".join(f"{x:>8}万" for x in THRESH_WAN))
    for y in HORIZONS:
        f = simulate(drift, vol, y) / 1e4
        row = f"{y:>4}" + "".join(f"{float((f <= x).mean())*100:>8.1f}" for x in THRESH_WAN)
        print(row)
    print("  P(≥105万):", "  ".join(f"{y}y={float((simulate(drift,vol,y)>=1050000).mean())*100:.1f}%" for y in HORIZONS))


def main():
    np.random.seed(SEED)
    p = new_params()
    print(f"新组合 2W-N1 回测参数提取：月度样本{n_pm if 0 else p['mon_n']}个月，"
          f"年化算术{p['ann_arith']*100:.2f}% / 几何(对数){p['geo_log_ann']*100:.2f}% / "
          f"波动{p['ann_vol']*100:.2f}% / 回测CAGR{p['cagr']*100:.2f}% / 回撤{p['maxdd']*100:.2f}%")

    report("旧口径 -8%止损", OLD_DRIFT, OLD_VOL, "几何7.44%/波动9.85%")
    report("新组合 2W-N1", p["geo_log_ann"], p["ann_vol"],
           f"年化CAGR≈{p['cagr']*100:.1f}%")


if __name__ == "__main__":
    main()