# -*- coding: utf-8 -*-
"""
monte_carlo_extension.py —— 蒙特卡洛补充：多时间尺度 5万 投影 + 归零/亏损阈值概率
======================================================================================
复现并扩展既有口径（见 .workbuddy/memory/2026-08-27.md 续12）：
  策略月度收益 → 年化几何增长 7.44%、年化波动 9.85%（140个月样本 2016-12~2026-08）；
  5万起步、2万条路径、月度 lognormal 复利。

输出：
  1) 1/2/5/10/20/40 年：中位终值 + 5%/95% 分位；
  2) 校验锚点：40年 达到105万(21x)≈45%、达到150万(30x)≈25% 是否复现；
  3) 每时间尺度：终值≤0,1,2,3,4,5 万 的概率（亏损/归零尾部）。
"""
import numpy as np

START = 50000.0          # 5万
N_PATHS = 20000
SEED = 42
DRIFT_ANN = 0.0744       # 年化几何(对数)增长，复现中位 98.1万
VOL_ANN = 0.0985         # 年化波动
HORIZONS = [1, 2, 5, 10, 20, 40]
THRESH_WAN = [0, 1, 2, 3, 4, 5]   # 单位：万


def simulate(N_years):
    rng = np.random.default_rng(SEED)
    m = 12 * N_years
    mu_m = DRIFT_ANN / 12.0
    sig_m = VOL_ANN / np.sqrt(12.0)
    # 月度对数收益累乘
    log_ret = mu_m + sig_m * rng.standard_normal((N_PATHS, m))
    final = START * np.exp(log_ret.sum(axis=1))
    return final


def main():
    np.random.seed(SEED)
    print(f"蒙特卡洛：本{0}/{0}？ no —— 起点 5万，{N_PATHS} 条路径，年化几何{DRIFT_ANN*100:.2f}% / 波动{VOL_ANN*100:.2f}%\n")

    # 校验锚点
    f40 = simulate(40)
    p105 = float((f40 >= 1050000).mean())   # 105万
    p150 = float((f40 >= 1500000).mean())   # 150万
    print(f"[锚点校验] 40年：达到105万概率={p105*100:.1f}% (记45.4%)  达到150万概率={p150*100:.1f}% (记25%)"
          f"\n  中位={np.median(f40)/1e4:.1f}万 (记98.1万)\n")

    print(f"{'年数':>4}{'中位终值':>10}{'5%分位':>9}{'95%分位':>9}{'≥105万':>9}{'≥150万':>9}")
    for y in HORIZONS:
        f = simulate(y)
        med = np.median(f) / 1e4
        q5 = np.percentile(f, 5) / 1e4
        q95 = np.percentile(f, 95) / 1e4
        p105 = float((f >= 1050000).mean()) * 100
        p150 = float((f >= 1500000).mean()) * 100
        print(f"{y:>4}{med:>9.1f}万{q5:>8.1f}万{q95:>8.1f}万{p105:>8.1f}%{p150:>8.1f}%")

    print("\n每时间尺度：终值 ≤ X 万 的概率（5万起步，单位：万）")
    hdr = f"{'年数':>4}" + "".join(f"{x:>9}万" for x in THRESH_WAN)
    print(hdr)
    for y in HORIZONS:
        f = simulate(y) / 1e4
        row = f"{y:>4}" + "".join(f"{float((f <= x).mean())*100:>8.1f}%" for x in THRESH_WAN)
        print(row)

    print("\n注：对数正态模型下终值恒>0，'到0万'理论概率=0（真破产需极端连续血本无归，模型不覆盖）；")
    print("≤5万=不但没赚还亏钱；≤2万≈亏超40%。")


if __name__ == "__main__":
    main()