# -*- coding: utf-8 -*-
"""
backtest_gold_tradeoff.py —— 黄金(518880)动量窗口：多维成本综合评估
================================================================================
目标：回答"哪个窗口最赚钱"，但把"赚钱"定义成【净可实现收益】，而非名义回测年化。
频率只是手段，收益(减掉磨损后)才是结果。

四个维度折算成单一目标(净可实现年化)：

  1) 名义年化    ：纯回测年化(扣手续费的毛利)
  2) 响应速度    ：动量窗口越短→对信号响应越快，但换手越高
  3) 磨损(摩擦成本)：
        a) 手续费：真实 A股ETF 双边费率 ≈ 万2.38 = 0.0238%/边，0.0476%/切换(memory口径)
        b) 滑点   ：GUI自动下单每笔难以精确吃理想价，估计双边 0.05~0.10% 损耗
        c) 执行损耗：每次下单触发同花顺验证码/确认弹窗，有失败重试、进入操作窗口的
                    时间与UI摩擦。用"单次额外的执行成本 e"建模(交易频率越高越被放大)。
  4) 限流        ：行情请求频率。窗口短→巡检/换手多→请求指数级上升→sina 限流风险↑。
                   以"每分钟换手期望"为代理(真限流难数值化，用风险档位标注)。

输出：对每个窗口给【名义年化 → 净可实现年化】的猎辑，让"1日动量高收益是纸面收益"显性化。
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from data_center import get_etf_k_history

# —— 成本参数（保守压力口径：恶劣情况下的可实现下限）——
FEE_SIDE = 0.000238     # 万2.38 单边（memory: 0.0476%/切换 双边）
SLIP_SIDE = 0.0003      # 单边滑点 0.03%（保守：GUI自动下单吃到更差价）
EXEC_FIX = 0.0010       # 单笔固定执行损耗 0.10%（验证码/确认弹窗/重试摩擦）
GOLD = "518880"

WINDOWS = [1, 3, 5, 10, 20]


def load_gold():
    df = get_etf_k_history(GOLD, days=3800)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").set_index("date")["close"].astype(float)


def simulate(close, window):
    """返回窗口的名义毛利 + 换手点。摩擦费用在【净路径】逐次扣除。"""
    hist = close.values
    n = len(hist)
    held = 0
    nom_equity = 1.0   # 名义(已含手续费,未含滑点/执行损耗)
    net_equity = 1.0   # 净(每次换手再扣滑点+执行损耗)
    peak_n = peak_net = 1.0
    maxdd_n = maxdd_net = 0.0
    switches = 0
    held_days = 0
    ret = close.pct_change().fillna(0.0).values
    for i in range(1, n):
        W = window
        if i >= W:
            m = hist[i] / hist[i - W] - 1.0
            if held == 0 and m > 0.005:
                held = 1
                switches += 1
                # 买入：扣一次单边滑点+一次执行损耗
                cost = SLIP_SIDE + EXEC_FIX
                net_equity *= (1 - cost)
            elif held == 1 and m < -0.008:
                held = 0
                switches += 1
                # 卖出：扣一次单边滑点+一次执行损耗
                cost = SLIP_SIDE + EXEC_FIX
                net_equity *= (1 - cost)
        r = (ret[i] - FEE_SIDE) if held else 0.0
        nom_equity *= (1 + r)
        net_equity *= (1 + r)
        if held:
            held_days += 1
        # 回撤维持(名义用净值走势近似的峰值)
        peak_n = max(peak_n, net_equity)
        peak_net = max(peak_net, net_equity)
        maxdd_n = min(maxdd_n, (net_equity - peak_n) / peak_n)
        maxdd_net = min(maxdd_net, (net_equity - peak_net) / peak_net)
    ann_nom = nom_equity ** (252.0 / n) - 1 if nom_equity > 0 else -1.0
    ann_net = net_equity ** (252.0 / n) - 1 if net_equity > 0 else -1.0
    return {"window": window, "nom_ann": ann_nom, "net_ann": ann_net,
            "nom_calmar": ann_net / abs(maxdd_n) if maxdd_n < 0 else float("nan"),
            "net_calmar": ann_net / abs(maxdd_net) if maxdd_net < 0 else float("nan"),
            "switches": switches, "hold%": held_days / n,
            "maxdd_net": maxdd_net, "ann_turn": switches / (n / 252.0),
            "friction%": 0.0}


def main():
    close = load_gold()
    n = len(close)
    print(f"黄金{GOLD} {close.index[0].date()}~{close.index[-1].date()} {n}交易日(T+0)\n")
    print("成本口径：费 万2.38/边 + 滑点 0.03%/边 + 每笔执行损耗 0.10%（保守压测）\n")
    hdr = (f"{'窗口':>6s}{'名义年化':>9s}{'净年化':>9s}{'名义Calmar':>11s}{'净Calmar':>10s}"
           f"{'换手/年':>8s}{'限流':>5s}")
    print(hdr); print("-" * len(hdr))
    for w in WINDOWS:
        r = simulate(close, w)
        print(f"{r['window']:>3d}日  {r['nom_ann']*100:>8.2f}  {r['net_ann']*100:>8.2f}"
              f"{r['nom_calmar']:>10.2f}  {r['net_calmar']:>9.2f}"
              f"{r['ann_turn']:>8.0f}  {'低' if r['ann_turn']<=150 else ('中' if r['ann_turn']<=300 else '高'):>4s}")
    print("\n净年化 = 名义回测扣除每次换手的滑点+执行损耗；换手/年>150 为中等以上限流风险档。")


if __name__ == "__main__":
    main()