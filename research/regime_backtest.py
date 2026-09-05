# -*- coding: utf-8 -*-
"""
regime_backtest.py —— 行情状态(regime)回测研究：每个市场状态下该用哪组参数？
================================================================================
正确/无前视评估"按行情状态调参是否值得"。

先前版本犯的错误（已弃用）：把"事后按状态挑当天最优参数的日收益"跨不同权益基准复利，
得到 536773% 的假收益——(a) 前视选择，(b) 把不同起始权益的策略收益叠加，不成一条收益线。
真金白银绝不能信那个数字。

本版（正确) walk-forward：
  TRAIN = 2014-01-01 .. 2020-12-31：逐候选在全期模拟后，把其收益按当天 regime 分层，
        选"每个 regime 下样本内年化最高"的参数 → 得到 regime→params 调度。
        同时选"样本内全局最优单参数"(single_best)。
  TEST  = 2021-01-01 .. 末：用上面选定（无测试期信息）的调度，各跑【一条连贯】的
        逐日模拟（引擎支持按日参数调度 sched）：
          A) regime 调度：每天用当天 regime 对应的参数
          B) 单参数：每天用 single_best
        比较 A vs B 的 2021+ 样本外年化。只有这条连贯曲线才是可信的 regime 增益证据。

市场状态判据（510300 代理，绝对阈值、先验固定，非样本内拟合）：
  HIGH_VOL    : 20日年化波动率 > 0.22
  UPTREND     : close > MA120 且 MA30>=MA120（站上长期均线且短均线上行）
  其它        : RANGING_WEAK（下行/牛皮并入"弱震荡"）
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)

import backtest_universe_ext as bue
import backtest_minhold as mh

CODES = sorted({'510300', '510500', '159915', '518880', '513100', '511010', '513500'})
T1 = {'510300', '510500', '159915'}
TRAIN_END = pd.Timestamp('2021-01-01')
PROXY = '510300'
VOL_HI = 0.22
SLIP = 0.001
GRID = [(sw, tr, moh) for sw in [26, 28, 30, 32]
        for tr in [0.02, 0.025, 0.03]
        for moh in [1, 3, 5, 7]]
REGIMES = ["UPTREND", "RANGING_WEAK", "HIGH_VOL"]


def detect_regime(px, vol_hi=VOL_HI):
    ma120 = px.rolling(120).mean()
    ma30 = px.rolling(30).mean()
    vol20 = px.pct_change().rolling(20).std() * np.sqrt(252)
    res = pd.Series(pd.NA, index=px.index, dtype="object")
    up = px.gt(ma120) & ma30.ge(ma120)
    hiv = vol20.gt(vol_hi)
    res[hiv] = "HIGH_VOL"
    res[up & ~hiv] = "UPTREND"
    res[(~up) & (~hiv)] = "RANGING_WEAK"
    return res


def params_for(sw, tr):
    G = dict(bue.GOLD)
    G['SLOPE_WINDOW'] = sw; G['MOM_WINDOW'] = 20
    G['DEF_MOM_ENTER'] = 0.005; G['DEF_MOM_EXIT'] = -0.008
    return G


def simulate_fixed(close, sw, tr, moh):
    r = mh.simulate_daily(close, moh, trail_pct=tr, sr_params=params_for(sw, tr),
                          def_trail_pct=tr, def_mom_days=10, universe=CODES,
                          t1_codes=T1, slip=SLIP)
    return r['equity']


def simulate_coherent(close, sched, base_head):
    """一条连贯逐日模拟：每天参数取 sched[d]，缺失日用 base_head 兜底。"""
    head_sw, head_tr, head_moh = base_head
    return mh.simulate_daily(close, head_moh, trail_pct=head_tr,
                             sr_params=params_for(head_sw, head_tr),
                             def_trail_pct=head_tr, def_mom_days=10,
                             universe=CODES, t1_codes=T1, slip=SLIP, sched=sched)


def year_ann(eq, start, end):
    sub = eq[(eq.index >= start) & (eq.index <= end)]
    if len(sub) < 2 or sub.iloc[-1] <= 0:
        return float('nan')
    n = len(sub)
    g = float(sub.iloc[-1] / sub.iloc[0])
    return float(g ** (252.0 / n) - 1) * 100


def stratify_ann(eq, regime, start, end):
    """把一条 equity 的日收益按 regime 分层(在 [start,end] 内)，返回 {regime: 年化%}。"""
    sub = eq[(eq.index >= start) & (eq.index <= end)]
    ret = sub.pct_change().replace([np.inf, -np.inf], 0.0).dropna()
    rg = regime.reindex(ret.index)
    out = {}
    for s in REGIMES:
        m = (rg == s)
        n = int(m.sum())
        if n >= 40:
            out[s] = float(np.mean(ret[m])) * 252 * 100
    return out


def main():
    close, _ = bue.load_prices_open(CODES)
    close = close[close.index >= '2014-01-01']
    regime = detect_regime(close[PROXY])

    print(f"口径: 当日收盘+双边滑点{SLIP*100:.1f}%+佣金 | 代理=510300 | "
          f"高波动=vol20>{VOL_HI:.0%} | 趋势=close>MA120且MA30>=MA120 | 其余=弱/震荡", flush=True)

    # ---- 预跑全部候选各一次全期（缓存复用） ----
    eq_cache = {(sw, tr, moh): simulate_fixed(close, sw, tr, moh) for (sw, tr, moh) in GRID}

    # ---- TRAIN 分层选优（只允许用 < 2021-01 的数据） ----
    best_by_regime_train = {}
    for s in REGIMES:
        best_cfg = None; best_a = -1e18
        for (sw, tr, moh) in GRID:
            lay = stratify_ann(eq_cache[(sw, tr, moh)], regime, close.index[0], TRAIN_END)
            if s in lay and lay[s] > best_a:
                best_a = lay[s]; best_cfg = (sw, tr, moh)
        best_by_regime_train[s] = (best_cfg, best_a)
        print(f"[TRN] {s:>13}: 最优 SW{best_cfg[0]} trail{best_cfg[1]:.1%} moh{best_cfg[2]}  年化{best_a:.1f}%",
              flush=True)

    # 样本内全局最优单参数
    single_best = None; sb_a = -1e18
    for (sw, tr, moh) in GRID:
        a = year_ann(eq_cache[(sw, tr, moh)], close.index[0], TRAIN_END)
        if a > sb_a:
            sb_a = a; single_best = (sw, tr, moh)
    print(f"[TRN] 全局单参数最优: SW{single_best[0]} trail{single_best[1]:.1%} "
          f"moh{single_best[2]}  年化{sb_a:.1f}%", flush=True)

    # ---- TEST 相干对比 ----
    def build_sched(mode):
        sched = {}
        for d, s in regime.items():
            if s is None:
                continue
            if mode == "regime":
                cfg = best_by_regime_train[s][0]
            else:
                cfg = single_best
            sw, tr, moh = cfg
            sched[d] = {"sr_params": params_for(sw, tr), "trail_pct": tr,
                        "def_trail_pct": tr, "min_hold": moh}
        return sched

    eq_regime = simulate_coherent(close, build_sched("regime"), single_best)['equity']
    eq_single = simulate_coherent(close, build_sched("single"), single_best)['equity']

    end = close.index[-1]
    a_regime = year_ann(eq_regime, TRAIN_END, end)
    a_single = year_ann(eq_single, TRAIN_END, end)
    print("\n== TEST 2021+ 样本外（连贯单条收益线，无前视） ==")
    print(f"   单参数(全局最优)  : {a_single:6.2f}%")
    print(f"   regime 按态选参   : {a_regime:6.2f}%")
    print(f"   差值              : {a_regime - a_single:+6.2f}pp")

    # regime 组合收益来自哪些状态
    print("\n   regime-coherent 组合在 TEST 各状态的年化贡献:")
    for s, a in stratify_ann(eq_regime, regime, TRAIN_END, end).items():
        print(f"      {s:>13}: {a:6.2f}%")

    margin = a_regime - a_single
    verdict = ("regime 值得上线" if margin > 2.0
               else "regime 增益有限(≤2pp)，单参数已够稳" if margin >= -2.0
               else "regime 反而更差")
    print(f"   结论: {verdict}（{margin:+.2f}pp）")

    out = {"train_window": f"{close.index[0]}..{TRAIN_END}",
           "test_window": f"{TRAIN_END}..{end}",
           "regime_distribution_test": {s: int((regime[regime.index >= TRAIN_END] == s).sum()) for s in REGIMES},
           "single_best_train": {"SW": single_best[0], "trail": single_best[1], "moh": single_best[2], "ann": round(sb_a, 2)},
           "single_test_ann": round(a_single, 2),
           "regime_test_ann": round(a_regime, 2),
           "margin_pp": round(margin, 2), "verdict": verdict}
    try:
        import json
        with open(os.path.join(HERE, "regime_backtest_result.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2, default=lambda x: str(x))
    except Exception as e:
        print("写结果文件失败:", e)


if __name__ == "__main__":
    main()