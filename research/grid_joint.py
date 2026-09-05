# -*- coding: utf-8 -*-
"""
grid_joint.py —— 全参数「联合寻优」（真金白银专项复核）
================================================================================
目的：回答"当前实盘参数是回测最优组合吗"，不再逐参数孤立扫描，而是把
      核心交互旋钮做**笛卡尔积联合网格**，在实盘口径下求全局 argmax。

实盘口径（与 grid_exec_close 完全一致）：
  - 当日收盘成交 + 双边滑点 slip(0.1%) + 佣金（引擎已计入）
  - T+1 标的(510300/510500/159915)止损次日离场
  - DEFENSIVE=518880 黄金防御、DEF_MOM_DAYS=10 + 迟滞带(0.005/-0.008，对齐实盘)
  - HOLD_N=1（backtest_freq_sweep 已单独验证，非本网格交互旋钮）

联合网格旋钮（3 维，核心交互项）：
  1) SLOPE_WINDOW (打分窗口)：20,24,26,28,30,32,34,36,40
  2) TRAIL          (跟踪止损%：普通+防御统一，对齐实盘"统一3%"设计)：0.02,0.03,0.04,0.06
  3) MIN_HOLD       (主动换仓最小间隔 自然日)：1,3,5

稳健判据（与往届一致）：
  滚动预热起点 2015..2020 共 6 个；固定评估窗 2021+；取 6 起点中位年化为主判据，
  同时输出 min/max/std(区间稳健) + 换仓 + 全期中位 + 回撤中位。

S/R 风控(STOP_CAP/ENTRY_BUF)与 HOLD_N 为独立验证的固定框架，不纳入本网格。

输出：
  ① 联合网格全表（按 2021+ 中位年化降序，打印前 TOP）
  ② 全局 argmax 及其 min/max/std/换仓/回撤
  ③ 生产配置(SW30/trail3%/moh1) vs 联合 argmax 对照
  ④ 防御层敏感性：对 argmax 配置再扫 def_trail_pct（仅防御资产跟踪止损）
     —— 确认"防御旋钮保持不变时是否会推翻主 winner"，闭合交互空档
"""
import os, sys, time, csv
import numpy as np
import pandas as pd
import multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)

RESULT_CSV = os.path.join(HERE, 'grid_joint_results.csv')   # 断点续跑持久化
CSV_FIELDS = ['SW', 'trail', 'moh', 'oos_med', 'oos_min', 'oos_max', 'oos_std',
              'full_med', 'sw_med', 'dd_med']

import backtest_universe_ext as bue
import backtest_minhold as mh

CODES = sorted({'510300', '510500', '159915', '518880', '513100', '511010', '513500'})
T1 = {'510300', '510500', '159915'}
EVAL = pd.Timestamp('2021-01-01')
WARMUPS = ['2015-01-01', '2016-01-01', '2017-01-01', '2018-01-01', '2019-01-01', '2020-01-01']
SLIP = 0.001                       # 双边滑点 0.1%（实盘口径）
DEF_MOM_DAYS = 10                  # 对齐实盘
DEF_ENTER, DEF_EXIT = 0.005, -0.008  # 对齐实盘迟滞带

SW_GRID = [20, 24, 26, 28, 30, 32, 34, 36, 40]
TRAIL_GRID = [0.02, 0.03, 0.04, 0.06]
MOH_GRID = [1, 3, 5]

CLOSE = None                       # 进程内缓存


def _init(_codes):
    global CLOSE
    c, _ = bue.load_prices_open(_codes)
    CLOSE = c[c.index >= '2014-01-01']


def _sim_one(wu, sw, trail, moh, def_trail):
    """单起点单配置模拟，返回 (full, oos, switches, maxdd)。"""
    G = dict(bue.GOLD)
    G['SLOPE_WINDOW'] = sw; G['MOM_WINDOW'] = 20
    G['DEF_MOM_ENTER'] = DEF_ENTER; G['DEF_MOM_EXIT'] = DEF_EXIT
    pr = CLOSE[CLOSE.index >= wu]
    r = mh.simulate_daily(pr, moh, trail_pct=trail, sr_params=G,
                          def_trail_pct=def_trail, def_mom_days=DEF_MOM_DAYS,
                          universe=CODES, t1_codes=T1, slip=SLIP)
    full = r['cum'] ** (252.0 / len(r['equity'])) - 1 if r['cum'] > 0 else -1.0
    so = mh.slice_metrics(r, EVAL)
    oos = so[0] * 100 if so else float('nan')
    return full * 100, oos, r['switches'], r['maxdd'] * 100


def _eval_combo(args):
    sw, trail, moh, def_trail = args
    fulls, ooss, swi, dd = [], [], [], []
    for wu in WARMUPS:
        f, o, s, d = _sim_one(wu, sw, trail, moh, def_trail)
        fulls.append(f); ooss.append(o); swi.append(s); dd.append(d)
    a = np.asarray(ooss)
    return {'SW': int(sw), 'trail': round(trail, 4), 'moh': int(moh),
            'oos_med': round(float(np.median(a)), 3), 'oos_min': round(float(a.min()), 3),
            'oos_max': round(float(a.max()), 3), 'oos_std': round(float(a.std()), 3),
            'full_med': round(float(np.median(fulls)), 3),
            'sw_med': round(float(np.median(swi)), 1), 'dd_med': round(float(np.median(dd)), 3)}


def load_done():
    """读取已算完的组合，返回 {(sw,trail,moh): row}，用于断点续跑跳过。"""
    done = {}
    if os.path.exists(RESULT_CSV):
        with open(RESULT_CSV, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                try:
                    key = (int(row['SW']), float(row['trail']), int(row['moh']))
                    done[key] = row
                except Exception:
                    continue
    return done


def persist_row(row):
    new = not os.path.exists(RESULT_CSV)
    with open(RESULT_CSV, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def load_all():
    return pd.read_csv(RESULT_CSV)


def main():
    t0 = time.time()
    n_jobs = mp.cpu_count()
    done = load_done()
    print(f"联合寻优(断点续跑) | CPU={n_jobs}并行 | 实盘口径(当日收盘+滑点{SLIP*100:.1f}%+佣金+T+1+黄金迟滞)"
          f" | 网格 {len(SW_GRID)}×{len(TRAIL_GRID)}×{len(MOH_GRID)}={len(SW_GRID)*len(TRAIL_GRID)*len(MOH_GRID)}组合"
          f"×6起点 | 已算完 {len(done)} | 待算 {len(SW_GRID)*len(TRAIL_GRID)*len(MOH_GRID)-len(done)}\n", flush=True)

    # ---------- 主网格：def_trail 与 trail 统一；跳过已完成 ----------
    pending = []
    for sw in SW_GRID:
        for tr in TRAIL_GRID:
            for moh in MOH_GRID:
                if (sw, tr, moh) not in done:
                    pending.append((sw, tr, moh, tr))
    if pending:
        with mp.Pool(n_jobs, initializer=_init, initargs=(CODES,)) as pool:
            for r in pool.imap_unordered(_eval_combo, pending, chunksize=1):
                persist_row(r)
                done[(r['SW'], r['trail'], r['moh'])] = r
                n = len(done)
                print(f"[{n}/{len(SW_GRID)*len(TRAIL_GRID)*len(MOH_GRID)}] "
                      f"SW{r['SW']} trail{r['trail']*100:.0f}% moh{r['moh']}d"
                      f" → oos中位 {r['oos_med']:6.1f}%", flush=True)
        print("网格计算完成\n", flush=True)
    else:
        print("网格已全部计算（从断点读取）\n", flush=True)

    df = load_all().sort_values('oos_med', ascending=False).reset_index(drop=True)
    hdr = (f"{'SW':>4}{'trail':>7}{'moh':>5}{'2021+中位':>10}{'min':>8}{'max':>8}"
           f"{'std':>7}{'全期中位':>9}{'换仓':>7}{'回撤':>8}")
    print("== ① 联合网格全表（top20，按2021+中位年化降序）==")
    print(hdr); print('-' * len(hdr))
    for _, r in df.head(20).iterrows():
        print(f"{r['SW']:>4.0f}{r['trail']*100:>6.0f}%{r['moh']:>5.0f}"
              f"{r['oos_med']:>10.1f}{r['oos_min']:>8.1f}{r['oos_max']:>8.1f}"
              f"{r['oos_std']:>7.1f}{r['full_med']:>9.1f}{r['sw_med']:>7.0f}{r['dd_med']:>8.1f}",
              flush=True)

    # ---------- ② 全局 argmax ----------
    best = df.iloc[0]
    print(f"\n== ② 全局联合最优 ==")
    print(f"SW={best['SW']:.0f} trail={best['trail']*100:.0f}% min_hold={best['moh']:.0f}d "
          f"| 2021+中位年化 {best['oos_med']:.1f}% (min{best['oos_min']:.1f}~max{best['oos_max']:.1f}, "
          f"std{best['oos_std']:.1f}) | 全期中位 {best['full_med']:.1f}% | 换仓{best['sw_med']:.0f} | "
          f"回撤中位{best['dd_med']:.1f}%", flush=True)

    # ---------- ③ 生产配置对照 ----------
    prod = df[(df.SW == 30) & (df.trail == 0.03) & (df.moh == 1)]
    if len(prod):
        p = prod.iloc[0]
        k = best['oos_med'] - p['oos_med']
        tag = '↑ 联合优于生产' if k > 0 else ('≡ 联合=生产' if abs(k) < 0.5 else '↓ 生产优于联合')
        print(f"\n== ③ 生产配置(SW30/trail3%/moh1) vs 联合argmax ==")
        print(f"   生产: oos中位 {p['oos_med']:.1f}%  min{p['oos_min']:.1f}~max{p['oos_max']:.1f} "
              f"换仓{p['sw_med']:.0f} 回撤{p['dd_med']:.1f}%")
        print(f"   联合: oos中位 {best['oos_med']:.1f}%  min{best['oos_min']:.1f}~max{best['oos_max']:.1f} "
              f"换仓{best['sw_med']:.0f} 回撤{best['dd_med']:.1f}%")
        print(f"   差距(联合-生产) = {k:+.1f}pp  → {tag}", flush=True)

    # ---------- ④ 防御层敏感性（argmax 主配置上扫 def_trail） ----------
    # 注：网格在 worker 中已初始化 CLOSE；此处在主进程直接跑，需先给主进程也加载
    if CLOSE is None:
        _init(CODES)
    print(f"\n== ④ 防御层敏感性：对 argmax(SW{best['SW']:.0f}/trail{best['trail']*100:.0f}%/moh{best['moh']:.0f}) "
          f"扫 def_trail_pct（仅防御资产跟踪止损，检验是否推翻主winner）==")
    print(f"{'def_trail':>10}{'2021+中位':>10}{'min':>9}{'max':>9}{'换仓':>8}{'回撤':>9}")
    base = best['oos_med']
    _sw, _trail, _moh = int(best['SW']), float(best['trail']), int(best['moh'])
    for dtr in [0.02, 0.03, 0.04, 0.06]:
        r = _eval_combo((_sw, _trail, _moh, dtr))
        shift = r['oos_med'] - base
        print(f"{dtr*100:>9.0f}%{r['oos_med']:>10.1f}{r['oos_min']:>9.1f}{r['oos_max']:>9.1f}"
              f"{r['sw_med']:>8.0f}{r['dd_med']:>9.1f}  ⊿{shift:+.1f}pp", flush=True)

    print(f"\n总耗时 {time.time()-t0:.0f}s", flush=True)


if __name__ == '__main__':
    main()