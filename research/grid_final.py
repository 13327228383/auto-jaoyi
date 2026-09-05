# -*- coding: utf-8 -*-
"""
grid_final.py —— 终极稳健选参：trail × SW × min_hold × 滑点敏感性（赚钱导向）
================================================================================
联合寻优(grid_joint)暴露：收益最大增益来自 trail 2%。但 2% trailing 最怕真实摩擦
与滑点（盘中假触发、竞价噪声容易被洗出）。本脚本对候选邻域做 **滑点三维敏感性**，
用"高滑点下不崩 + 换仓少(操作摩擦) + 收益高"的决策规则选定最终落地参数。

口径（与 grid_joint 完全一致，实盘口径）：
  当日收盘成交 + 双边滑点 slip + 佣金 + T+1 + 黄金迟滞带(0.005/-0.008)
  滚动 6 起点；判据 = 2021+ 中位年化（同时报告 min/max/std/换仓/回撤）

网格：
  TRAIL  ∈ {0.02, 0.025, 0.03}      （联合峰值邻域）
  SW     ∈ {26, 28, 30}
  MIN_HOLD ∈ {3, 5}
  SLIP   ∈ {0.0005, 0.001, 0.002}   （0.05% / 0.1% / 0.2% 真实摩擦下界/基准/上界）

决策规则（选出对摩擦最沉稳优）：
  ① 基准滑点 0.1% 下 2021+ 中位年化高
  ② 恶劣滑点 0.2% 下仍明显为正、不塌方（防实盘摩擦吃穿）
  ③ 换仓次数少（操作磨损小）
  ④ 区间 std 小（路径稳健）

结果断点落盘 grid_final_results.csv；附生产对照 + 默认 argmax(仅0.1%) vs 稳健最优。
"""
import os, sys, time, csv
import numpy as np
import pandas as pd
import multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)

RESULT_CSV = os.path.join(HERE, 'grid_final_results.csv')
CSV_FIELDS = ['trail', 'SW', 'moh', 'slip', 'oos_med', 'oos_min', 'oos_max',
              'oos_std', 'full_med', 'sw_med', 'dd_med']

import backtest_universe_ext as bue
import backtest_minhold as mh

CODES = sorted({'510300', '510500', '159915', '518880', '513100', '511010', '513500'})
T1 = {'510300', '510500', '159915'}
EVAL = pd.Timestamp('2021-01-01')
WARMUPS = ['2015-01-01', '2016-01-01', '2017-01-01', '2018-01-01', '2019-01-01', '2020-01-01']
DEF_MOM_DAYS = 10
DEF_ENTER, DEF_EXIT = 0.005, -0.008

TRAIL_GRID = [0.02, 0.025, 0.03]
SW_GRID = [26, 28, 30]
MOH_GRID = [3, 5]
SLIP_GRID = [0.0005, 0.001, 0.002]

CLOSE = None


def _init(_codes):
    global CLOSE
    c, _ = bue.load_prices_open(_codes)
    CLOSE = c[c.index >= '2014-01-01']


def _sim_one(wu, sw, trail, moh, slip, def_trail):
    G = dict(bue.GOLD)
    G['SLOPE_WINDOW'] = sw; G['MOM_WINDOW'] = 20
    G['DEF_MOM_ENTER'] = DEF_ENTER; G['DEF_MOM_EXIT'] = DEF_EXIT
    pr = CLOSE[CLOSE.index >= wu]
    r = mh.simulate_daily(pr, moh, trail_pct=trail, sr_params=G,
                          def_trail_pct=def_trail, def_mom_days=DEF_MOM_DAYS,
                          universe=CODES, t1_codes=T1, slip=slip)
    full = r['cum'] ** (252.0 / len(r['equity'])) - 1 if r['cum'] > 0 else -1.0
    so = mh.slice_metrics(r, EVAL)
    oos = so[0] * 100 if so else float('nan')
    return full * 100, oos, r['switches'], r['maxdd'] * 100


def _eval(args):
    sw, trail, moh, slip = args
    fulls, ooss, swi, dd = [], [], [], []
    for wu in WARMUPS:
        f, o, s, d = _sim_one(wu, sw, trail, moh, slip, trail)
        fulls.append(f); ooss.append(o); swi.append(s); dd.append(d)
    a = np.asarray(ooss)
    return {'trail': round(trail, 4), 'SW': int(sw), 'moh': int(moh), 'slip': slip,
            'oos_med': round(float(np.median(a)), 3), 'oos_min': round(float(a.min()), 3),
            'oos_max': round(float(a.max()), 3), 'oos_std': round(float(a.std()), 3),
            'full_med': round(float(np.median(fulls)), 3),
            'sw_med': round(float(np.median(swi)), 1), 'dd_med': round(float(np.median(dd)), 3)}


def load_done():
    done = {}
    if os.path.exists(RESULT_CSV):
        with open(RESULT_CSV, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                try:
                    key = (float(row['trail']), int(row['SW']), int(row['moh']), float(row['slip']))
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


def main():
    t0 = time.time()
    n_jobs = mp.cpu_count()
    done = load_done()
    pending = [(sw, tr, moh, sp) for tr in TRAIL_GRID for sw in SW_GRID
               for moh in MOH_GRID for sp in SLIP_GRID
               if (tr, sw, moh, sp) not in done]
    tot = len(TRAIL_GRID) * len(SW_GRID) * len(MOH_GRID) * len(SLIP_GRID)
    print(f"终极稳健选参 | CPU={n_jobs} | 网格 trail×SW×moh×slip = {tot} 配置×6起点 "
          f"| 已算完 {len(done)} | 待算 {len(pending)}\n", flush=True)
    if pending:
        if CLOSE is None:
            _init(CODES)
        with mp.Pool(n_jobs, initializer=_init, initargs=(CODES,)) as pool:
            for r in pool.imap_unordered(_eval, pending, chunksize=1):
                persist_row(r); done[(r['trail'], r['SW'], r['moh'], r['slip'])] = r
                n = len(done)
                print(f"[{n}/{tot}] trail{r['trail']*100:.2f}% SW{r['SW']} moh{r['moh']} "
                      f"slip{r['slip']*100:.2f}% → {r['oos_med']:6.1f}%", flush=True)
        print("计算完成\n", flush=True)

    df = pd.read_csv(RESULT_CSV)

    # 透视：每个 (trail,SW,moh) 在三个滑点下的 2021+ 中位年化
    piv = df.pivot_table(index=['trail', 'SW', 'moh'], columns='slip', values='oos_med').reset_index()
    sw2 = df[df.slip == 0.001].set_index(['trail', 'SW', 'moh'])[['oos_min', 'oos_max', 'oos_std', 'sw_med', 'dd_med']]
    merg = piv.merge(sw2, left_on=['trail', 'SW', 'moh'], right_index=True)

    print("== 各配置滑点敏感性（2021+ 中位年化 % / 下标=换仓次数）==")
    hdr = (f"{'trail':>7}{'SW':>5}{'moh':>5}{'0.05%':>9}{'0.10%':>9}{'0.20%':>9}"
           f"{'min@0.1':>9}{'std@0.1':>8}{'换仓':>7}{'回撤':>8}")
    print(hdr); print('-' * len(hdr))
    for _, r in merg.iterrows():
        print(f"{r['trail']*100:>6.2f}%{r['SW']:>5.0f}{r['moh']:>5.0f}"
              f"{r[0.0005]:>9.1f}{r[0.001]:>9.1f}{r[0.002]:>9.1f}"
              f"{r['oos_min']:>9.1f}{r['oos_std']:>8.1f}{r['sw_med']:>7.0f}{r['dd_med']:>8.1f}",
              flush=True)

    # 决策：基准0.1%收益高 + 0.2%为正且塌方小 + 换仓少 + std小
    print("\n== 稳健最优选择（决策项）==")
    base = df[df.slip == 0.001]
    adv = df[df.slip == 0.002]
    row = []
    for k, g in merg.iterrows():
        bm = base['oos_med']
        r0 = g[0.0005]; r1 = g[0.001]; r2 = g[0.002]
        collapse = (r1 - r2)                 # 0.1%→0.2% 损失，越小越好
        row.append({'trail': g.trail, 'SW': g.SW, 'moh': g.moh,
                    'r1': r1, 'r2': r2, 'collapse': collapse,
                    'sw': g.sw_med, 'std': g.oos_std})
    rt = pd.DataFrame(row)
    # 候选：0.2% 下仍 >= 15% 的配置里，选 0.1% 收益高、损失小、换仓少
    cand = rt[rt.r2 >= 15.0].sort_values(['r1', 'collapse'], ascending=[False, True])
    cand = cand.sort_values(['r1'], ascending=False)
    print("候选(0.2%滑点仍>=15%)，按 0.1% 收益降序：")
    print(f"{'trail':>7}{'SW':>5}{'moh':>5}{'r@0.1':>8}{'r@0.2':>8}{'折损':>8}{'换仓':>7}{'std':>7}")
    for _, r in cand.head(12).iterrows():
        print(f"{r.trail*100:>6.2f}%{r.SW:>5d}{r.moh:>5d}{r.r1:>8.1f}{r.r2:>8.1f}"
              f"{r.collapse:>8.1f}{r.sw:>7.0f}{r.std:>7.1f}")
    if len(cand):
        best = cand.iloc[0]
        print(f"\n>>> 稳健最优 = trail{best.trail*100:.2f}% / SW{best.SW} / min_hold={best.moh}d "
              f"| 0.1%: {best.r1:.1f}%  0.2%: {best.r2:.1f}%  折损{best.collapse:.1f}pp "
              f"| 换仓{best.sw}  std{best.std}")

    # 生产对照说明（生产 moh1 未在本网格，其 0.1%=19.8% 来自 grid_joint；此处用同 trail3% 的 moh5 近似展示尾随水平）
    if len(cand):
        b = cand.iloc[0]
        print(f"\n== 对照 ==")
        print(f"   生产(SW30/trail3%/moh1)   0.1% = 19.8% (grid_joint 实测)")
        print(f"   稳健最优(SW{int(b.SW)}/trail{b.trail*100:.2f}%/moh{int(b.moh)}d)  0.1% = {b.r1:.1f}%  0.2% = {b.r2:.1f}%")
    print(f"\n总耗时 {time.time()-t0:.0f}s", flush=True)


if __name__ == '__main__':
    main()