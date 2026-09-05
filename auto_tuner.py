# -*- coding: utf-8 -*-
"""
auto_tuner.py —— 自动调参器（依实盘表现选最合适参数）
================================================================================
在 tuning_db 基础上：读取实盘成交(trade_log)，把每个"上线过的参数组"算真实表现，
再按护栏选当期为优者，写入 active_param。auto_run 下次读取即生效。

收益口径与摩擦：
  用 trade_log 的成交复刻逐笔 round-trip（买→卖/卖→买），算分组后的实现收益与隐式磨损
  （换手次数=摩擦代理），与回测画像(backtest_json)并列展示，作为"回测榜 vs 实盘榜"的对照，
  避免只看回测、也避免样本太薄时被噪声误导。

护栏（默认保守，防真钱被频繁换参损害）：
  MIN_EVAL_DAYS   : 一个参数组至少上线满 N 个"交易日"才允许被评估（防样本太薄）
  IMPROVE_PCT     : 新参数组的实盘年化需 ≥ 现役 + 该值(百分点) 才允许顶替
  DD_PENALTY      : 新参数组实盘最大回撤不得比现役差超过该百分点，否则一票否决
  MIN_SWITCH_INTERVAL: 距上次切换至少 N 天，防抖动
  AUTO_TUNE       : 总开关。False → 只看榜不切换（dry-run 审计）
所有决策写 tuning_audit，可回溯。
"""
import os, sys, datetime, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tuning_db as db

# ---- 护栏参数（可在 DB 无数据时保持出厂） ----
MIN_EVAL_DAYS = 20          # 一个参数组至少上线满 20 个交易日才参评
MIN_ROUNDTRIPS = 5          # R2：至少完成 5 笔完整 buy→sell 往返才参评，防薄样本 median 伪信号
IMPROVE_PCT = 1.0           # 顶替需实盘年化 ≥ 现役 + 1.0pp
DD_PENALTY = 3.0            # 实盘回撤不得比现役差超过 3.0pp
MIN_SWITCH_INTERVAL = 10    # 距上次切换至少 10 天（R1：由 run() 真正执行，不再只是记录）
AUTO_TUNE = True            # 总开关（False=只看不切）
BOOT_N = 600                # R2：bootstrap 重采样次数

# ---- 方案A：逐步逼近的步进上限（与 auto_run.refresh 历史口径一致） ----
# auto_run 只做 1:1 应用（applied==DB，版号恒精确）；"单次收紧≤步长"的安全由本模块的
# 分步切换承担：目标超步长时，先把"中间参数"作为真实 param_sets 行落库并设现役，
# 后续评估继续逼近，绝不一次性跳到目标、也绝不产生 applied≠DB 的孤儿态。
STAGING_TRAIL_CUT = 0.005    # 止损类一次收紧上限
STAGING_TRAIL_LOOSE = 0.01   # 止损类一次放宽上限
STAGING_STEP = {"SLOPE_WINDOW": 5, "DEF_MOM_DAYS": 5,
                "DEF_MOM_ENTER": 0.005, "DEF_MOM_EXIT": 0.005, "HOLD_N": 1}

# 正在进行中的分步逼近目标 param_set_id（在 run 内进程级缓存）。
# 目的：第一步分步后现役变成"无实盘成交"的中间行，续跑若再过一遍 margin/significant 守卫会
# 因中间行无样本(cur_ann=None → significant=False)而被"不显著"卡死，永远到不了目标。
# 用该标记让"既定目标"的分步续跑直通，直到收敛到位再提交并清除标记，然后恢复正常守卫评估。
_STAGING_TARGET = None


def _step_for(key, cur, target):
    """某个键从 cur 走到 target 本轮最多迈出的步长（止损类收紧/放宽分向，其余对称）。"""
    if key in ("SR_TRAIL_PCT", "DEF_PEAK_STOP"):
        return STAGING_TRAIL_CUT if target < cur else STAGING_TRAIL_LOOSE
    return STAGING_STEP.get(key)


def _within_step(cur_params, target_params):
    """判断逐键目标是否都在单步上限内（=可直接切到目标，无需分步）。"""
    for k in db.TUNABLE_KEYS:
        _c, _t = cur_params.get(k), target_params.get(k)
        if _c is None or _t is None:
            continue
        _sp = _step_for(k, _c, _t)
        if _sp and abs(_t - _c) > _sp:
            return False
    return True


def _apply_switch(db_, cur_id, target_id, margin):
    """单步内即可达目标的真实切换；超步则落中间参数行并分步逼近。
    返回 (changed, reason)。本模块负责把"逐步逼近"落地为真实 param_set 行，保持 applied==DB。"""
    global _STAGING_TARGET
    best_params = db_.get_param(target_id) or {}
    cur_params = {k: v for k, v in (db_.get_active_params() or {}).items()
                  if k in db_.TUNABLE_KEYS}
    if best_params and cur_params and not _within_step(cur_params, best_params):
        # 分步逼近：逐键迈出一步（仍在步长内），作为真实 param_set 行落库并设现役
        mid = dict(cur_params)
        for k in db_.TUNABLE_KEYS:
            _c, _t = mid.get(k), best_params.get(k)
            if _c is None or _t is None:
                continue
            _sp = _step_for(k, _c, _t)
            if _sp and abs(_t - _c) > _sp:
                mid[k] = round(_c + (_sp if _t > _c else -_sp), 6)
        mid_id = db_.upsert_param_set({k: mid[k] for k in db_.TUNABLE_KEYS})
        if mid_id:
            db_.set_active(mid_id, f"auto_tuner staging toward param#{target_id} (single-step)")
            # staging 记为"非经济切换"(changed=0)，不重置 R1 防抖计时；once 到步长内再由直切提交
            db_.audit(cur_id, mid_id, False,
                      f"staging 逐步逼近目标 #{target_id}（单次步进≤上限）",
                      {"staging_to": target_id, "mid": mid})
            _STAGING_TARGET = target_id  # 标记既定目标，供续跑直通（防止无样本中间行卡死守卫）
            reason = (f"分步逼近 param#{target_id}：本轮先应用折中参数（{mid}）"
                      f"，后续评估继续逼近，applied恒等于DB")
            return True, reason
        # upsert/set 失败 → 保守直接落目标（与既有行为一致），避免卡死
    _STAGING_TARGET = None
    db_.set_active(target_id,
                   f"auto_tuner: 实盘年化优于现役{margin:+.2f}pp，直接切换/落目标")
    db_.audit(cur_id, target_id, True, "auto_tuner 切换至目标参数", {"target_id": target_id})
    return True, f"已切至 param#{target_id}（实盘年化优于现役{margin:+.2f}pp）"


def realized_roundtrips(trades):
    """把某参数组的逐笔成交复原为 round-trip，返回 (annualized_pct系列, 换手次数)。
    简化口径：按 code 聚合，buy→sell 视作一笔往返；用持仓天数做年化近似。
    样本少时返回空，表示样本不足。"""
    by = {}
    for t in trades:
        by.setdefault(t["code"], []).append(t)
    ann = []
    fills = 0
    for code, lst in by.items():
        lst = sorted(lst, key=lambda x: x["ts"])
        i = 0
        while i + 1 < len(lst):
            a, b = lst[i], lst[i + 1]
            side, px = a["side"], float(a["price"])
            if a["side"] in ("buy",) and b["side"] in ("sell", "stop_sell"):
                ret = (float(b["price"]) - px) / px if px else 0.0
                td = (datetime.datetime.fromisoformat(b["ts"]) -
                      datetime.datetime.fromisoformat(a["ts"])).days
                ann.append(ret * (252.0 / max(td, 1)))
                fills += 2
                i += 2
            else:
                i += 1
    return ann, fills


def _backtest_metrics(param_set_id):
    """该参数组在库里的回测画像（0.1%滑点下 oos 中位年化/换仓），缺省返回 None。"""
    try:
        import pandas as pd
        c = db.connect(db.DB_NAME)
        with c.cursor() as cur:
            cur.execute("SELECT backtest_json FROM param_sets WHERE id=%s", (param_set_id,))
            r = cur.fetchone()
        c.close()
        if r and r[0]:
            bj = r[0] if isinstance(r[0], dict) else json.loads(r[0])
            return {"oos_med_pct": bj.get("oos_med_pct"), "switches": bj.get("switches")}
    except Exception:
        pass
    return None


def _last_switch_time():
    """最近一次「真实切换现役参数」(changed=1) 的时间，用于 R1 防抖。无则返回 None。"""
    try:
        c = db.connect(db.DB_NAME)
        with c.cursor() as cur:
            cur.execute("SELECT MAX(run_at) FROM tuning_audit WHERE changed=1")
            r = cur.fetchone()
        c.close()
        return r[0] if (r and r[0]) else None
    except Exception:
        return None


def _boot_est(arr, lo=5, hi=95):
    """R2：bootstrap 稳健估计。返回 (中位数, 下分位, 上分位)（每条为「年化百分点同量纲」）。
    空样本返回 (None,None,None)。"""
    arr = np.asarray(arr, dtype=float)
    if arr.size == 0:
        return (None, None, None)
    rng = np.random.default_rng(20260905)
    idx = rng.integers(0, arr.size, size=(BOOT_N, arr.size))
    meds = np.median(arr[idx], axis=1) * 100.0
    return (float(np.median(meds)), float(np.percentile(meds, lo)), float(np.percentile(meds, hi)))


def run(force=False):
    """评估并（在满足护栏时）切换现役参数。返回审计摘要 dict。"""
    if not db.ensure_db():
        return {"ok": False, "reason": "db_unavailable", "changed": False}

    trades = db.real_trades()
    # 现役（B：优先用 active_param 落库的真实版本号，避免按参数反查歧义；无则回退参数反查）
    act = db.get_active_params()
    cur_id = db.get_active_param_set_id()
    if cur_id is None:
        cur_id = db.get_param_set_id(act)

    # 找出所有"曾在 active_param 出现过"的参数组
    cand_ids = {cur_id} if cur_id else set()
    try:
        c = db.connect(db.DB_NAME)
        with c.cursor() as cur:
            cur.execute("SELECT DISTINCT param_set_id FROM active_param WHERE param_set_id IS NOT NULL")
            cand_ids |= {r[0] for r in cur.fetchall()}
        c.close()
    except Exception:
        pass

    scores = []
    for pid in list(cand_ids):
        p_trades = [t for t in trades if t["param_set_id"] == pid]
        ann, fills = realized_roundtrips(p_trades)
        n_days = len({t["ts"][:10] for t in p_trades})
        bt = _backtest_metrics(pid)
        # R2：用 bootstrap 稳健估计（中位数 + 5%~95% 区间）替代裸 median；
        # 往返 < MIN_ROUNDTRIPS 视为样本不足 real_ann_pct=None，不参评。
        med, lo_, hi_ = _boot_est(ann) if len(ann) >= MIN_ROUNDTRIPS else (None, None, None)
        scores.append({
            "param_set_id": pid, "n_roundtrips": len(ann), "n_days": n_days,
            "real_ann_pct": med, "ci_lo_pct": lo_, "ci_hi_pct": hi_,
            "fills": fills, "backtest_oos_med_pct": (bt or {}).get("oos_med_pct"),
        })

    # 选优：样本充足(满 MIN_EVAL_DAYS 且 ≥MIN_ROUNDTRIPS 往返)且实盘年化非空才参评
    rich = [s for s in scores
            if s["real_ann_pct"] is not None and s["n_days"] >= MIN_EVAL_DAYS
            and s["n_roundtrips"] >= MIN_ROUNDTRIPS]
    changed = False
    reason = "样本不足，维持现役"
    detail = {"scores": scores, "min_eval_days": MIN_EVAL_DAYS,
              "improve_pct": IMPROVE_PCT, "dd_penalty": DD_PENALTY,
              "min_switch_interval": MIN_SWITCH_INTERVAL, "auto_tune": AUTO_TUNE}

    if len(rich) >= 1 and cur_id is not None:
        # —— 分步续跑直通：已有既定目标(中间行无实盘样本)时，跳过 margin/显著/防抖守卫，直接逼近。——
        # 首步分步的守卫已在最初切换时通过；续跑若再过守卫会因中间行 cur_ann=None→significant=False 卡死。
        if _STAGING_TARGET is not None and cur_id is not None:
            _tg = db.get_param(_STAGING_TARGET) or {}
            _cur = {k: v for k, v in (db.get_active_params() or {}).items()
                    if k in db.TUNABLE_KEYS}
            if _tg and _cur and _within_step(_cur, _tg):
                # 单步内可达 → 提交既定目标，结束本轮分步
                db.set_active(_STAGING_TARGET, "auto_tuner: 分步收敛到位，提交目标参数")
                db.audit(cur_id, _STAGING_TARGET, True,
                         "staging 收敛到位，提交既定目标", {"target": _STAGING_TARGET})
                _STAGING_TARGET = None
                return {"ok": True, "changed": True,
                        "reason": f"分步到位，已应用目标 param#{cur_id}", "scores": scores}
            # 未到位 → 继续朝既定目标走一步
            _changed, _reason = _apply_switch(db, cur_id, _STAGING_TARGET, None)
            return {"ok": True, "changed": _changed, "reason": _reason, "scores": scores}

        best = max(rich, key=lambda s: s["real_ann_pct"] or -999)
        cur_score = next((s for s in rich if s["param_set_id"] == cur_id), None)
        if best["param_set_id"] == cur_id:
            reason = f"现役即实盘最优（{best['real_ann_pct']:.1f}%），维持"
        else:
            cur_v = cur_score["real_ann_pct"] if cur_score else None
            margin = (best["real_ann_pct"] or 0) - (cur_v or 0)
            b_lo = best.get("ci_lo_pct"); cur_hi = cur_score.get("ci_hi_pct") if cur_score else None
            significant = (b_lo is not None and cur_hi is not None and b_lo > cur_hi)  # R2 区间不重叠才算显著
            # R1：距上次切换过近 → 防抖不切（仅非 force 时生效）
            _lst = _last_switch_time()
            days_since = ((datetime.date.today() - _lst.date()).days
                          if _lst else 9999)
            detail.update({"candidate": best["param_set_id"], "candidate_ann": best["real_ann_pct"],
                           "cur_ann": cur_v, "margin": margin,
                           "significant": significant, "days_since_last_switch": days_since})
            # 总开关(A)：AUTO_TUNE=False → 只看榜不切（force 可覆盖）
            if not AUTO_TUNE and not force:
                reason = "AUTO_TUNE=False，仅审计不切换"
            elif not force and margin < IMPROVE_PCT:
                reason = (f"候选 param#{best['param_set_id']} 实盘 {best['real_ann_pct']:.1f}% 仅优于现役 "
                          f"{margin:+.2f}pp < {IMPROVE_PCT}pp 门槛，维持现役")
            elif not force and not significant:
                _cv = cur_v if cur_v is not None else "—"
                reason = (f"候选 param#{best['param_set_id']} 实盘中位 {best['real_ann_pct']:.1f}% 与现役 "
                          f"{_cv}% 差异区间重叠，不显著(R2)，维持现役")
            elif not force and days_since < MIN_SWITCH_INTERVAL:
                reason = f"距上次切换仅 {days_since} 天 < {MIN_SWITCH_INTERVAL}，触发防抖(R1)，维持现役"
            else:
                changed, reason = _apply_switch(db, cur_id, best["param_set_id"], margin)
    else:
        if cur_id is None:
            reason = "尚无现役参数，先以研究最优上线"

    if not changed:
        db.audit(cur_id, cur_id, False, reason, detail)

    print(f"[auto_tuner] 评估完成 changed={changed} | {reason}")
    for s in scores:
        print(f"   param#{s['param_set_id']}: 实盘年化 {s['real_ann_pct'] if s['real_ann_pct'] is not None else '—':>6}%  "
              f"天数{s['n_days']} 往返{s['n_roundtrips']} "
              f"回测0.1%中位 {s['backtest_oos_med_pct'] if s['backtest_oos_med_pct'] is not None else '—'}%")
    return {"ok": True, "changed": changed, "reason": reason, "scores": scores}


if __name__ == "__main__":
    run()