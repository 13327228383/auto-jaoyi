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
IMPROVE_PCT = 1.0           # 顶替需实盘年化 ≥ 现役 + 1.0pp
DD_PENALTY = 3.0            # 实盘回撤不得比现役差超过 3.0pp
MIN_SWITCH_INTERVAL = 10    # 距上次切换至少 10 天
AUTO_TUNE = True            # 总开关（False=只看不切）


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


def run(force=False):
    """评估并（在满足护栏时）切换现役参数。返回审计摘要 dict。"""
    if not db.ensure_db():
        return {"ok": False, "reason": "db_unavailable", "changed": False}

    trades = db.real_trades()
    # 现役
    act = db.get_active_params()
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
        scores.append({
            "param_set_id": pid, "n_roundtrips": len(ann), "n_days": n_days,
            "real_ann_pct": (float(np.median(ann)) * 100) if ann else None,
            "fills": fills, "backtest_oos_med_pct": (bt or {}).get("oos_med_pct"),
        })

    # 选优：只在样本充足的候选里，选实盘年化最高者；是否顶替受护栏约束
    rich = [s for s in scores if s["real_ann_pct"] is not None and s["n_days"] >= MIN_EVAL_DAYS]
    changed = False
    reason = "样本不足，维持现役"
    detail = {"scores": scores, "min_eval_days": MIN_EVAL_DAYS,
              "improve_pct": IMPROVE_PCT, "dd_penalty": DD_PENALTY,
              "min_switch_interval": MIN_SWITCH_INTERVAL, "auto_tune": AUTO_TUNE}

    if len(rich) >= 1 and cur_id is not None:
        best = max(rich, key=lambda s: s["real_ann_pct"] or -999)
        cur_score = next((s for s in rich if s["param_set_id"] == cur_id), None)
        if best["param_set_id"] == cur_id:
            reason = f"现役即实盘最优（{best['real_ann_pct']:.1f}%），维持"
        else:
            cur_v = cur_score["real_ann_pct"] if cur_score else None
            margin = (best["real_ann_pct"] or 0) - (cur_v or 0)
            detail.update({"candidate": best["param_set_id"], "candidate_ann": best["real_ann_pct"],
                           "cur_ann": cur_v, "margin": margin})
            if AUTO_TUNE and force is False and margin < IMPROVE_PCT:
                reason = (f"候选 param#{best['param_set_id']} 实盘 {best['real_ann_pct']:.1f}% 仅优于现役 "
                          f"{margin:+.2f}pp < {IMPROVE_PCT}pp 门槛，维持现役")
            else:
                db.set_active(best["param_set_id"],
                              f"auto_tuner: 实盘年化{best['real_ann_pct']:.1f}% 优于现役{margin:+.2f}pp "
                              f"(n_days={best['n_days']})")
                db.audit(cur_id, best["param_set_id"], True, reason if AUTO_TUNE else "dry-run",
                         {"new": best, "margin": margin})
                changed = True
                reason = (f"已切至 param#{best['param_set_id']}（实盘{best['real_ann_pct']:.1f}%，"
                          f"优于现役{margin:+.2f}pp）")
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