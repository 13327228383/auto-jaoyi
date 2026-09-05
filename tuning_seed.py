# -*- coding: utf-8 -*-
"""
tuning_seed.py —— 播种：研究候选入库 + 设定初始现役参数 + 回填历史成交
================================================================================
1) 把 grid_final_results.csv / grid_joint_results.csv 的候选参数组合及回测画像写入 param_sets。
2) 把抗摩擦稳健最优(SW30/trail2%/moh5, 0.2%滑点抗性最高)设为初始现役 active_param。
3) 把 cache/trade_journal.jsonl 的历史成交回填 trade_log（历史成交无参数版本号，
   param_set_id 记 NULL 表示"演进期遗留"，后续成交由 auto_run 打现役版本）。
"""
import os, sys, json
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tuning_db as db

RES_FINAL = os.path.join(HERE, "research", "grid_final_results.csv")
RES_JOINT = os.path.join(HERE, "research", "grid_joint_results.csv")
JOURNAL = os.path.join(HERE, "cache", "trade_journal.jsonl")


def _mk_row(sw, trail, moh, metrics, source):
    return {
        "name": f"SR{int(trail*100)} SW{sw} MOH{moh}",
        "SLOPE_WINDOW": int(sw), "SR_TRAIL_PCT": round(trail, 4),
        "MIN_HOLD_DAYS": int(moh), "DEF_MOM_DAYS": 10,
        "DEF_MOM_ENTER": 0.005, "DEF_MOM_EXIT": -0.008, "HOLD_N": 1,
        "backtest": {"oos_med_pct": metrics.get("oos_med_pct"),
                     "oos_min_pct": metrics.get("oos_min_pct"),
                     "oos_max_pct": metrics.get("oos_max_pct"),
                     "oos_std": metrics.get("oos_std"),
                     "full_med_pct": metrics.get("full_med_pct"),
                     "switches": metrics.get("switches"),
                     "maxdd_pct": metrics.get("maxdd_pct"),
                     "slip_basis": "0.10%"},
        "source": source,
        "note": "",
    }


def load_candidates():
    rows = []
    if os.path.exists(RES_FINAL):
        df = pd.read_csv(RES_FINAL)
        df = df[df.slip == 0.001]          # 只取 0.1% 基准滑点那组画像
        for _, r in df.iterrows():
            rows.append(_mk_row(r["SW"], r["trail"], r["moh"], {
                "oos_med_pct": round(float(r["oos_med"]), 2), "oos_min_pct": round(float(r["oos_min"]), 2),
                "oos_max_pct": round(float(r["oos_max"]), 2), "oos_std": round(float(r["oos_std"]), 2),
                "full_med_pct": round(float(r["full_med"]), 2), "switches": int(r["sw_med"]),
                "maxdd_pct": round(float(r["dd_med"]), 2)}, "grid_final"))
    elif os.path.exists(RES_JOINT):
        df = pd.read_csv(RES_JOINT)
        df = df.sort_values("oos_med", ascending=False).head(24)
        for _, r in df.iterrows():
            rows.append(_mk_row(r["SW"], r["trail"], r["moh"], {
                "oos_med_pct": round(float(r["oos_med"]), 2), "oos_min_pct": round(float(r["oos_min"]), 2),
                "oos_max_pct": round(float(r["oos_max"]), 2), "oos_std": round(float(r["oos_std"]), 2),
                "full_med_pct": round(float(r["full_med"]), 2), "switches": int(r["sw_med"]),
                "maxdd_pct": round(float(r["dd_med"]), 2)}, "grid_joint"))
    # 始终附带研究守卫（防 candidates 源缺失）
    rows.append(_mk_row(30, 0.02, 5, {"oos_med_pct": 25.6, "oos_min_pct": 17.2, "oos_max_pct": 35.5,
                                      "oos_std": 5.6, "full_med_pct": 15.3, "switches": 310,
                                      "maxdd_pct": -23.4}, "research_fallback"))
    return rows


def backfill_journal():
    if not os.path.exists(JOURNAL):
        return 0
    n = 0
    with open(JOURNAL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            rec.setdefault("ts", "")
            if db.insert_trade(rec, None):   # 历史遗留无参数版本
                n += 1
    return n


def main():
    if not db.ensure_db():
        print("DB 不可用，无法播种")
        return
    rows = load_candidates()
    print(f"候选参数组合 {len(rows)} 个")
    # 去重（按 cfg 唯一键）
    seen = set()
    uniq = []
    for r in rows:
        k = (r["SLOPE_WINDOW"], r["SR_TRAIL_PCT"], r["MIN_HOLD_DAYS"])
        if k not in seen:
            seen.add(k); uniq.append(r)
    ids = db.seed_from_research(uniq)
    print(f"已入库 param_sets {len(ids)} 行")
    # 明确锁定初始现役 = 抗摩擦稳健最优(SW30/trail2%/moh5)
    winner = {"SLOPE_WINDOW": 30, "SR_TRAIL_PCT": 0.02, "MIN_HOLD_DAYS": 5,
              "DEF_MOM_DAYS": 10, "DEF_MOM_ENTER": 0.005, "DEF_MOM_EXIT": -0.008, "HOLD_N": 1}
    wid = db.get_param_set_id(winner)
    if wid:
        db.set_active(wid, "初始现役=抗摩擦稳健最优(grid_final: trail2%/SW30/moh5)")
        print(f"已设初始现役 param#{wid} = 抗摩擦稳健最优")

    n = backfill_journal()
    print(f"已回填历史成交 {n} 条")
    print("现役参数:", db.get_active_params())


if __name__ == "__main__":
    main()