# -*- coding: utf-8 -*-
"""启动提示：参数定期复盘 + 未落地研究脚本对账提醒（每 GAP_DAYS 天提示一次，避免每次启动刷屏）。
在启动 bat 于 `python auto_run.py` 之前调用。
   python review_hint.py        按周期提示；未到周期则打印一行说明，不打扰。
   python review_hint.py --done 标记"已完成本轮复盘"（下次按阈值再提示）。
状态存 review_state.json（放 D:\\LICAISOFT\\Auto-Jaoyi）。
"""
import os, sys, json, datetime

GAP_DAYS = 7
STATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "review_state.json")


def _load():
    if os.path.exists(STATE):
        try:
            with open(STATE, encoding="utf-8") as fp:
                return json.load(fp)
        except Exception:
            return {}
    return {}


def _save(d):
    try:
        with open(STATE, "w", encoding="utf-8") as fp:
            json.dump(d, fp, ensure_ascii=False, indent=2)
    except Exception:
        pass


def main():
    today = datetime.date.today()
    state = _load()

    if "--done" in sys.argv:                      # 人工标记"已完成复盘"
        state["last_done"] = today.isoformat()
        _save(state)
        print("[复盘提醒] 已标记完成本轮参数复盘；下次将在 %d 天后再次提醒。" % GAP_DAYS)
        return

    def _day(s):
        try:
            return datetime.date.fromisoformat(s) if s else None
        except Exception:
            return None

    done = _day(state.get("last_done"))
    hint = _day(state.get("last_hint"))

    # 是否需要提醒：从未标记"完成"时按"上次提示"节流；标记过"完成"则按完成日节流。
    needs = False
    if done is None:
        needs = (hint is None) or ((today - hint).days >= GAP_DAYS)
    else:
        needs = (today - done).days >= GAP_DAYS

    if not needs:
        ref = done or hint
        d = (today - ref).days if ref else 0
        print("[复盘提醒] 上次参数复盘/提示：%s（距今 %d 天，未到 %d 天周期，本次不重复打扰）"
              % (ref or "无", d, GAP_DAYS))
        return

    state["last_hint"] = today.isoformat()
    _save(state)
    print("=" * 62)
    print("[复盘提醒] 距上次参数复盘已超过窗口，建议本轮：")
    print(" 1) 定期换数据窗口重验关键参数：跟踪止损 3% (SR_TRAIL_PCT/DEF_PEAK_STOP)、")
    print("    防御动量 10 日 (DEF_MOM)、防御资产 518880、标的池 (ETF_UNIVERSE 6沪+1深+513500)。")
    print(" 2) 未落地研究脚本逐一对账取舍（结论未绑定到 auto_run 生产参数）：")
    print("    backtest_sr_v3 / cash_deploy / intraday_trail / per_target_trail /")
    print("    def_stop / defensive_variants / reentry_cooldown / swap_fail /")
    print("    topup_early / freq_sweep  —— 见 research\\*.py")
    print(" 3) 完成后运行： python review_hint.py --done")
    print("=" * 62)


if __name__ == "__main__":
    main()