# -*- coding: utf-8 -*-
"""改动后运行时自检：
1) strategy_rotator 能加载、DEFENSIVE 已改为 518880；
2) 风险-off 且防御资产(黄金)在 MA60 上方 → decide() 返回 ["518880"]；
3) 风险-off 但防御资产跌破 MA60 → 正确回退现金(应急分支)。
"""
import sys, numpy as np
sys.path.insert(0, r"d:\LICAISOFT\Auto-Jaoyi")
sys.path.insert(0, r"d:\LICAISOFT\Auto-Jaoyi\research")
import strategy_rotator as sr

print("DEFENSIVE =", sr.DEFENSIVE)
assert sr.DEFENSIVE == "518880", "DEFENSIVE 未生效！"

N = 250
t = np.arange(N, dtype=float)

# 黄金：确定性上升（末值显著高于 MA60），从而规避"随机游走跌破均线"的干扰
gold = 100.0 + 0.15 * t            # 从100升到 ~137
# 其余标的：温和区间震荡（避免趋势分干扰主目标）
flat = 100.0 + 5.0 * np.sin(t / 6.0)

def build(gold_series):
    prices = {}
    for c in sr.ETF_UNIVERSE:
        prices[c] = gold_series if c == "518880" else flat
    prices["513100"] = np.linspace(100.0, 80.0, N)   # 纳指熊 → 触发风险-off
    return prices

# 场景1：黄金走强 → 风险-off 应切到黄金
res = sr.decide(build(gold))
print("场景1 黄金强 >> target =", res["target"], "| risk_off =", res.get("risk_off"))
assert res["risk_off"] is True, "应触发风险-off"
assert res["target"] == ["518880"], f"应返回黄金，实际 {res['target']}"

# 场景2：黄金转弱（跌破 MA60）→ 回退现金（应急分支）
gold_weak = np.concatenate([100.0 + 0.15*np.arange(50),    # 先涨
                            128.0 - 0.30*np.arange(N-50)])  # 后跌到 68，末值 < MA60
res2 = sr.decide(build(gold_weak))
print("场景2 黄金弱 >> target =", res2["target"], "| risk_off =", res2.get("risk_off"))
assert res2["risk_off"] is True
assert res2["target"] == "cash", f"黄金跌破MA60应回现金，实际 {res2['target']}"

# 场景3：黄金在 MA60 上方但自身动量门(10d)转负 → 应空仓等待（本次落地核心逻辑）
top = 100.0 + 0.15 * np.arange(240)
tail = np.linspace(top[-1], top[-1] * 1.05, 10)   # 末段先再创新高
gold_gate = np.concatenate([top, tail, np.linspace(tail[-1], tail[-1] * 1.03, 5),  # 微抬
                            np.linspace(tail[-1]*1.03, tail[-1]*0.97, 10)])        # 末10日动量转负但仍在MA60上
res3 = sr.decide(build(gold_gate))
print("场景3 黄金MA60上但动量门转负 >> target =", res3["target"], "| risk_off =", res3.get("risk_off"))
assert res3["risk_off"] is True
assert res3["target"] == "cash", f"动量门应使空仓，实际 {res3['target']}"

print("返回字段齐全:", all(k in res for k in ("target", "reason", "scores", "risk_off")))
print("OK 自检通过：DEFENSIVE 生效、风险-off 防御切换与应急回退、动量门均正确")