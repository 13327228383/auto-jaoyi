# -*- coding: utf-8 -*-
"""
strategy_rotator.py —— 行情自适应的「多资产 ETF/指数 动量轮动」信号模块
================================================================================
为什么是它（对照用户的两点质疑）：
1) 数据效率高：标的范围只有 ~6 只 ETF/指数，回测/实盘都只需极少量数据，
   不存在"拉几千只股票等几十分钟"的问题（实盘一次 akshare 快照即可）。
2) 低频 + 自适应：双周调仓，单笔收益远大于手续费（不失血手续费）；
   带「市场状态识别(强势/震荡/危险 via 沪深300 vs MA200)」+「绝对动量过滤
   (最强国动量转负→切国债/货基)」+「趋势质量过滤(slope×R²)」+「跌破长期均线剔除」，
   随行情切换，而不是死扛一种形态。

文献依据（已检索）：
- 二八轮动：沪深300/中证500 ETF 动量切换，双弱转货基；2005-2022 回测年化~15%。
- 浙商证券《多资产ETF轮动》(2019-2026)：年化20.7%、最大回撤10.1%、Calmar 2.05；
  含市场状态划分(强势/震荡/危险)+风险平价+绝对动量过滤。
- klang.org.cn ETF 动量：加权 斜率×R² 打分、跌破60日线剔除、至少持有5日；
  2020-2026 实测年化29.2%。
- 中低频动量被大量实证支持为 A 股个人投资者最易落地的策略。

本模块只产出「目标持仓 + 理由 + 各标的动量分」，不做任何交易。
数据来源与获取解耦：select_rotation(prices) 接收预取好的收盘价序列，
实盘由 auto_run 用 data_center 取、回测由 backtest_rotation 取。
"""
import numpy as np
import pandas as pd

# 轮动标的池（宽基 + 商品 + 海外 + 防御），低成本、高流动性 ETF
# 2026-09 标的池扩展(backtest_universe_ext.py + add_one_universe.py)：加标普500(513500)增益最明显
#   —— 年化27.84%→29.48%、OOS年化61.96%→66.08%、Calmar1.57→1.66、换仓358→353(反降)；
#   与纳指(513100)同为美股代理但相关性互补→扩大跨市场分散。逐只隔离测试显示 159920/512010 系拖累
#   （全加11只反降至25.69%），故只采纳 513500。
ETF_UNIVERSE = {
    "510300": "沪深300ETF",
    "510500": "中证500ETF",
    "159915": "创业板ETF",
    "518880": "黄金ETF",
    "513100": "纳指ETF",
    "511010": "国债ETF",
    "513500": "标普500ETF",
}
# 防御资产（熊市/绝对动量转负时的承接）。
# 2026-09 回测(backtest_defense_assets.py)：风险-off 用黄金 518880 优于国债 511010
# （年化18.27%→27.27%/样本外51%/Calmar1.07→1.31，代价回撤-20.76%）。
# 自洽性：风险-off 触发依赖「黄金走升确认」，避险窗内黄金本身走强→顺势承接。
DEFENSIVE = "518880"

# 宏观/全球风险-off 监控（用跨资产价格，不读新闻，可严格回测）
# —— 用户要求"关注美国国情、战争"的专业落地：价格已把衰退/战争/恐慌瞬间计入，
#    故用 美股(纳指) / 黄金 / A股宽基 的联动做防守触发，而非人工读新闻（不可回测、会滞后）。
GLOBAL_US = "513100"          # 纳指ETF（美股代理）
GLOBAL_GOLD = "518880"        # 黄金ETF（避险代理）
MA_GLOBAL = 200              # 美股长期趋势均线（判断美股牛熊）
MA_GOLD = 60                 # 黄金趋势均线
GOLD_LOOK = 20               # 黄金近期涨幅窗口（确认避险）

# 可调参数（defaults 与回测/实盘一致；实盘可经 params 覆盖）
MOM_WINDOW = 20              # 朴素动量窗口（交易日）
SLOPE_WINDOW = 60            # slope×R² 打分窗口（交易日）
MA_LONG = 200                # 大盘长期趋势均线（判断牛熊）
# 同时持有最强前 N 只。当前 N=1（只持一只最强 ETF，非 top2）：
# 废弃 N=2 的原因 —— research/backtest_freq_sweep.py(2021+样本外)：
#   2W-N1 年化20.01%/回撤-10.03%/Calmar2.00/操作68，全面优于 2W-N2(18.05%/91次)：
#   单强票集中 + 更干净止损，赚更多、回撤相近、操作省约1/4 → 采纳 HOLD_N=1。
HOLD_N = 1
ABS_MOM_THRESHOLD = 0.0      # 绝对动量阈值：最强标的动量<=此值 → 转防御
SCORE_METHOD = "slope_r2"    # 'mom' 朴素动量 | 'slope_r2' 斜率×R²（推荐）
MA_FILTER = 60               # 跌破该均线则剔除（防御资产豁免）；None 关闭

# —— 风险-off 退抖 / 防御仓位模式（默认=原版行为，可经 params 覆盖；在变体回测中开启）——
RISK_CONFIRM_DAYS = 0        # 风险-off 需连续 N 天信号为真才确认（0=关闭，即原版立即触发）
RISK_EXIT_DAYS = 0           # 风险解除需连续 N 天信号为假才回撤（0=立即回撤）
# DEFENSE_MODE: 'defensive'(原版:全仓国债) | 'keep_trend'(国债+保留最强正动量档) | 'cash'(空仓等待)
DEFENSE_MODE = "defensive"
# 防御资产「自身动量门」窗口(交易日)。None/0=关闭(原版：风险-off 无脑持有防御资产)。
# 2026-09 回测(backtest_def_stop.py + validate_momg.py，先测B后落地)：
#   直接给防御资产上硬止损无效(触发后次日因 risk-off 仍触发而立即买回)；改为
#   防御资产自身 N 日动量转负→空仓等待(动量负值会持续、不会次日买回)。
#   门10d：年化33.56%(基线27.27%)、回撤-15.16%(基线-20.76%)、Calmar2.21；滚动17窗口
#   年化胜率76%、回撤浅胜率71%，跨牛熊段一致非过拟合 → 采纳 DEF_MOM_DAYS=10。
DEF_MOM_DAYS = 10
# —— 防御资产动量门「迟滞带」参数（2026-09 打掉零轴抖仓，回测验证后落地）——
# 现状：_def_momentum_ok 用 `m>0`（零阈值）。518880 的 10 日动量常在 0% 附近抖动，
#   下午 14:19 判负→清仓、14:26 判正→买回，一个下午来回两次开交易窗抢鼠标，纯 whipsaw。
# 迟滞（两者均默认 0.0=等效原版 `m>0`，保持向后兼容）：
#   DEF_MOM_ENTER：当前【未持有】防御资产 → 需动量 > DEF_MOM_ENTER 才进入（>0：要求明确转强）
#   DEF_MOM_EXIT ：当前【已持有】防御资产 → 需动量 > DEF_MOM_EXIT  才继续持有（<0：允许小回撤不抖出）
#   组合成带：动量在 (EXIT, ENTER) 区间内「维持原状」，杜绝零轴附近来回切。
#   【注意】这是【状态相关】判据，调用方(C2回测/实盘)须逐日回传上一交易日前是否持有防御资产。
DEF_MOM_ENTER = 0.0     # 进入阈值（未持有→需动量高于此才买入防御资产）
DEF_MOM_EXIT = 0.0      # 退出阈值（已持有→动量跌破此才卖出防御资产）


def momentum(close, window=MOM_WINDOW):
    """窗口动量：close[t]/close[t-window]-1；数据不足返回 nan。"""
    close = pd.Series(close).dropna()
    if len(close) < window + 1:
        return np.nan
    return float(close.iloc[-1] / close.iloc[-1 - window] - 1.0)


def slope_r2_score(close, window=SLOPE_WINDOW):
    """
    斜率×R² 趋势质量分（klang/浙商思路）：
    对最近 window 日收盘价做一元线性回归，score = 日均收益率(slope/mean) × R²。
    - 收益率为正且趋势平滑(R²高) → 高分；
    - 震荡/下行(R²低或 slope 负) → 低分，被自然淘汰。
    数据不足返回 nan。
    """
    s = pd.Series(close).dropna()
    if len(s) < window:
        return np.nan
    y = s.iloc[-window:].values.astype(float)
    x = np.arange(len(y), dtype=float)
    try:
        coeffs = np.polyfit(x, y, 1)
    except Exception:
        return np.nan
    yhat = np.polyval(coeffs, x)
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    slope_per_day = coeffs[0]
    slope_norm = slope_per_day / y.mean() if y.mean() != 0 else 0.0
    return float(slope_norm * r2)


def above_ma(close, n):
    """收盘价是否站在 n 日均线之上；数据不足返回 True（不剔除）。"""
    s = pd.Series(close).dropna()
    if len(s) < n:
        return True
    return bool(s.iloc[-1] > s.iloc[-n:].mean())


def get_regime(broad_close, ma_long=MA_LONG):
    """大盘状态：沪深300 收盘价 vs MA200。返回 'bull' / 'bear' / 'unknown'。"""
    broad_close = pd.Series(broad_close).dropna()
    if len(broad_close) < ma_long:
        return "unknown"
    ma = float(broad_close.iloc[-ma_long:].mean())
    return "bull" if broad_close.iloc[-1] > ma else "bear"


def select_rotation(prices, regime=None, params=None, universe=None):
    """
    输入 prices: {code: 收盘价序列(数组/Series)}，至少包含 universe 中各标的。
    返回 {'target': [code...] 或 'cash', 'reason': str, 'scores': {code: 动量分}}
    规则：
      - 用 SCORE 指定的方法算各标的分（默认 slope×R²）。
      - MA 过滤：非防御标的跌破 MA_FILTER 均线 → 直接剔除（避免接飞刀）。
      - 绝对动量过滤：最强标的分 <= ABS_MOM_THRESHOLD → 转防御(国债ETF)；若国债也弱→现金。
      - 熊市(regime='bear')且最强分不突出 → 偏防御/现金。
      - 否则持有分前 HOLD_N 名。
    universe: 参与轮动的标的集合（默认 ETF_UNIVERSE；也可传指数代码做回测）。
    """
    p = params or {}
    hold_n = p.get("HOLD_N", HOLD_N)
    thr = p.get("ABS_MOM_THRESHOLD", ABS_MOM_THRESHOLD)
    score_method = p.get("SCORE", SCORE_METHOD)
    mom_win = p.get("MOM_WINDOW", MOM_WINDOW)
    slope_win = p.get("SLOPE_WINDOW", SLOPE_WINDOW)
    ma_filter = p.get("MA_FILTER", MA_FILTER)
    defensive = p.get("DEFENSIVE", DEFENSIVE)
    uni = universe if universe is not None else ETF_UNIVERSE

    scores, passed = {}, {}
    for code in uni:
        s = prices.get(code)
        if s is None:
            scores[code] = np.nan
            passed[code] = False
            continue
        if score_method == "slope_r2":
            scores[code] = slope_r2_score(s, slope_win)
        else:
            scores[code] = momentum(s, mom_win)
        # 跌破长期均线则剔除（防御资产豁免）
        if ma_filter and code != defensive and not above_ma(s, ma_filter):
            passed[code] = False
        else:
            passed[code] = True

    valid = {c: v for c, v in scores.items()
             if (not (v is None or np.isnan(v))) and passed[c]}
    if not valid:
        return {"target": "cash", "reason": "无有效/未跌破均线的标的", "scores": scores}

    ranked = sorted(valid.items(), key=lambda x: x[1], reverse=True)
    best_code, best = ranked[0]

    # 绝对动量过滤：最强国也偏弱 → 防御
    if best <= thr:
        if defensive in valid and valid[defensive] > thr:
            return {"target": [defensive],
                    "reason": f"绝对动量<=0(最强{best_code}={best:.4f})，转防御资产({defensive})",
                    "scores": scores}
        return {"target": "cash",
                "reason": f"所有标的动量<=0(最强{best_code}={best:.4f})，空仓",
                "scores": scores}

    # 熊市：动量不突出也偏防御
    if regime == "bear" and best < (thr + 0.05):
        if defensive in valid and valid[defensive] > thr:
            return {"target": [defensive],
                    "reason": f"熊市且动量偏弱(最强{best:.4f})，转防御资产({defensive})",
                    "scores": scores}
        return {"target": "cash", "reason": f"熊市且动量弱(最强{best:.4f})，空仓",
                "scores": scores}

    hold = [c for c, _ in ranked[:hold_n]]
    return {"target": hold,
            "reason": f"持有动量前{hold_n}: {[(c, round(float(scores[c]), 4)) for c in hold]}",
            "scores": scores}


def global_risk_off(prices, us_code=GLOBAL_US, gold_code=GLOBAL_GOLD,
                    broad_code="510300", params=None):
    """
    跨资产「全球风险-off」判定（市场已把美国国情/战争/衰退/恐慌计入价格）。
    返回 (bool, reason)。
    触发条件（满足任一即视为风险-off）：
      1) 美股熊：纳指 < MA200（覆盖美国衰退、加息冲击、战争外溢）
      2) A股熊 + 黄金确认：沪深300 < MA200 且 黄金站在 MA_GOLD 之上、近 GOLD_LOOK 日上涨
         （黄金走升=真避险，避免把普通震荡误判成崩溃）
      3) 双熊：美股与A股同时 < MA200 → 强风险-off
    键名说明：实盘 prices 用 ETF 代码(513100/518880/510300)，回测可传映射键(ndx/gold/300)，
              通过 params 的 GLOBAL_US/GLOBAL_GOLD/BROAD 覆盖。缺少数据默认不触发（不误杀）。
    """
    p = params or {}
    ma_g = p.get("MA_GLOBAL", MA_GLOBAL)
    ma_gold = p.get("MA_GOLD", MA_GOLD)
    gold_look = p.get("GOLD_LOOK", GOLD_LOOK)
    us = prices.get(us_code)
    gold = prices.get(gold_code)
    broad = prices.get(broad_code)
    if us is None or broad is None:
        return False, "缺少美股/A股数据，不触发全局风险-off"
    us_s, broad_s = pd.Series(us).dropna(), pd.Series(broad).dropna()
    us_bear = bool(us_s.iloc[-1] < us_s.iloc[-ma_g:].mean()) if len(us_s) >= ma_g else False
    broad_bear = bool(broad_s.iloc[-1] < broad_s.iloc[-ma_g:].mean()) if len(broad_s) >= ma_g else False
    gold_confirm = False
    if gold is not None:
        g_s = pd.Series(gold).dropna()
        if len(g_s) >= ma_gold:
            g_up = bool(g_s.iloc[-1] > g_s.iloc[-ma_gold:].mean())
            g_ret = momentum(g_s, gold_look)
            gold_confirm = bool(g_up and (g_ret is not None and not np.isnan(g_ret) and g_ret > 0))
    if us_bear and broad_bear:
        return True, "美股+A股双熊(均<MA200)，强全球风险-off"
    if us_bear:
        return True, "美股熊市(纳指<MA200)，全球风险-off"
    if broad_bear and gold_confirm:
        return True, "A股熊市且黄金走升(避险确认)，风险-off"
    return False, "无全局风险-off信号"


def risk_off_raw_axis(price_map, params=None):
    """沿时间轴逐日计算原始风险-off 布尔信号（向量化，O(n)）。
    输入 price_map: {code: 已按日期对齐的收盘价 Series/DataFrame列}。
    返回 pd.Series[bool]，覆盖参与信号三标的的公共日期（NaN→False）。
    逻辑与 global_risk_off 完全一致，只是展开成逐日序列。
    """
    p = params or {}
    ma_g = p.get("MA_GLOBAL", MA_GLOBAL)
    ma_gold = p.get("MA_GOLD", MA_GOLD)
    gold_look = p.get("GOLD_LOOK", GOLD_LOOK)
    us_code = p.get("GLOBAL_US", GLOBAL_US)
    gold_code = p.get("GLOBAL_GOLD", GLOBAL_GOLD)
    broad_code = p.get("BROAD", "510300")

    idx = None
    for c in (us_code, broad_code):
        s = price_map.get(c)
        if s is not None:
            idx = pd.to_datetime(pd.Series(s).index) if idx is None else idx
            break
    if idx is None:
        return pd.Series(dtype=bool)
    out = pd.Series(False, index=idx)

    us = pd.Series(price_map[us_code], index=idx) if us_code in price_map else None
    broad = pd.Series(price_map[broad_code], index=idx) if broad_code in price_map else None
    gold = pd.Series(price_map[gold_code], index=idx) if gold_code in price_map else None
    if us is None or broad is None:
        return out

    us_ma = us.rolling(ma_g, min_periods=ma_g).mean()
    broad_ma = broad.rolling(ma_g, min_periods=ma_g).mean()
    us_bear = (us < us_ma).fillna(False)
    broad_bear = (broad < broad_ma).fillna(False)
    if gold is not None and len(gold) >= ma_gold:
        gold_ma = gold.rolling(ma_gold, min_periods=ma_gold).mean()
        gold_chg = gold.pct_change(gold_look)
        g_confirm = ((gold > gold_ma) & (gold_chg > 0)).fillna(False)
    else:
        g_confirm = pd.Series(False, index=idx)

    raw = us_bear.align(broad_bear, join="left")[0] | (broad_bear & g_confirm)
    return raw.reindex(idx).fillna(False).astype(bool)


def apply_confirm(raw, confirm_days, exit_days):
    """给原始布尔信号加"连续确认退抖"：
    状态转「开」需信号连续 confirm_days 天为真；转「关」需连续 exit_days 天为假。
    confirm_days<=0 时立即转开、exit_days<=0 时立即转关（等价原版）。"""
    if confirm_days is None:
        confirm_days = RISK_CONFIRM_DAYS
    if exit_days is None:
        exit_days = RISK_EXIT_DAYS
    state, up, dn = False, 0, 0
    states = []
    for v in raw.fillna(False):
        if v:
            up += 1; dn = 0
            if confirm_days <= 0 or up >= confirm_days:
                state = True
        else:
            dn += 1; up = 0
            if exit_days <= 0 or dn >= exit_days:
                state = False
        states.append(state)
    return pd.Series(states, index=raw.index)


def confirmed_risk_off_axis(price_map, params=None):
    """沿时间轴：先算原始风险-off(逐日)，再套连续确认退抖。返回状态 Series[bool]。
    供回测逐日驱动；实盘可把 RiskState 复刻到 auto_run 的 STATE 里。"""
    p = params or {}
    raw = risk_off_raw_axis(price_map, p)
    return apply_confirm(raw, p.get("RISK_CONFIRM_DAYS", RISK_CONFIRM_DAYS),
                         p.get("RISK_EXIT_DAYS", RISK_EXIT_DAYS))


def _def_momentum_ok(close, window, prev_held=False, enter=None, exit=None):
    """防御资产动量门（带迟滞带）：
    自身 N 日动量须满足——【已持有】→动量 > exit 才继续持有；【未持有】→动量 > enter 才进入。
    enter/exit 构成迟滞带：动量落在 (exit, enter) 内维持原状，打掉零轴附近来回切。
    两者默认取备 DEF_MOM_ENTER/DEF_MOM_EXIT；都为 0.0 时等效原版 `m>0`（无迟滞）。
    数据不足保守视为不允许(空仓)。"""
    m = momentum(close, window)
    if m is None or np.isnan(m):
        return False
    ent = DEF_MOM_ENTER if enter is None else enter
    ext = DEF_MOM_EXIT if exit is None else exit
    return m > (ext if prev_held else ent)


def _best_risk_positive(prices, params=None, universe=None):
    """风险-off 下"保留最强正动量档"：在非防御标的中，取动量分>0 且最高者；
    无正动量则返回 None（此时防御只留国债）。"""
    p = params or {}
    defensive = p.get("DEFENSIVE", DEFENSIVE)
    score_method = p.get("SCORE", SCORE_METHOD)
    mom_win = p.get("MOM_WINDOW", MOM_WINDOW)
    slope_win = p.get("SLOPE_WINDOW", SLOPE_WINDOW)
    uni = universe if universe is not None else ETF_UNIVERSE
    best_code, best_val = None, None
    for code in uni:
        if code == defensive:
            continue
        s = prices.get(code)
        if s is None:
            continue
        sc = slope_r2_score(s, slope_win) if score_method == "slope_r2" else momentum(s, mom_win)
        if isinstance(sc, (int, float)) and not (sc is None or np.isnan(sc)) and sc > 0:
            if best_val is None or sc > best_val:
                best_val, best_code = sc, code
    return best_code


def decide(prices, params=None, universe=None, risk_state=None, prev_def_held=False):
    """
    统一决策入口（实盘与回测共用，避免逻辑分叉导致"回测好、实盘歪"）。
    返回 {'target': [code...]或'cash', 'reason', 'scores', 'risk_off': bool}
    层序：
      0) 全局风险-off（MACRO_OFF 可关，用于对比）→ 切防御(国债ETF)；国债也弱→现金。
      1) 否则升级轮动(slope×R²) Top HOLD_N；自带 熊市/绝对动量/跌破均线 过滤。
    prev_def_held: 上一交易日前是否已持有防御资产（供防御动量门迟滞用；默认 False 无迟滞）。
    键名约定：prices 的键既可为 ETF 代码(实盘) 也可为映射键(回测)，由 params 的
              GLOBAL_US/GLOBAL_GOLD/BROAD/DEFENSIVE/ETF_UNIVERSE 指定。
    """
    p = params or {}
    defensive = p.get("DEFENSIVE", DEFENSIVE)
    us_code = p.get("GLOBAL_US", GLOBAL_US)
    gold_code = p.get("GLOBAL_GOLD", GLOBAL_GOLD)
    broad_code = p.get("BROAD", "510300")
    uni = universe if universe is not None else ETF_UNIVERSE
    mode = p.get("DEFENSE_MODE", DEFENSE_MODE)

    if not p.get("MACRO_OFF"):
        ro, reason_ro = global_risk_off(prices, us_code=us_code, gold_code=gold_code,
                                        broad_code=broad_code, params=p)
        # 退抖后的实际风险-off：外部已算好(risk_state)则用其；否则(确认天数为0)用原始信号
        effective_ro = ro if risk_state is None else bool(risk_state)
        if effective_ro:
            if mode == "cash":
                return {"target": "cash", "reason": f"全局风险-off→空仓等待({reason_ro})",
                        "scores": {}, "risk_off": True}
            if defensive in prices and prices[defensive] is not None:
                mf = p.get("MA_FILTER", MA_FILTER)
                if mf and not above_ma(prices[defensive], mf):
                    return {"target": "cash", "reason": f"全局风险-off但防御资产也弱→现金({reason_ro})",
                            "scores": {}, "risk_off": True}
                # 防御资产动量门（带迟滞）：已持有→m>exit 才留，未持有→m>enter 才进。
                # 数据不足保守视为不允许(return False)→空仓，避免无脑持有下跌中的防御资产。
                def_mom = p.get("DEF_MOM_DAYS", DEF_MOM_DAYS)
                if def_mom and not _def_momentum_ok(
                        prices[defensive], def_mom, prev_def_held,
                        enter=p.get("DEF_MOM_ENTER", DEF_MOM_ENTER),
                        exit=p.get("DEF_MOM_EXIT", DEF_MOM_EXIT)):
                    return {"target": "cash",
                            "reason": f"全局风险-off但防御资产自身动量转负({def_mom}日)→空仓等待({reason_ro})",
                            "scores": {}, "risk_off": True}
                if mode == "keep_trend":
                    keep = _best_risk_positive(prices, p, universe=uni)
                    tgt = [defensive] + ([keep] if keep else [])
                    return {"target": tgt,
                            "reason": f"全局风险-off→防御+保留强势档({reason_ro})" + (f"，附加{keep}" if keep else f"，无强势档只持防御({defensive})"),
                            "scores": {}, "risk_off": True}
                return {"target": [defensive], "reason": f"全局风险-off→防御资产({defensive})({reason_ro})",
                        "scores": {}, "risk_off": True}
            return {"target": "cash", "reason": f"全局风险-off→现金({reason_ro})", "scores": {}, "risk_off": True}

    broad_series = prices.get(broad_code, pd.Series(dtype=float))
    regime = get_regime(broad_series) if len(pd.Series(broad_series).dropna()) >= MA_LONG else "unknown"
    res = select_rotation(prices, regime, p, universe=uni)
    res["risk_off"] = False
    return res


if __name__ == "__main__":
    print("ETF_UNIVERSE:", ETF_UNIVERSE)
    # 自测：上升趋势 vs 下行/震荡
    up = list(np.linspace(1, 1.3, 80)) + [1.31, 1.32, 1.33]
    down = list(np.linspace(1.2, 0.9, 80)) + [0.89, 0.88, 0.87]
    choppy = [1 + 0.05 * np.sin(i / 3) for i in range(80)]
    print("上行 slope_r2:", round(slope_r2_score(up), 5))
    print("下行 slope_r2:", round(slope_r2_score(down), 5))
    print("震荡 slope_r2:", round(slope_r2_score(choppy), 5))
    print("示例 select_rotation:",
          select_rotation({"510300": up, "511010": [1, 1, 1, 1.01], "518880": down}))
