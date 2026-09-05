# -*- coding: utf-8 -*-
# =============================================================================
# auto_run.py —— 全自动交易入口（"开机即无人驾驶"）
#
# ⚠️ 风险提示（务必先读）：
#   1. 本程序可接管真实账户下单，涉及真实盈亏。模式（模拟/实盘）不含任何配置项，
#      完全由启动后同花顺客户端「账户通道下拉框」自动识别（客户端记住上次选的是模拟还是实盘）。
#   2. 任何策略都不保证盈利。本程序采用「多资产 ETF 动量轮动」：已在 2015-2026
#      真实指数历史上回测（年化约 6.7%、最大回撤约 -26%，样本外 2021-26 约 1%/年、
#      回撤控制稳健）。它提高存活率与相对胜率，不是稳赢印钞机，要有 -26% 级别回撤的心理准备。
#   3. 轮动为低频（约双周再平衡），自带「市场状态识别+绝对动量过滤+防御切换(国债ETF)」，
#      单笔收益远大于手续费；含 S/R 支撑阻力动态止损 + 跟踪止损(峰值回撤6%，锁利润) + 单日-3%熔断，
#      止损均不受5天持有锁限制，立即执行。
#   4. 数据源全部免费（akshare/baostock），无付费接口、无 AI token 消耗。
#
# 使用：
#   python auto_run.py            # 一直运行，每日定时检查并再平衡
#   创建空文件 KILL_SWITCH         # 立即全停（删除后恢复）
#   窗口里设为登录时启动，实现"打开电脑就自动跑"。
#   模式由客户端自动识别，无模拟/实盘配置。
# =============================================================================
import datetime
import sys
import time
import os
import subprocess
import json
import configparser
import logging
import pandas as pd

import data_center as dc
import strategy_rotator as rot
from pywinauto import Desktop

try:
    # research/sr_levels.py：支撑/阻力价位计算（已改为 scipy-free 纯 numpy，实盘环境可直接 import）
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "research"))
    import sr_levels as _srl
except Exception:
    _srl = None


# ===== 给 easytrader 的委托弹窗处理器打补丁 =====
# 原因：本客户端"委托价格小数部分为3位，是否继续？"的文案与 easytrader 内置匹配（"小数价格应为"）不一致，
#      导致确认框不被处理 → 单子被取消 → 假 success、永不成交。补丁：提示信息里含"小数"→ 回车确认。
try:
    from easytrader import pop_dialog_handler as _pdh
    _orig_handle = _pdh.TradePopDialogHandler.handle

    def _patched_handle(self, title):
        if title == "提示信息":
            try:
                _content = self._extract_content() or ""
            except Exception:
                _content = ""
            if "小数" in _content:  # 委托价格小数3位等 → 回车确认(是)
                try:
                    self._submit_by_shortcut()
                except Exception:
                    pass
                return None
        return _orig_handle(self, title)

    _pdh.TradePopDialogHandler.handle = _patched_handle
except Exception:
    pass

try:
    from plyer import notification
except Exception:
    notification = None

# ===== 把 easytrader 读持仓/余额用的验证码识别从 tesseract 换成 ddddocr =====
# 根因：easytrader 默认 grid_strategy=Copy，读持仓=模拟 Ctrl+C 复制表格，会弹"正在拷贝数据"
#       验证码；而 easytrader 内置 captcha_recognize 依赖 pytesseract(tesseract)，本机未装
#       → 弹出后识别失败点"取消"，复制不成功，broker.position/balance 读不到真实数据。
#       本补丁把 grid_strategies 模块内的 captcha_recognize 局部绑定替换为 ddddocr 实现
#       （grid_strategies 用 `from easytrader.utils.captcha import captcha_recognize` 直接引用
#        函数对象，只改 utils.captcha 属性无效，必须改 grid_strategies 的绑定）。
#       与脚本内 _solve_captcha 是同一套 ddddocr 思路，互为补充。
try:
    from easytrader import grid_strategies as _gs
    _GS_ORIG = _gs.captcha_recognize
    import ddddocr as _dd_ocr

    def _dd_captcha_recognize(img_path):
        """用 ddddocr 识别同花顺"正在拷贝数据"验证码图片（4位数字+干扰线）。
        实测易股传的是小尺寸强干扰验证码图，直接识别易错 → 先预处理：
        灰度→自动对比度→(可选)二值化→放大4x→锐化；识别结果再取纯数字。
        仍失败才回退原 tesseract 路径（通常也失败）。"""
        import io as _io
        try:
            import PIL.Image as _im
            import PIL.ImageOps as _op
            import PIL.ImageFilter as _f
            img = _im.open(img_path)
            img = img.convert("L")                 # 灰度
            img = _op.autocontrast(img)            # 自动对比度
            try:
                img = img.point(lambda p: 255 if p > 130 else 0)  # 二值化
            except Exception:
                pass
            if img.width < 400:
                img = img.resize((img.width * 4, img.height * 4), _im.LANCZOS)  # 放大
            img = img.filter(_f.SHARPEN)
            _buf = _io.BytesIO()
            img.save(_buf, "PNG")
            _data = _buf.getvalue()
            r = _dd_ocr.DdddOcr(show_ad=False).classification(_data)
            return "".join(ch for ch in (r or "") if ch.isdigit())
        except Exception:
            try:
                r = _dd_ocr.DdddOcr(show_ad=False).classification(img_path)
                return "".join(ch for ch in (r or "") if ch.isdigit())
            except Exception:
                return _GS_ORIG(img_path)  # 失败回退原 tesseract 路径（通常也失败）

    _gs.captcha_recognize = _dd_captcha_recognize
except Exception:
    pass

# ===== 关键：修复 easytrader 读持仓时漏解验证码 =====
# 现状：easytrader Copy 策略复制持仓后，靠 `window(title_re="验证码")` 判断是否弹验证码（grid_strategies.py L100）。
# 但同花顺这个"检测到您正在拷贝数据"弹窗标题是"提示"（User 截图实证），匹配不到 → easytrader 直接读剪贴板(空)
# → broker.position/balance 返回 None。我们增强版 _solve_captcha 能识别"提示"标题弹窗，故在它读剪贴板前一环强制执行。
try:
    _CS_OGC = _gs.Copy._get_clipboard_data

    def _patched_get_clipboard_data(self):
        # 复制(^A^C)已触发验证码 → 轮询探测并自动解，再走原读取。
        # [2026-09 关键修复] ① 探测从 pywinauto Desktop.descendants 改为 win32gui 原生枚举
        #   （空标题自绘弹窗 pywinauto 探不到 → 之前 _solve_captcha 从没被触发 → 只反复复制不填码）；
        #   ② 验证码弹窗是复制后【异步】弹出的，需轮询等待它出现，不能只查一次（否则超时取空剪贴板）。
        try:
            for _i in range(6):
                if _find_captcha_dialog_win32() is not None:
                    _solve_captcha(self._trader)
                    time.sleep(0.6)
                    if _find_captcha_dialog_win32() is None:
                        break            # 已解掉 → 走原读取
                else:
                    time.sleep(0.6)      # 弹窗未现 → 等异步渲染
        except Exception:
            pass
        return _CS_OGC(self)

    _gs.Copy._get_clipboard_data = _patched_get_clipboard_data
except Exception:
    pass

# ----------------------------- 全局配置（按你情况改这里） -----------------------------
CFG = {
    "PAPER": True,                 # 运行期标志：由「客户端通道下拉框」自动识别（模拟盘/实盘），非配置项，见 connect_broker()
    "TOTAL_FUND": 10000.0,         # 兜底资金（读不到账户时用；一般由客户端账户决定）
    "MAX_DAILY_LOSS_PCT": 0.03,    # 单日组合亏损超 3% 暂停当日再平衡（安全闸）
    "HARD_STOP_PCT": -0.08,        # 单只持仓固定-8%硬止损（作为 S/R 动态止损的安全兜底：无有效支撑/数据不足/未开启时回退此项）
    # —— S/R 支撑阻力动态止损（回测2026-08 验证：V2a S/R动态止损 优于 固定-8%，年化7.44%→10.29%、回撤-17.34%→-12.30%）——
    "SR_STOP": True,              # True=用支撑阻力动态止损替换固定-8%（无有效支撑时回退-8%）
    "SR_WIN": 120,                # 计算支撑/阻力的回看窗口(交易日)
    "SR_ORDER": 3,                # 极值 order
    "SR_MERGE": 2.0,              # 聚类百分比：相近极值聚成同一价位
    "SR_MIN_STRENGTH": 2,         # 价位最少 touch 次数(强度)才采用
    "SR_STOP_CAP_LO": 0.04,       # 支撑距成本太近(<4%)->回退固定-8%
    "SR_STOP_CAP_HI": 0.15,       # 支撑距成本太远(>15%)->封顶-15%
    # —— S/R 入场过滤（回测2026-08 验证：V2d S/R止损+入场过滤 优于 V2a，累计2.763x/年化11.94%/回撤-12.64%；
    #    逻辑：买价落在最近阻力 SR_ENTRY_BUF 以内则跳过该标的，不在压力位前追高）——
    "SR_ENTRY_FILTER": True,      # True=开启入场过滤（对齐 V2d，默认开；不利改动绝不上）
    "SR_ENTRY_BUF": 0.01,         # 买价在最近阻力 1% 以内即视为"贴近阻力"而跳过
    # —— 跟踪止损（回测 V3 2026-09 验证 trail6% 优于无；2026-09 再升级：普通标的 4% vs 6%
    #    逐只隔离+全局部署对比 backtest_per_target_trail.py —— 统一 4% 全样本年化 28.06%/-14.38%、
    #    样本外 73.96%，全面优于 6%(25.34%/-18.68%)与现混合，仅沪深300略偏好6%、国债无差别；
    #    原理=随价格创新高把止损上移，锁利润、少回吐）——
    "SR_TRAIL": True,             # True=在 S/R 动态止损之上叠加跟踪止损（取两者较高者）
    "SR_TRAIL_PCT": 0.03,         # 跟踪回撤阈值：从持仓最高价回撤 3% 即离场，只升不降（普通标的+防御资产统一；2026-09 诚实口径回测：4%→3% 年化29.6%→34.3%、回撤持平、换手不变）
    # —— 防御资产(黄金518880) 高点回落卖出（2026-09 回测 backtest_gold_peak_stop.py：4% 最优；
    #   与普通标的统一 4%，见 backtest_per_target_trail.py）——
    #   旧版对 DEFENSIVE 明确排除跟踪止损（code != CFG['DEFENSIVE']），黄金冲高获利后利润守不住，
    #   例：某日盘中盈利 900+ → 回落只剩 500+ 未锁定。回测：给防御黄金叠加「从持仓最高收盘价
    #   回撤 X% 卖出」→ 4% 年化 15.63→20.52%(+4.9pp)、回撤持平(-22.55%)、换手仅+8次；
    #   样本外(2021+) 37.85% vs 28.36% → 采纳 4%（只升不降，锁利润少回吐）。
    "DEF_PEAK_STOP": 0.03,        # 防御资产专用：从持仓以来最高价回撤 3% 即离场（值与 SR_TRAIL_PCT 统一；2026-09 同步收紧）
    # —— 止损后「再入冷却」（2026-09 回测 backtest_reentry_cooldown.py：N=2 交易日最优，
    #   年化+0.33pp、样本外+0.99pp、省换手；N≥3 明显变差——错过强势反弹）——
    #   某标的触发跟踪/S-R止损卖出后，N 个交易日内禁止再买回，防「卖了当天又接回」抖振。
    #   数值单位=交易日；0=关闭（不限）。冷却按「实测交易日历」精确计数。
    "REENTER_COOLDOWN_DAYS": 2,
    # —— A项：只读快循环频率（秒）——
    #   峰值/止损判定从「120s决策周期」抽成独立「5-10s只读快巡检」：只读 sina 行情+本地记录，
    #   更新持仓峰值、触发止损判定；不读客户端、不点鼠标（不触发复制验证码）。
    #   仅在真正触发止损时才碰客户端操作。价值=从120s内冲高回落漏更新的峰值中更早离场。
    #   （回测 backtest_intraday_trail.py 显示"日内高低判定"过渡版会让化降4.66pp，但那是极端上界；
    #   本实现保持同一4%阈值、仅提升峰更新频率，实测为准。）
    # 挡损快循环巡检周期(秒)。回测 research/backtest_sweep_freq.py：10~120s 对收益/回撤
    # 零影响(捕捉率99.86%->98.36%，远小于6%止损阈值)，60s 把 sina 请求量降到 1/6，
    # 显著降低限流风险，是"请求安全 vs 精度"的平衡点。
    "STOP_SWEEP_INTERVAL": 60,
    # 盘中自动调参评估间隔(秒)。交易时段内每隔 TUNE_INTERVAL 呼叫一次 auto_tuner.run()，
    # 满足护栏才切现役参数 → 决策边界前 refresh_active_params 即时热更；
    # 收紧 SR_TRAIL_PCT 会按新阈值从峰值重算跟踪止损，跌破即真实离场（用户确认允许）。
    "TUNE_INTERVAL": 1800,
    "REBALANCE_CHECK_TIME": (14, 50),  # 每日该时刻检查是否需再平衡
    # 距上次再平衡至少 N 自然日才允许换仓。
    # R3 堵漏：MIN_HOLD_DAYS 已从 tuning_db.TUNABLE_KEYS 移出（交易节奏，禁止盘中热改）。
    # 此处固定为代码常量，与现役 grid_final(联合最优 SW30/trail2%/moh5) 一致，行为不变。
    # （历史对照：单体 backtest_minhold 建议 5→1 略优 +1.25pp，但动态调参不该改持仓锁，故不采纳。）
    "MIN_HOLD_DAYS": 5,
    # 行情缓存过期保护(交易日)：取数连续失败时，允许复用上次缓存价做决策的最大期龄。
    # 超过则不复用 → 该标的本轮视为无数据 → 决策自然趋于空仓/防御，避免"该避险没避险"。
    # 注意：执行/止损永远是实时价(_l1_price)，本键只影响决策层(fetch_etf_closes)。
    "MAX_STALE_PRICE_DAYS": 2,
    "ETF_UNIVERSE": {              # 轮动标的池（宽基+商品+海外+防御），低成本高流动
        "510300": "沪深300ETF", "510500": "中证500ETF", "159915": "创业板ETF",
        "518880": "黄金ETF", "513100": "纳指ETF", "511010": "国债ETF",
        # 2026-09 标的池扩展(backtest_universe_ext.py)：加标普500(513500)增益最明显
        # 年化27.84%→29.48%、OOS 61.96%→66.08%、Calmar1.57→1.66、换仓反降到353；
        # 与纳指(513100)复相关但互补→扩大跨市场分散；逐只隔离 159920/512010 是拖累，只采纳这个。
        "513500": "标普500ETF",
    },
    # 境内股票ETF = T+1（当日买入最早次日才能卖出）。卖出门禁据此判断"当日买入不可当日卖"，
    # 把 T+1 制度约束显式化：优雅延迟到次日，避免"下单失败→报错/暂停当日交易"的粗糙处理，
    # 也避免当日买入的 T+1 止损/换仓卖出时造成双持仓或抛异常。回测 backtest_t1_constraint.py 佐证
    # T+1 延迟止损的代价；这里保证至少不因制度约束引发错误停摆。
    "T1_CODES": {"510300", "510500", "159915"},
    "BROAD": "510300",             # 大盘状态判定用的宽基（沪深300ETF）
    # 防御资产（熊市/绝对动量转负承接）。2026-09 回测(backtest_defense_assets.py)证明：
    # 风险-off 用黄金 518880 优于国债 511010 —— 年化18.27%→27.27%、样本外33%→51%、
    # Calmar1.07→1.31；代价回撤-17.14%→-20.76%。黄金在避险窗口既避跌又有上行，
    # 且自洽于风险-off 触发里的「黄金走升确认」(GLOBAL_GOLD=相同标的)。
    "DEFENSIVE": "518880",
    # 防御资产「自身动量门」窗口(交易日)。风险-off 时防御黄金自身 N 日动量转负→空仓等待
    # （不买回）。直接上硬止损无效(次日因 risk-off 仍触发立即买回)；动量门才真正压缩防御回撤。
    # 回测 backtest_def_stop.py + validate_momg.py：门10d 全期年化33.56%/回撤-15.16%/Calmar2.21
    # (基线27.27%/-20.76%/1.31)；滚动17窗口年化胜率76%、回撤浅胜率71% → 采纳 10。
    "DEF_MOM_DAYS": 10,
    # 防御资产动量门「迟滞带」：已持有→动量跌破退出阈值才卖；未持有→动量突破进入阈值才买。
    # 2026-09 回测(backtest_def_mom_hyst.py)：进+0.5%/退-0.8% 年化12.93%->14.62%、回撤持平、
    # 换仓96->79(-18%)。打掉 518880 10日动量在零轴附近抖动导致的"清仓→又买回"无谓换仓。
    # （0.0=关闭迟滞=等效原版 m>0）
    "DEF_MOM_ENTER": 0.005,        # 进入阈值：未持有防御资产，动量需 > +0.5% 才买入
    "DEF_MOM_EXIT": -0.008,        # 退出阈值：已持有防御资产，动量跌破 -0.8% 才卖出
    # —— 宏观/全球风险-off（用跨资产价格，不读新闻；专业落地"关注美国国情/战争"）——
    "GLOBAL_US": "513100",         # 纳指ETF（美股代理）
    "GLOBAL_GOLD": "518880",       # 黄金ETF（避险代理）
    "MA_GLOBAL": 200,              # 美股长期趋势均线（判断美股牛熊）
    "MA_GOLD": 60,                 # 黄金趋势均线（确认真避险）
    "GOLD_LOOK": 20,               # 黄金近期涨幅窗口（确认避险）
    "MACRO_OFF": False,            # 调试用：True 关闭宏观层（仅轮动）
    # 轮动参数（与 backtest_compare.py / backtest_rotation.py 一致）
    "SCORE": "slope_r2",           # 'slope_r2' 斜率×R² 趋势质量分 | 'mom' 朴素动量
    "SLOPE_WINDOW": 30,            # 打分窗口（交易日）。2026-09 实盘口径重测(grid_exec_close/grid_refine，当日收盘成交+双边滑点0.1%+佣金，滚动预热6起点+评估2021+中位年化，min_hold=1 trail=3%黄金也3%)：SW30 20.2% ≥ SW26 20.8%精扫峰值 ≈ SW28 20.3%（26-30稳健高原，滑点0.05%~0.2%敏感性均领先SW32/SW40）；原SW20 13.6%、SW40 15.6%、换仓450→370、回撤-28.9→-23.4。旧 megagrid 288组合"SW20最优35.9%"为当日收盘零滑点口径(白拿确认日跳涨+无摩擦，虚高SW20)，故从20改为30
    "MA_FILTER": 60,               # 跌破该均线则剔除（防御资产豁免）；0 关闭
    "HOLD_N": 1,                   # 同时持有前几名。回测 research/backtest_freq_sweep.py(2021+样本外)：
                                   #   2W-N1 年化20.01%/回撤-10.03%/Calmar2.00/操作68 全面优于 2W-N2(18.05%/91次)，
                                   #   单强票集中+更干净止损，赚更多、回撤相近、操作省约1/4 → 采纳 HOLD_N=1。
    # 购买比例（目标内资金分配）：False=等权；True=动量平方加权(mom2)。
    # 回测 research/backtest_alloc_v4.py(2016-2026，S/R止损+入场过滤+跟踪止损6%)：
    #   equal 年化16.17%/回撤-8.57%  vs  mom2 年化18.05%/回撤-10.17%，样本外(2021+) equal 29.41%
    #   vs mom2 33.03%/Sharpe 1.47——趋势方向显著更优且不增回撤，故并入实盘。
    "MOM2_WEIGHT": True,
}

# 模式完全以"客户端通道下拉框"为准（客户端记住上次选的是实盘还是模拟），不留任何模拟/实盘配置。
# 仅保留 config.ini [mode] 的 total_fund（兜底本金）可配置，模式相关已删除。
try:
    _mode_cfg = configparser.ConfigParser()
    _mode_cfg.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.ini"), encoding="utf-8")
    if "mode" in _mode_cfg:
        _m = _mode_cfg["mode"]
        if _m.get("total_fund"):
            try:
                CFG["TOTAL_FUND"] = float(_m["total_fund"])
            except Exception:
                pass
except Exception:
    pass
# 不再支持命令行 live/paper：模式由客户端登录账户自动识别（下拉框含"模拟"→模拟盘，否则→实盘）。
KILL_SWITCH_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "KILL_SWITCH")
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "auto_trade.log")
LOCK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "auto_run.lock")  # 单实例锁，防双开下重单
STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "auto_state.json")  # 跨开机持久化（上次调仓日）
OWNED_POS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "owned_positions.json")  # 我们自己买的ETF持仓记录（日常决策用，不读客户端=不触发复制验证码）

# 运行状态（供主循环读取：风险-off 时允许突破 MIN_HOLD_DAYS 立即防守）
STATE = {"risk_off": False, "last_reason": "", "daily_loss_halt": False, "day_open_equity": None,
         "confirm_halt": False}  # 下单核对失败 → 暂停当日交易（防重复下单）


class T1TradeDeferred(Exception):
    """T+1 制度约束：当日买入的境内股票ETF最早次日才能卖出。
    卖出被此门禁拦下时抛出，让主循环优雅暂停本次调仓——不误判为下单失败、
    不触发 confirm_halt 停摆、不急着买入新目标造成双持仓；次日可卖时自然重试。"""


def _t1_today_sell_blocked(code, pos):
    """T+1 标的是否「今日刚买入→今日不可卖」。
    pos: owned_positions 的对应记录 dict，含 'buy_date'(真实买入日)。
    当日买入的 T1(「buy_date==今天」)→ 不可卖返回 True；其余(隔日/非T1/无记录)→ False。
    注意对账(_sync_owned_from_client)会刷新 date 但保留 buy_date，故此处用 buy_date 判真实买入日。"""
    if code not in set(CFG.get("T1_CODES", set())):
        return False
    bd = (pos or {}).get("buy_date")
    return bool(bd) and bd == _today()

# ----------------------------- 单实例锁（防双开下重单） -----------------------------
import atexit

def _remove_lock():
    try:
        if os.path.exists(LOCK_FILE):
            os.remove(LOCK_FILE)
    except Exception:
        pass

def _pid_alive(pid):
    """跨平台进程存活探测。注意：Windows 上 os.kill(pid, 0) 并非纯探活，可能向目标
    发送信号/引发异常，不能用于单例锁；改用 psutil.pid_exists 纯查询更稳。"""
    try:
        import psutil
        return bool(pid and psutil.pid_exists(int(pid)))
    except Exception:
        # psutil 不可用时按"不确定"保守处理：若读到了合法 PID 就当作已存活（宁拒放，防双开）
        return pid is not None

def acquire_singleton_lock():
    """同一时刻只允许一个 auto_run 实例；已有存活实例时返回 False（调用方应退出）。"""
    try:
        if os.path.exists(LOCK_FILE):
            try:
                with open(LOCK_FILE, "r", encoding="utf-8") as f:
                    old = int(f.read().strip())
                if _pid_alive(old):
                    logger.error(f"已有实例运行(PID={old})，拒绝双开，本进程退出")
                    return False
                logger.warning(f"锁文件中的 PID={old} 已不在，视为死锁，本进程接管")
            except (ValueError, OSError):
                pass  # 锁文件损坏/无法解析，覆盖接管
        with open(LOCK_FILE, "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))
        # 二次校验：防止两进程同刻竞争(都以为自己拿到锁)。睡眠后回读，
        # 若锁里不是自己的PID，说明另一实例抢先，本进程退出。
        time.sleep(0.3)
        try:
            with open(LOCK_FILE, "r", encoding="utf-8") as f:
                if int(f.read().strip()) != os.getpid():
                    logger.error("单实例锁竞争失败(另一实例已持锁)，本进程退出")
                    return False
        except Exception:
            pass
        atexit.register(_remove_lock)
        return True
    except Exception as e:
        logger.warning(f"单例锁检查异常(放行启动)：{e}")
        return True

# ----------------------------- 日志 + 通知 -----------------------------
logger = logging.getLogger("auto_run")
logger.setLevel(logging.INFO)
_fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
_fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(_fh)
_ch = logging.StreamHandler()
_ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(_ch)

# ----------------------------- 参数落库 + 自动调参（DB 单一事实源） -----------------------------
# 若本地 MySQL 可用，用现役参数覆盖 CFG 的可调键（SLOPE_WINDOW/SR_TRAIL_PCT/...），
# decide_target/止损/最小持仓 等消费方随 CFG 一并继承。DB 连不上则保持本地回测稳健最优守卫，
# 绝不停摆、不误下单。当前现役参数缓存于 _ACTIVE_PARAM（journal 打版本号用）。
_ACTIVE_PARAM = {}
_ACTIVE_PARAM_ID = None
_DB_LAST_HASH = None     # 上次读取的现役参数指纹，用于热更去重
# R7' 堵漏：DB 下发可调参数的【安全值域】。越出安全域的键整键拒绝(不入 CFG、打日志)。
# 值域贴近研究生效域(grid_joint SW20-40/trail2-6%，外扩防呆)；越界=拒绝，绝不因 DB 误写极端值改变交易行为。
# "单次收紧≤步长"的逐步安全由 auto_tuner 分步切换承担(落真实 param_set 行分步逼近)，故这里始终 1:1 应用
# (applied==DB)、版号精确——不会出现"实际参数≠DB目标"的折中孤儿态。
TUNABLE_BOUNDS = {
    "SLOPE_WINDOW": (10, 80),        # 打分窗口(交易日)：研究域 20-40，外扩防呆
    "SR_TRAIL_PCT": (0.01, 0.08),    # 普通标的跟踪止损%
    "DEF_PEAK_STOP": (0.01, 0.08),   # 防御资产高点回落%
    "DEF_MOM_DAYS": (5, 20),         # 防御动量窗口(交易日)
    "DEF_MOM_ENTER": (0.001, 0.03),  # 进入阈值 动量>+x%
    "DEF_MOM_EXIT": (-0.06, -0.001), # 退出阈值 动量< -x%
    "HOLD_N": (1, 5),                # 同时持有前 N 名
}


def refresh_active_params(log_change=True):
    """（热更新）从 DB 重读现役参数并覆盖 CFG 可调键。
    只在主循环【决策边界】前调用——新参数对后续 target/止损生效，不强迫已持仓立即换仓；
    但收紧 SR_TRAIL_PCT 会按新阈值从峰值重算跟踪止损，可能触发真实离场（调参的预期效果）。
    返回值：True=发生变更并已应用；False=未变更或 DB 不可用。"""
    global _ACTIVE_PARAM, _ACTIVE_PARAM_ID, _DB_LAST_HASH
    try:
        import tuning_db as _tn
        tu = _tn.get_active_params()
        if tu and tu.get("_fallback", False):
            return False
        if not tu:
            return False
        # R7' 堵漏：值域钳制，1:1 应用。越出安全域的键整键拒绝(不入 CFG、打日志)；
        # 其余键原样应用——方案A已把"逐步逼近"上移到 auto_tuner(落真实 param_set 行分期逼近)，
        # 故此处始终 applied==DB、版号精确，不存在折中孤儿态。DB 并发写时 active_param 是单个完整行，
        # 要么全量生效、要么保持当前，绝无"半个参数松动"。
        rejected = []
        chg = {}
        for _k in _tn.TUNABLE_KEYS:
            if tu.get(_k) is None:
                continue
            _v = tu[_k]
            _b = TUNABLE_BOUNDS.get(_k)
            if _b is not None and not (_b[0] <= _v <= _b[1]):
                rejected.append(_k)
                tu.pop(_k, None)
                continue
            _v = float(round(_v, 6))
            chg[_k] = _v
            tu[_k] = _v
        fp = tuple(sorted(chg.items()))
        if fp == _DB_LAST_HASH:
            return False                       # 无变化，跳过
        _DB_LAST_HASH = fp
        if chg:
            CFG.update(chg)
            if log_change:
                logger.info(f"[DB调参] 已热更应用现役参数: {chg}"
                            + (f"（越界拒绝 {rejected}）" if rejected else ""))
        _ACTIVE_PARAM = {k: v for k, v in tu.items() if k in _tn.TUNABLE_KEYS}
        # 版号：经 1:1 应用后 applied==DB active，二者即同一 param_set 行 → 参数反查精确定位；
        # 极少数查不到行(如本行无 id/手工写脏)时回退 active_param 落库的权威 id。恒不重复、不污染。
        _ACTIVE_PARAM_ID = _tn.get_param_set_id(_ACTIVE_PARAM) or _tn.get_active_param_set_id()
        return True
    except Exception as _e:
        logger.warning(f"[DB调参] 热更读取失败，保持当前参数: {_e}")
        return False


refresh_active_params(log_change=True)  # 启动即应用一次


def notify(title, msg):
    logger.info(f"[通知] {title}：{msg}")
    if notification:
        try:
            notification.notify(title=title, message=msg, timeout=15)
        except Exception:
            pass


def _today():
    return datetime.datetime.now().strftime("%Y-%m-%d")


def _date(s):
    return datetime.datetime.strptime(s, "%Y-%m-%d").date()


def _sleep_until_next_session():
    """[时序优化] 非交易时段精确睡到下一个交易时段起点，避免固定 sleep(300) 造成
    无谓空转/电费浪费与"下午开盘最多延迟5分钟才恢复监控"。

    交易时段：上午 09:15-11:30，下午 13:00-15:00。
    场景：
      - 午休中(11:30-13:00)  → 睡到当日 13:00
      - 收盘后(15:00-24:00) → 睡到下一交易日 09:15（跨周末/节假日，用真实日历）
      - 盘前(00:00-09:15)   → 睡到当日 09:15（若今日非交易日，则睡到下一交易日）
    天然对齐交易日历，且每次醒来必然落在交易时段起点，无需再 sleep(300) 空转。
    -- 注意：这只会"对齐起点"，盘中监控仍由 60s 挡损 + 120s 决策心跳驱动。"""
    import time as _t
    now = datetime.datetime.now()
    hm = now.hour * 100 + now.minute

    # 午休：睡到当日 13:00
    if 1130 < hm < 1300:
        nxt = now.replace(hour=13, minute=0, second=0, microsecond=0)
        _t.sleep(max(0, (nxt - now).total_seconds()))
        return

    # 收盘后或盘前：睡到下一个交易时段起点（09:15）
    # 未到 09:15 先看当日是否为交易日；已到/已过 09:15 则顺延到下一交易日 09:15。
    start = now.replace(hour=9, minute=15, second=0, microsecond=0)
    nxt = start
    if now >= start:
        nxt += datetime.timedelta(days=1)
    # 逐日推进到最近的真实交易日（含调休），最多 7 天防止异常死循环
    for _ in range(7):
        if dc.is_trading_day(nxt):
            break
        nxt += datetime.timedelta(days=1)
    _t.sleep(max(0, (nxt - now).total_seconds()))


def check_kill_switch():
    if os.path.exists(KILL_SWITCH_FILE):
        logger.warning("检测到 KILL_SWITCH，停止交易。")
        return True
    return False


# ----------------------------- 券商连接（实盘，懒加载） -----------------------------
def _ths_pids(exe):
    """返回正在运行的 xiadan.exe 进程 PID 集合（同花顺会拉起多个进程，须按 exe 路径匹配）。"""
    import psutil
    exe_n = os.path.normpath(exe).lower()
    pids = set()
    for p in psutil.process_iter(["pid", "exe"]):
        try:
            if p.info["exe"] and os.path.normpath(p.info["exe"]).lower() == exe_n:
                pids.add(p.info["pid"])
        except Exception:
            continue
    return pids


def _find_ths_main_window(pids):
    """轮询找「网上股票交易系统」主窗口，返回 (backend, pid, window) 或 None。"""
    from pywinauto import Desktop
    for pid in pids:
        for backend in ("win32", "uia"):
            try:
                wins = Desktop(backend=backend).windows()
            except Exception:
                continue
            for w in wins:
                try:
                    if w.process_id() == pid and "网上股票交易系统" in (w.window_text() or ""):
                        return backend, pid, w
                except Exception:
                    continue
    return None


def _is_logged_in_via_tree(pid):
    """只读检测是否已登录：看客户端左侧菜单树有没有内容（不点击、不动鼠标）。
    登录后左侧才有 SysTreeView32+买入[F1]；未登录则没有。"""
    from pywinauto import Desktop
    try:
        for w in Desktop(backend="win32").windows():
            try:
                if w.process_id() != pid:
                    continue
                for c in w.descendants(class_name="SysTreeView32"):
                    try:
                        txts = c.texts()
                    except Exception:
                        continue
                    if any(t and "买入" in t for t in txts):
                        return True
            except Exception:
                continue
    except Exception:
        pass
    return False


def connect_broker():
    """连接同花顺交易客户端（easytrader ths，官方支持；东财不受支持）。
    流程：确保 xiadan.exe 运行 → 轮询主窗口 → 按 PID 精确 attach（避开同花顺多进程错选）。
    客户端需已登录：登录时勾选『记住密码』即可开机自动登录。
    失败返回 None，main_loop 将【拒绝运行】，绝不回退模拟。"""
    import easytrader
    # 静音 easytrader 内部噪音（未登录时"找菜单"失败的堆栈刷屏），只保留我们自己的日志
    logging.getLogger("easytrader").setLevel(logging.CRITICAL)
    logging.getLogger("pywinauto").setLevel(logging.WARNING)
    config = configparser.ConfigParser()
    config.read(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.ini"), encoding="utf-8")
    if "tonghuashun" not in config:
        raise Exception("config.ini 缺少 [tonghuashun] 配置")
    sec = config["tonghuashun"]
    exe = sec.get("exe_path", "")
    if not os.path.exists(exe):
        raise Exception(f"同花顺客户端不存在: {exe}（请安装同花顺独立交易版并修正 exe_path）")
    # 1) 确保客户端在运行
    pids = _ths_pids(exe)
    if not pids:
        subprocess.Popen([exe], cwd=os.path.dirname(exe))
        time.sleep(8)
        pids = _ths_pids(exe)
    if not pids:
        raise Exception("同花顺客户端启动失败")
    # 2) 轮询主窗口（最多90秒，等待自动登录完成）
    main_win = None
    for _ in range(30):
        main_win = _find_ths_main_window(pids)
        if main_win:
            break
        time.sleep(3)
    if main_win is None:
        raise Exception("未找到同花顺交易主窗口（客户端未登录成功？请手动登录一次并勾选'记住密码'）")
    backend, pid, w = main_win
    # 3) 按 PID attach
    from pywinauto import Application
    h = w.handle
    if callable(h):
        h = h()
    user = easytrader.use("ths")
    app = Application(backend=backend).connect(process=pid)
    user._app = app
    user._main = app.window(handle=h)
    # 连接后立刻还原+置前主窗口：easytrader 的 balance/position 用 top_window().type_keys 切菜单，
    # 无人值守启动时若前台/可见性时序不对会抛 ElementNotVisible 直接崩溃。先强制置前兜底。
    try:
        import ctypes
        _u = ctypes.windll.user32
        _u.ShowWindow(h, 9)          # SW_RESTORE：最小化/隐藏都恢复成可见
        _u.SetForegroundWindow(h)
        time.sleep(0.3)
    except Exception:
        pass
    user._init_toolbar()
    # 4) 等待登录（一直等，只读检测不点鼠标）：程序只负责"等你登录→自动接上"
    t0 = time.time()
    notified = False
    while True:
        if _is_logged_in_via_tree(pid):  # 左侧菜单树出现=已登录（不点击、不抢鼠标）
            break
        if not notified:
            notify("请完成同花顺登录", "客户端已启动，请在登录窗口输密码+验证码后点OK")
            logger.warning("等待用户完成同花顺登录（输密码+验证码）...")
            notified = True
        if time.time() - t0 > 300:
            notify("仍在等待登录", "还没检测到登录。登录成功后程序自动继续；想停就关闭本窗口")
            logger.info("仍在等待登录（每5分钟提醒一次；登录后自动继续）")
            t0 = time.time()
        # 客户端进程没了 → 没东西可等，退出
        if not _ths_pids(exe):
            raise Exception("同花顺客户端已关闭，程序退出")
        time.sleep(30)
    # 5) 账户确认 + 模式识别：完全以"客户端账户通道下拉框"为准——客户端记住上次选的是模拟还是实盘，
    #    （模拟盘下拉框文案含"模拟"；实盘显示券商名，无"模拟"字样）。不再用金额/账号等任何配置兜底。
    #    识别原则：下拉框含"模拟"→模拟盘；否则→实盘（宁可当实盘醒目提示，也不误当模拟）。
    try:
        _bal = None
        _err = None
        # —— 启动自愈修复(2026-09)：账户读取失败不再一失败就默默退出。
        # 临时性失败(验证码锁/窗口不可见/客户端短暂异常) → 自动恢复窗口前台重试最多 10 次，
        # 期间每 3 次发一次"仍在自愈等待"提醒，让用户知道进程还活着、在等客户端可用；
        # 只有持续失败才强通知"需人工重启客户端"后退出，杜绝"以为启动成功实际已退"的操作风险。
        _MAX_RETRY = 10
        for _retry in range(_MAX_RETRY):
            try:
                _bal = user.balance
                break
            except Exception as _e:
                _err = _e
                try:
                    import ctypes
                    _u = ctypes.windll.user32
                    _u.ShowWindow(h, 9)
                    _u.SetForegroundWindow(h)
                except Exception:
                    pass
                if (_retry + 1) % 3 == 0:
                    notify("启动自愈中", f"账户读取第{_retry+1}次仍未成功，程序自动重试中（客户端可能触发验证码/反爬锁），请确认同花顺可见可用；不会退出。")
                time.sleep(2.0)
        if _bal is None:
            _msg = (f"启动失败：连续 {_MAX_RETRY} 次读取账户资金仍失败（客户端异常/窗口不可见/验证码反爬锁）。"
                    f"请重启同花顺客户端冷却后再点【启动交易.bat】重试；本次已停止以规避重复下单风险。")
            logger.critical(f"账户读取失败：{_err} {_msg}")
            notify("启动失败-需人工", _msg)   # 强通知，避免"以为启动成功"
            raise RuntimeError("启动失败：账户读取失败（详见通知）")
        _ta = float(_bal.get("总资产") or 0)
        logger.info(f"账户确认：总资产={_ta} 资金余额={_bal.get('资金余额')} 市值={_bal.get('股票市值')}")
        _sim_hit = None
        try:
            for _c in user._main.descendants(class_name="ComboBox"):
                try:
                    _t = (_c.window_text() or "").strip()
                except Exception:
                    continue
                if "模拟" in _t and len(_t) < 50:
                    _sim_hit = _t
                    break
        except Exception:
            pass
        CFG["PAPER"] = bool(_sim_hit)  # 下拉框为券商名(非模拟)→实盘
        _mode = "模拟盘" if CFG["PAPER"] else "实盘"
        logger.info(f"模式（客户端通道下拉框为准）：{_mode}（下拉框显示：{_sim_hit or '券商账户·实盘'}）")
        notify("账户已连接", f"总资产 {_ta} 元，识别模式：{_mode}（请人工核对）")
    except Exception as e:
        logger.critical(f"账户读取失败：{e}")
        notify("账户读取失败", str(e))
        raise
    logger.info(f"券商客户端连接成功（同花顺 pid={pid}）")
    return user


def _click_bbox(e):
    """对 win32 后端控件做真实鼠标点击（比 .click() 更可靠，自绘/自定义按钮也能触发）。
    提取控件中心点，用 SetCursorPos+mouse_event 原生点击。返回 True=已点击。"""
    try:
        r = e.rectangle()
        if r.width() <= 0 or r.height() <= 0:
            return False
        x, y = (r.left + r.right) // 2, (r.top + r.bottom) // 2
        import ctypes
        ctypes.windll.user32.SetCursorPos(x, y)
        time.sleep(0.05)
        ctypes.windll.user32.mouse_event(0x0002, 0, 0, 0, 0)  # LEFTDOWN
        ctypes.windll.user32.mouse_event(0x0004, 0, 0, 0, 0)  # LEFTUP
        return True
    except Exception:
        return False


def _find_captcha_dialog_win32():
    """[2026-09 关键修复] win32gui【原生枚举】「正在拷贝数据/验证码」弹窗。

    为什么不用 pywinauto Desktop.descendants：验证码弹窗是行情子进程(hexin.exe)下的
    自绘窗、标题常为空字符串''，pywinauto 跨进程枚举不到 → 实盘里 `_solve_captcha`
    永远探不到 → “只反复复制、从不填验证码”。本函数用与 _read_pos_full.find_dialog()
    同款的 win32gui.EnumWindows + EnumChildWindows + 按正文(含“拷贝/验证码”) + 按钮判据，
    空标题也能命中（实测解出 0872/7666）。

    返回 (hwnd, title, statics_list, edit_hwnd, btn_list) 或 None。
      statics_list: [(hwnd, text)]（含文字型与空文本图片型）
      edit_hwnd   : 首个 Edit 输入框的句柄（可能 None）
      btn_list    : [(hwnd, raw_text, norm_text)]
    """
    import win32gui
    import re as _re
    dialogs = []

    def _child_cb(ch, ctx):
        cls = win32gui.GetClassName(ch)
        ct = win32gui.GetWindowText(ch) or ""
        if cls == "Static":
            ctx["statics"].append((ch, ct))
        elif cls == "Edit" and ctx["edit"] is None:
            ctx["edit"] = ch
        elif cls == "Button":
            n = ct.replace("&", "")
            n = _re.sub(r"\([A-Za-z]\)\s*$", "", n).strip()
            ctx["btns"].append((ch, ct, n))
        return True

    def _score(ctx):
        """对候选弹窗打分：得分高者更可能直接含可解的验证码（避免附着到"验证码错误"提示框）。"""
        s = 0
        for _ch, _t in ctx["statics"]:
            try:
                if _re.search(r"(?<!\d)\d{4}(?!\d)", _t or ""):
                    s += 100        # 明文 4 位验证码，最稳
                elif not (_t or "").strip():
                    # 空文本 Static=图片型验证码；尺寸有效才有意义（>0 才能截图）
                    _r = win32gui.GetWindowRect(_ch)
                    if (_r[2] - _r[0]) > 0 and (_r[3] - _r[1]) > 0:
                        s += 50
            except Exception:
                pass
        if ctx["edit"] is not None:
            s += 10                 # 有输入框
        return s

    def _enum_cb(hwnd, _):
        try:
            if not win32gui.IsWindowVisible(hwnd):
                return True
            if win32gui.GetClassName(hwnd) not in ("#32770", "Dialog"):
                return True
            title = win32gui.GetWindowText(hwnd) or ""
            ctx = {"statics": [], "edit": None, "btns": []}
            win32gui.EnumChildWindows(hwnd, _child_cb, ctx)
            body = " ".join(t for _, t in ctx["statics"])
            has_btn = any(n in ("确定", "取消", "OK", "确认", "是", "否")
                          for _, _, n in ctx["btns"])
            hit = ((("验证码" in body or "拷贝" in body or "检测到您正在" in body) and has_btn)
                   or "拷贝数据" in title or "验证码" in title or "股票复制" in title
                   or "正在拷贝数据" in title)
            if hit:
                dialogs.append((_score(ctx), hwnd, title, ctx["statics"], ctx["edit"], ctx["btns"]))
        except Exception:
            pass
        return True
    try:
        win32gui.EnumWindows(_enum_cb, None)
    except Exception:
        return None
    if not dialogs:
        return None
    # 同现多弹窗时按"可解性"降序，返回最高分者（最可能含明文/图片型验证码 + 输入框）
    dialogs.sort(key=lambda x: x[0], reverse=True)
    _, hwnd, title, statics, edit, btns = dialogs[0]
    return (hwnd, title, statics, edit, btns)


def _solve_captcha(broker, max_tries=4):
    """检测'正在拷贝数据/请先输入验证码'弹窗 → 截图 + ddddocr 多次识别取共识 → 填入 → 确定。
    ★ 用 win32 后端（同花顺客户端 uia 枚举不到顶层弹窗/控件，win32 才可见）。
    ★ 增强：对截图里"检测到您正在拷贝数据"这类"提示"标题弹窗，先尝试从 Static 明文读验证码，
    （此样式验证码是直接在弹窗把 4 位数字显示为文本摆在输入框旁，而非干扰图像，直接读比 OCR 准），
    读不到再退而走整窗 OCR，解决“弹了码但没识别到”根因。
    识别失败自动点取消并跳过（宁空不乱动）。返回 True=已尝试自动解。"""
    import re
    import ddddocr
    from PIL import ImageGrab
    import PIL.Image as _im, PIL.ImageOps as _op, PIL.ImageFilter as _f
    # 用 win32 后端连接客户端，保证能看到验证码弹窗及其中 Edit/按钮。
    # 注意：验证码“股票复制识别/拷贝数据”弹窗属于同花顺**行情子进程**(hexin.exe)，
    #   与交易主进程(xiadan.exe)父子不同、pid 不同。不能只 connect 到 broker 的进程，
    #   否则根本看不到弹窗 → 读持仓必失败。
    # [2026-09 关键修复] 改用 `_find_captcha_dialog_win32()` win32gui【原生枚举】顶层弹窗
    #   （空标题自绘弹窗 pywinauto Desktop.descendants 枚举不到 → 旧逻辑在这里探不到 →
    #   永远只复制不填验证码）。找到后用 pywinauto 按句柄包装，复用下方成熟填码逻辑。
    try:
        ocr = ddddocr.DdddOcr(show_ad=False)
    except Exception as e:
        logger.warning(f"OCR 不可用：{e}")
        return False
    from pywinauto import Desktop as _Desk
    for _ in range(max_tries):
        _res = _find_captcha_dialog_win32()
        if _res is None:
            return False  # 弹窗已消失（可能已被解掉）
        try:
            win = _Desk(backend="win32").window(handle=_res[0])
        except Exception as e:
            logger.warning(f"弹窗句柄包装失败：{e}")
            return False
        # ★ 增强：先尝试从弹窗静态控件直接读明文 4 位验证码，这比 OCR 准 100%
        #    （你的截图就是这种：弹窗正文显示"检测到...请输入验证码："，旁边 Static 直接放着"0711"明文）
        cand = None
        for c in win.descendants(class_name="Static"):
            try:
                txt = (c.window_text() or "")
                if not txt:
                    continue
                # 找纯 4 位数字（同花顺防爬就是 4 位，少了更宽可能正文里混进去了）
                m = re.search(r"(?<!\d)\d{4}(?!\d)", txt)
                if m:
                    cand = m.group(0)
                    logger.info(f"验证码明文提取：从 Static 控件读到 {cand!r}")
                    break
            except Exception:
                continue
        if not cand:
            # 找不到明文 → 退而走 OCR：先裁剪验证码区域，再三次识别取共识
            try:
                r = win.rectangle()
                if r.width() <= 0 or r.height() <= 0:
                    break
                img = ImageGrab.grab(bbox=(r.left, r.top, r.right, r.bottom))
            except Exception as e:
                logger.warning(f"截图弹窗失败：{e}")
                break
            # ★ 改进(2026-09-02)：验证码是【图片型空文本 Static】。逐个枚举空文本 Static 候选，
            #    对每个候选裁剪→OCR，取第一个能出≥4位连续数字的；保守起见小图先放大2x再卷锐化。
            img_list = []
            try:
                import re as _re
                import win32gui as _wg
                # ★ 改用 win32gui 原生枚举(与 _read_pos_full 同款)：pywinauto 的 descendants 在跨进程
                #   自绘弹窗上枚举不到空文本 Static(图片型验证码) → 整窗兜底 0x0 → 识别失败 → 反复刷屏。
                _win_h = win.handle
                _statics = []
                def _cb(_ch, _):
                    try:
                        if _wg.GetClassName(_ch) == "Static":
                            _statics.append(_ch)
                    except Exception:
                        pass
                    return True
                _wg.EnumChildWindows(_win_h, _cb, None)
                for _ch in _statics:
                    try:
                        if (_wg.GetWindowText(_ch) or "").strip():
                            continue      # 只要空文本(图片型)
                        _rr = _wg.GetWindowRect(_ch)
                        _wd = _rr[2] - _rr[0]
                        _ht = _rr[3] - _rr[1]
                        if _wd <= 0 or _ht <= 0:
                            continue
                        # 过滤宽松全收(对齐 _read_pos_full 实测解出 0872/7666)
                        img_list.append((_rr[1], _wd, _ht, _rr))
                    except Exception:
                        continue
                img_list.sort(key=lambda x: x[0])  # 由上到下
            except Exception:
                img_list = []
            # 图片型验证码可能延迟渲染：第一次枚举为空 → 稍候再枚举一次再落兜底
            if not img_list:
                try:
                    time.sleep(0.15)
                    _statics2 = []
                    def _cb2(_ch2, _):
                        try:
                            if _wg.GetClassName(_ch2) == "Static":
                                _statics2.append(_ch2)
                        except Exception:
                            pass
                        return True
                    _wg.EnumChildWindows(_win_h, _cb2, None)
                    for _ch in _statics2:
                        if (_wg.GetWindowText(_ch) or "").strip():
                            continue
                        _rr2 = _wg.GetWindowRect(_ch)
                        if (_rr2[2] - _rr2[0]) > 0 and (_rr2[3] - _rr2[1]) > 0:
                            img_list.append((_rr2[1], _rr2[2]-_rr2[0], _rr2[3]-_rr2[1], _rr2))
                    img_list.sort(key=lambda x: x[0])
                except Exception:
                    pass
            # 原整窗截图兜底
            if not img_list:
                img_list = [(None, 0, 0, None)]
            # 逐个候选 OCR 取共识，取到 4 位即停
            cand = None
            for _top, _w, _h, _cr in img_list:
                if _cr is None:
                    _rr = win.rectangle()
                    img = ImageGrab.grab(bbox=(_rr.left, _rr.top, _rr.right, _rr.bottom))
                else:
                    img = ImageGrab.grab(bbox=(_cr[0], _cr[1], _cr[2], _cr[3]))
                # 对齐 _read_pos_full ocr_code：灰度+自动对比度+小图放大3x+锐化，提升识别率
                try:
                    img = img.convert("L")
                    img = _op.autocontrast(img)
                    if img.width < 200:
                        img = img.resize((img.width * 3, img.height * 3), _im.LANCZOS)
                    img = img.filter(_f.SHARPEN)
                except Exception:
                    pass
                texts = []
                for _ in range(3):
                    try:
                        texts.append(ocr.classification(img))
                    except Exception:
                        pass
                if not texts:
                    continue
                s = max(set(texts), key=texts.count)
                _dig = "".join(_re.findall(r"\d", s or ""))
                if len(_dig) >= 4:
                    cand = _dig[:4]
                    logger.info(f"验证码OCR结果：{texts}（{_w}x{_h}）→ {cand!r}")
                    break
                logger.info(f"验证码OCR候选({_w}x{_h})：{texts} 非4位，跳过")
            if cand is None:
                continue  # 没读到码 → 点取消，下轮再试
        # 找 Edit 输入框填入
        edit = None
        try:
            for c in win.descendants():
                try:
                    if str(c.class_name()).lower() == "edit" and c.is_enabled():
                        # 不再按 is_visible 严格卡——有的弹窗输入框可见但返回 False
                        edit = c
                        break
                except Exception:
                    continue
        except Exception:
            edit = None
        solved = False
        if edit:
            try:
                # ★ 填入必须用【键盘输入】，不能用 WM_SETTEXT（自绘输入框不认 WM_SETTEXT，
                #   读回为空 → 提交空值 → 恒判"验证码错误"）。2026-09-02 实测定：先物理点击
                #   输入框中心聚焦（set_focus 在跨进程/自绘框上可能失效），再 type_keys 敲入。
                try:
                    win.set_focus()
                except Exception:
                    pass
                time.sleep(0.1)
                _click_bbox(edit)            # 物理点击聚焦验证码输入框
                time.sleep(0.2)
                try:
                    edit.set_focus()          # 双保险
                except Exception:
                    pass
                edit.type_keys("^a", with_spaces=False)
                time.sleep(0.05)
                edit.type_keys("{DEL}", with_spaces=False)  # 清空
                time.sleep(0.1)
                edit.type_keys(cand, with_spaces=False)
                time.sleep(0.3)
                # 点确定（从底往顶搜，"确定"按钮通常靠下）
                for b in reversed(list(win.descendants(class_name="Button"))):
                    try:
                        bt = (b.window_text() or "").strip()
                        if bt in ("确定", "OK", "确认") and b.is_enabled() and _click_bbox(b):
                            logger.info(f"已填入验证码 {cand!r} 并点确定（若错误，弹窗会刷新重试）")
                            solved = True
                            break
                    except Exception:
                        continue
            except Exception as e:
                logger.warning(f"填入验证码失败：{e}")
        if not solved:
            # 找不到 Edit/确定 → 退而直接键盘输入（自绘弹窗/找不到控件兜底）
            try:
                win.set_focus()
                time.sleep(0.2)
                from pywinauto import keyboard as _kb2
                _kb2.send_keys(cand, with_spaces=False)
                time.sleep(0.3)
                _kb2.send_keys("{ENTER}")
                time.sleep(0.5)
                logger.info(f"自绘/无控件：已键盘输入 {cand!r} 并回车提交")
                solved = True
            except Exception as e:
                logger.warning(f"自绘验证码键盘输入失败：{e}")
        if solved:
            time.sleep(0.5)
            return True
        # 解不掉 → 点取消/ESC 关掉弹窗，不挡路不饿死下一轮重试
        dismissed = False
        try:
            for b in win.descendants():
                try:
                    if (b.window_text() or "").strip() == "取消" and b.is_enabled() and _click_bbox(b):
                        dismissed = True
                        break
                except Exception:
                    continue
        except Exception:
            pass
        if not dismissed:
            try:
                win.set_focus()
                from pywinauto import keyboard as _kb3
                _kb3.send_keys("{ESC}")
                dismissed = True
            except Exception:
                pass
        if dismissed:
            logger.warning("验证码未能自动识别，已点取消/ESC（下一轮再尝试）")
            return False
        time.sleep(1)
    return False


def _captcha_blocked(broker, notify_throttle=1800):
    """检测同花顺"正在拷贝数据"防爬弹窗：先尝试 OCR 自动解（填验证码+确定），解不掉再点取消并跳过本轮。
    返回 True=本轮跳过（验证码仍在，避免反复重读持续触发）；False=可正常读持仓/交易。
    [2026-09 关键修复] 探测改用 _find_captcha_dialog_win32() 原生枚举（旧 uia/descendants 探不到）。
    [2026-09 优化] 连续多次解不出=疑似账号临时反爬锁 → 升温冷却并提示重启客户端冷却，
    避免在 0x0/难以识别时硬啃同一把锁、反复触发更多反爬（实测曾连续约50分钟失败）。"""
    try:
        if _find_captcha_dialog_win32() is None:
            _captcha_blocked._fails = 0
            return False
        fails = getattr(_captcha_blocked, "_fails", 0)
        now = time.time()
        # —— 反爬锁冷却：连续失败达到阈值 → 不再反复尝试/反复触发 ——
        if fails >= 3:
            _last = getattr(_captcha_blocked, "_cool_last", 0)
            if now - _last > 1800:
                _captcha_blocked._cool_last = now
                notify("验证码反爬锁", "连续多次自动识别失败，疑似账号被临时反爬锁。建议关闭并重启同花顺客户端冷却 5-10 分钟后继续；程序会先自动跳过本轮。")
                logger.warning("验证码连续失败(疑似反爬锁)，进入冷却跳过；建议重启同花顺客户端冷却")
            return True
        _solve_captcha(broker)          # 优先原生枚举+OCR 自动解
        time.sleep(1.0)                 # 等弹窗刷新/消失
        if _find_captcha_dialog_win32() is None:
            _captcha_blocked._fails = 0
            return False                # 已解掉 → 本轮可正常读持仓/交易
        _captcha_blocked._fails = fails + 1   # 累计连续失败
        if not hasattr(_captcha_blocked, "_last") or now - _captcha_blocked._last > notify_throttle:
            _captcha_blocked._last = now
            notify("验证码处理", "检测到'正在拷贝数据'弹窗，自动识别未解掉，请手动看图填一下")
            logger.warning("检测到'正在拷贝数据'防爬弹窗 → 自动解未成功，本轮跳过")
        return True                     # 验证码仍在 → 本轮跳过（避免刚处理完又立刻重读触发）
    except Exception:
        return False


def _find_field_after_label(broker, keywords, exclude=None):
    """按静态标签文本，找到其右侧同一行的可输入 Edit 框（比'前3个Edit'可靠，不受窗口内其它编辑框干扰）。
    keywords: 标签命中关键词列表（任一命中即可）；exclude: 需要排除的 Edit 对象集合（防同框被多标签复选）。"""
    main = broker._main
    # 当前可见的 Edit，全部参与匹配；exclude 用于避免同一个框被多个字段抢到
    edits = [e for e in main.descendants(class_name="Edit") if e.is_visible()
             and (exclude is None or e not in exclude)]
    if not edits:
        return None
    # 标签：在同窗口里找 Static 文本命中关键词的，取它正右方、水平居中的 Edit
    cand = []
    for t in main.descendants(class_name="Static"):
        try:
            txt = (t.window_text() or "").strip()
        except Exception:
            continue
        if not txt:
            continue
        if not any(k in txt for k in keywords):
            continue
        try:
            tr = t.rectangle()
        except Exception:
            continue
        for e in edits:
            try:
                er = e.rectangle()
            except Exception:
                continue
            # Edit 需在标签右侧、且纵向与标签同高度带（自动吸附）
            if er.left < tr.right:
                continue
            if er.top > tr.bottom + 15 or er.bottom < tr.top - 15:
                continue
            # 存 (竖向距离, 横向距离, Edit) 三要素，排序仅按坐标，避免比较 Edit 对象本身
            cand.append((abs(er.top - tr.top), abs(er.left - tr.left), e))
    if not cand:
        return None
    cand.sort(key=lambda t: (t[0], t[1]))
    return cand[0][2]


def _fill_edit(broker, e, value):
    """清空后填值 + 回读确认，值不符视为失败（防脏单/错位）。
    注意：这类自定义 Edit 是"追加式"输入，Ctrl+A 只清当前槽位、回读值在 texts()[1]，
    需 Ctrl+End 到行尾后连续 Backspace 才能彻底清空（否则新旧值拼接成脏单）。"""
    try:
        e.click_input(); time.sleep(0.2)
        # 光标到文末，再连续 Backspace 清到底（比依赖 ^a 可靠，避免多槽位残留拼接）
        e.type_keys("{END}"); time.sleep(0.1)
        for _ in range(24):  # 最多清 24 位（覆盖 140014014400 这类脏值长度）
            e.type_keys("{BACKSPACE}", with_spaces=False)
            time.sleep(0.02)
        # 再点一次聚焦并补一次 Ctrl+A+DEL（兜底）
        e.type_keys("^a", with_spaces=False)
        e.type_keys("{DELETE}")
        time.sleep(0.2)
        e.type_keys(str(value), with_spaces=False)
        time.sleep(0.3)
        back = self_read_edit(e)
        return back, back == str(value)
    except Exception as ex:
        return None, False


def self_read_edit(e):
    """读 Edit 实际值：优先取 texts() 的最后一个非空槽（这些控件 [0] 是标签位、[1] 是真实值）。"""
    try:
        ts = e.texts() or ['']
        for t in reversed(ts):
            if t.strip():
                return t.strip()
        return ''
    except Exception:
        return ''


PY32 = r"E:\Python\envs\py32\python.exe"
EXEC32 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exec32_trade.py")


def _sec_symbol(code):
    """6位代码 → exec32 需要的交易所前缀。ETF: 5xxxxx=sh, 1xxxxx=sz；股票 6xx=sh,0xx=sz。"""
    c = str(code)
    if c[:2] in ("60", "68", "51", "52", "58") or c[0:1] in ("5", "6", "9"):
        return "sh" + c
    return "sz" + c


def _shortcut_trade(broker, code, side, price, qty):
    """下单：把委托交给 32 位执行器 exec32_trade.py（control_id 填表 + 物理点击确认），返回 True。

    架构（project_memory 约定）：64 位 auto_run 只做策略决策，32 位 Python 驱动 32 位
    xiadan.exe 执行。此前本函数用"标签锚定 Edit + 鼠标点击"在 64 位进程内驱动，实测
    SysTreeView32/Edit 枚举不稳定导致反复'未能锚定输入框'。改为 subprocess 调 exec32_trade.py，
    其按稳定 control_id(1032代码/1033价格/1034数量) 填表、物理点击 id=1006 按钮、
    物理点击'是(Y)'确认，并自带重试（3 次）与价格复核，成熟可靠。"""
    import subprocess
    px = round(float(price), 3)
    sym = _sec_symbol(code)
    if not os.path.exists(PY32):
        raise RuntimeError(f"32位执行器缺失: {PY32}")
    cmd = [PY32, EXEC32, side, sym, f"{px:f}", str(qty)]
    logger.info(f"[下单] 调32位执行器: {' '.join(cmd)}")
    # 关键：exec32 依赖 pywin32_system32 里的 pythoncom38/pywintypes38 DLL，
    # 必须把该目录注入子进程 PATH（否则 "import win32api -> DLL load failed")。
    _sys32 = os.path.join(os.path.dirname(PY32), "Lib", "site-packages", "pywin32_system32")
    _env = dict(os.environ)
    _env["PATH"] = _sys32 + os.pathsep + _env.get("PATH", "")
    _env["PYTHONIOENCODING"] = "utf-8"  # 强制 exec32 输出 UTF-8，与下方 encoding 保持一致
    proc = subprocess.run(
        cmd, capture_output=True, text=True, cwd=os.path.dirname(EXEC32),
        timeout=120, env=_env, encoding="utf-8", errors="replace",
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    # exec32 成功会打印 "TRADE_DONE"；失败打印 FAIL/ERROR
    ok = "TRADE_DONE" in (proc.stdout or "")
    fail_hit = any(k in out for k in ("ERROR", "FAIL", "Traceback", "未找到"))
    if ok:
        logger.info(f"[下单成功] {side} {sym} {qty}股 @ {px}\n{out}")
        return True
    if proc.returncode not in (0, None) or fail_hit or not ok:
        logger.error(f"[下单失败] {side} {sym} @ {px} returncode={proc.returncode}\n{out[-1800:]}")
        raise RuntimeError(f"32位执行器下单失败({side} {sym}): {out.strip()[-400:]}")
    return False

def _dismiss_pending_prompts(broker, timeout=2.0, expect_qty=None):
    """扫客户端顶层弹窗并按内容选按钮。'不是同花顺注册用户'点立即注册；其它点确定/是。

    expect_qty：期望委托数量（股）。若为委托确认弹窗，先在弹窗全文里读取'数量'字段，
    只有与期望一致（数量误差 <=2%）才点'是(Y)'，否则点'否'取消——这是防脏单的最后一道闸。
    """
    t0 = time.time()
    register_clicked = False
    while time.time() - t0 < timeout:
        clicked = False
        try:
            for w in Desktop(backend="uia").windows():
                if w.process_id() != broker._app.process_id:
                    continue
                wt = (w.window_text() or "")
                want_register = "不是同花顺注册用户" in wt
                # 读该窗口全部文本（拼起来供数量/金额复核）
                full_text = ""
                try:
                    full_text = " ".join([x for x in w.window_texts() or []])
                    for d in w.descendants():
                        try:
                            dt = (d.window_text() or "").strip()
                            if dt and len(dt) < 120:
                                full_text += " " + dt
                        except Exception:
                            continue
                except Exception:
                    pass
                confirm_ok = None  # None=未决定；True=通过可点是；False=不通过应点否
                if expect_qty is not None and ("委托确认" in wt or "确认" in wt):
                    import re as _re
                    m = _re.search(r"数量[:：]?\s*([0-9,，]+)", full_text)
                    qty_in_dialog = int(m.group(1).replace(",", "").replace("，", "")) if m else None
                    if qty_in_dialog is None:
                        confirm_ok = False  # 读不到数量 → 保守不点是
                        logger.warning(f"确认弹窗读不到委托数量，保守点'否'避免脏单（全文: {full_text[:120]}）")
                    else:
                        diff = abs(qty_in_dialog - expect_qty) / expect_qty if expect_qty else 1.0
                        if diff <= 0.02:
                            confirm_ok = True
                        else:
                            confirm_ok = False
                            logger.warning(
                                f"确认弹窗数量 {qty_in_dialog} != 期望 {expect_qty}（偏差 {diff:.2%}），点'否'取消以防脏单")
                for b in w.descendants():
                    try:
                        bt = (b.window_text() or "").strip()
                    except Exception:
                        continue
                    if want_register and bt == "立即注册" and b.is_enabled():
                        try:
                            _click_bbox(b)
                            register_clicked = True
                            notify("需完成同花顺注册", "程序已点『立即注册』，请在浏览器/弹窗里完成手机注册，完成后程序自动继续")
                            logger.warning("撞到'不是同花顺注册用户'，已点立即注册（等用户完成手机注册）")
                        except Exception:
                            pass
                    elif confirm_ok is False and bt in ("否(N)", "否", "取消") and b.is_enabled():
                        try:
                            _click_bbox(b); clicked = True
                        except Exception:
                            pass
                    elif confirm_ok is not False and bt in ("是(Y)", "确定", "是", "Yes", "OK") and b.is_enabled():
                        try:
                            _click_bbox(b); clicked = True
                        except Exception:
                            pass
        except Exception:
            pass
        if not clicked:
            break
        time.sleep(0.2)
    return register_clicked


def _clear_panel_residual(broker):
    """买入/卖出前清掉买卖面板残留（防 21:25 那种 510300 残留导致下错单）。"""
    try:
        for e in broker._main.descendants(class_name="Edit"):
            try:
                e.set_focus()
                e.type_keys("^a", with_spaces=False)
                e.type_keys("{DELETE}")
            except Exception:
                continue
    except Exception:
        pass


# ----------------------------- 持仓管理（读客户端对账；本地自有持仓记录供日常决策） -----------------------------
def _load_owned():
    """我们自己的ETF持仓记录（本地，跨重启）。日常决策用它——不读客户端持仓=不触发'正在拷贝数据'验证码。"""
    if os.path.exists(OWNED_POS_FILE):
        try:
            with open(OWNED_POS_FILE, "r", encoding="utf-8") as fp:
                return json.load(fp)
        except Exception:
            pass
    return {}


def _save_owned(pos):
    try:
        with open(OWNED_POS_FILE, "w", encoding="utf-8") as fp:
            json.dump(pos, fp, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"保存自有持仓失败：{e}")


def _avail_qty(row):
    """从 easytrader 持仓行里取「可用/可卖数量」。兼容客户端表头差异：
    easytrader(ths, Copy策略)实际返回 `可用余额`/`股票余额`，但部分客户端/版本用 `可用数量`——统一优先取存在的键，
    避免因字段名不匹配误把持仓读成 0/空。"""
    for k in ("可用余额", "可用数量"):
        if k in row:
            try:
                return int(row.get(k, 0) or 0)
            except Exception:
                return 0
    return 0


def _sync_owned_from_client(broker):
    """14:50 例行核对：读一次客户端持仓，把我们的记录与真实对账（低频，一天一次；读持仓易触发"正在拷贝数据"验证码）。
    [2026-09 关键修复] 读持仓(复制表格)是**间歇性**的：验证码没解掉/表格未就绪时会返回**空**。
    绝不能把"空读"当"空仓"去覆盖已有记录（否则策略误判空仓、该卖不卖/该换不换）。
    处理：① 空结果自动重试(每次顺带解验证码)；② 重试后仍空 + 本地有昨日持仓 → **保留不清空**并告警。"""
    if broker is None:
        return
    uni = set(CFG["ETF_UNIVERSE"].keys())
    prev = _load_owned()               # 变更前的本地记录（用于防误清空）
    last_err = None
    owned = None
    for _try in range(3):
        try:
            pos = broker.position or []
        except Exception as e:
            last_err = e
            pos = None
        else:
            had = {}
            for h in pos:
                c = h.get("证券代码")
                if c in uni and _avail_qty(h) > 0:
                    had[c] = {"qty": _avail_qty(h), "cost": float(h.get("成本价", 0) or 0), "date": _today(),
                              **({"buy_date": prev.get(c, {}).get("buy_date")} if prev.get(c, {}).get("buy_date") else {})}
            if pos:                    # 读到非空列表＝本次复制成功（哪怕过滤后无ETF也是真读）
                owned = had
                last_err = None
                break
            last_err = Exception(f"读持仓返回空(第{_try+1}次，可能验证码未解/表未就绪)")
        time.sleep(1.2)                # 给验证码刷新/表格渲染留时间再重试
    if owned is None and prev:         # 连续空读 + 本地有持仓 → 保留，禁清空
        logger.warning(f"客户端对账读持仓连续失败(空/异常：{last_err})，且本地有持仓 {sorted(prev)}，保留本地记录，避免误判空仓")
        return
    owned = owned or {}
    _save_owned(owned)
    logger.info(f"已与客户端对账，自有ETF持仓：{owned}")
    if last_err is not None and not owned:
        logger.warning(f"客户端对账读持仓{last_err}，确认无持仓需手工核对")


def current_holdings(broker):
    """当前持仓 = 我们自己的ETF记录（本地）。非ETF持仓（002716等）用户自管，策略不碰。"""
    return set(_load_owned().keys())


def _tail_pursuit(broker):
    """尾盘(14:57)未成交追单兜底：读一次客户端**真实持仓**对账，若与当日目标仍有偏离
    （说明有委托挂在委托栏没成交），则重新跑 once 调仓以追平，收盘前把仓位拉回目标。
    与 14:50 对账(_sync_owned_from_client)互补——对账只更新记录，这里直接追单。
    触发验证码/读持仓失败时安全跳过（沿用本地，交给次日）。
    """
    try:
        target = decide_target()                      # 重新取当日目标（幂等，与主循环同源）
        _sync_owned_from_client(broker)               # 用真实持仓覆盖本地记录
        target_set = set(target) if target != "cash" else set()
        cur = current_holdings(broker)
        if target_set == cur:
            logger.info(f"[尾盘追单] 持仓已==目标 {sorted(target_set)}，无需追单")
            return
        logger.warning(f"[尾盘追单] 持仓 {sorted(cur)} ≠ 目标 {sorted(target_set)}，尝试收盘前追平")
        rebalance_to(target, broker)
        notify("尾盘追单", f"持仓≠目标，已收盘前重试调仓：目标={sorted(target_set)}")
    except Exception as e:
        logger.warning(f"[尾盘追单] 失败(触发验证码/读持仓不可用)，沿用本地记录：{e}")
        # 超时预警：拖过收盘仍未能追平 → 主动通知，避免静默遗漏（次日对账/开盘再平衡兜底）
        try:
            _hn = datetime.datetime.now()
            if _hn.hour >= 15:
                notify("尾盘追单-已收盘", f"已过收盘仍未能追平（{e}）。当天委托若未成交，"
                                           f"留待次日14:50对账+开盘调仓处理；不会错误成交。")
        except Exception:
            pass


# ----------------------------- 轮动决策（基于 data_center，多源兜底） -----------------------------
_PRICE_CACHE = {}       # {code: 上次成功拉取的收盘价Series}；取数失败时代用(带过期保护)
_PRICE_CACHE_DATE = {}  # {code: 缓存数据最后交易日 'YYYY-MM-DD'}，用于过期天数保护


def _trading_days_since(date_str):
    """缓存数据日期 date_str 到「今天」之间的交易日数；无效数据返回 -1。
    用真实交易日历(dc.is_trading_day 含调休)，跨周末/节假日不计过期天数。"""
    try:
        start = _date(date_str)
    except Exception:
        return -1
    today = datetime.date.today()
    if start >= today:
        return 0
    n, cur = 0, start
    while cur < today:
        cur += datetime.timedelta(days=1)
        try:
            n += 1 if dc.is_trading_day(cur) else 0
        except Exception:
            n += 1   # 交易日历不可用时按工作日近似，宁严勿松
    return n


def _cache_reusable(code):
    """缓存过期保护：仅当缓存期龄 ≤ MAX_STALE_PRICE_DAYS(交易日) 才复用缓存价做决策，
    避免连续取数失败时用过期价决策（错过换仓/风险-off避险）。过期则视作无数据。"""
    if code not in _PRICE_CACHE:
        return False
    maxd = int(CFG.get("MAX_STALE_PRICE_DAYS", 2))
    stale = _trading_days_since(_PRICE_CACHE_DATE.get(code, ""))
    return 0 <= stale <= maxd


def fetch_etf_closes():
    """拉取全部 ETF 最近收盘价序列，返回 {code: Series}。"""
    # 取数长度必须覆盖最长均线（MA_GLOBAL=200），否则决策会判「数据不足」。
    need = max(CFG["SLOPE_WINDOW"], CFG["MA_FILTER"], CFG["MA_GLOBAL"]) + 30
    closes = {}
    for c in CFG["ETF_UNIVERSE"]:
        try:
            h = dc.get_etf_k_history(c, days=need)
            if h is not None and not h.empty:
                s = h.set_index("date")["close"]
                closes[c] = s
                _PRICE_CACHE[c] = s
                _PRICE_CACHE_DATE[c] = pd.to_datetime(s.index[-1]).strftime("%Y-%m-%d")
            elif _cache_reusable(c):
                closes[c] = _PRICE_CACHE[c]
                logger.warning(f"ETF {c} 本次取数为空，复用缓存价(期龄{_trading_days_since(_PRICE_CACHE_DATE.get(c,''))}d)")
        except Exception as e:
            logger.error(f"ETF {c} 行情获取失败：{e}")
            if _cache_reusable(c):
                closes[c] = _PRICE_CACHE[c]
                logger.warning(f"ETF {c} 复用缓存价(期龄{_trading_days_since(_PRICE_CACHE_DATE.get(c,''))}d)")
    return closes


def decide_target():
    """返回轮动目标：ETF 代码列表 或 'cash'。单一逻辑源 = strategy_rotator.decide()。
    决策层序：① 全局风险-off(美股熊/A股熊+黄金确认)→防御黄金ETF；② 否则升级轮动 Top HOLD_N。
    """
    codes = list(CFG["ETF_UNIVERSE"].keys())
    # [2026-09 B项·拆盘中空转] 日线决策结果在当日盘中恒定（输入=昨日/收盘K线，盘中不变）。
    # 当日首次计算后缓存，盘中后续调用直接复用，避免每120s重复拉K线+重复决策空转。
    # 因结果恒同，本次去重不改变任何策略行为；次日(日期变化)自动重算。
    _today_key = datetime.date.today().isoformat()
    if STATE.get("_target_cached_date") == _today_key:
        # 复用当日缓存目标，并按缓存结果恢复派生状态（risk_off/_def_held/scores）
        STATE["risk_off"] = bool(STATE.get("_cached_risk_off", False))
        STATE["_def_held"] = bool(STATE.get("_cached_def_held", False))
        STATE["scores"] = STATE.get("_cached_scores", {}) or {}
        rot_target = STATE.get("_target_cached", "cash")
        return rot_target
    prices = fetch_etf_closes()
    broad = CFG["BROAD"]
    if broad not in prices or len(prices[broad]) < max(CFG["MA_FILTER"], CFG["MA_GLOBAL"]):
        logger.warning("宽基数据不足，暂定空仓")
        return "cash"
    params = {
        "HOLD_N": CFG["HOLD_N"], "SLOPE_WINDOW": CFG["SLOPE_WINDOW"],
        "MA_FILTER": CFG["MA_FILTER"] if CFG["MA_FILTER"] else None,
        "SCORE": CFG["SCORE"], "DEFENSIVE": CFG["DEFENSIVE"],
        "DEF_MOM_DAYS": CFG.get("DEF_MOM_DAYS") or None,
        "DEF_MOM_ENTER": CFG.get("DEF_MOM_ENTER", 0.0),
        "DEF_MOM_EXIT": CFG.get("DEF_MOM_EXIT", 0.0),
        "GLOBAL_US": CFG["GLOBAL_US"], "GLOBAL_GOLD": CFG["GLOBAL_GOLD"],
        "BROAD": CFG["BROAD"], "MA_GLOBAL": CFG["MA_GLOBAL"],
        "MA_GOLD": CFG["MA_GOLD"], "GOLD_LOOK": CFG["GOLD_LOOK"],
        "MACRO_OFF": CFG["MACRO_OFF"],
        # 显式钉死防御档=回测最优（backtest_defense_assets.py）：全仓黄金(518880)、退抖关闭。
        # 防 strategy_rotator 默认值将来被改动而悄悄改变实盘行为。
        "DEFENSE_MODE": "defensive",
        "RISK_CONFIRM_DAYS": 0,
        "RISK_EXIT_DAYS": 0,
    }
    # 迟滞状态：上一决策是否已持有防御资产（供 DEF_MOM_ENTER/EXIT 决定进/退阈值）。
    prev_def_held = STATE.get("_def_held", False)
    res = rot.decide(prices, params, universe=codes, prev_def_held=prev_def_held)
    # 记录当前是否持有防御资产，供下一决策日判定迟滞退出阈值。
    t = res["target"]
    STATE["_def_held"] = bool((t != "cash") and (CFG["DEFENSIVE"] in t))
    STATE["risk_off"] = bool(res.get("risk_off", False))
    STATE["last_reason"] = res.get("reason", "")
    # 记录各标的动量分（供 rebalance_to 做购买比例 mom2 加权；风险-off 时为空）
    STATE["scores"] = res.get("scores", {}) or {}
    tag = "【全局风险-off·防御】" if res.get("risk_off") else ""
    logger.info(f"{tag}决策：{res['target']} | {res['reason']}")
    # 缓存当日目标及其派生状态（B项去重；次日自动重算）
    STATE["_target_cached"] = res["target"]
    STATE["_target_cached_date"] = datetime.date.today().isoformat()
    STATE["_cached_risk_off"] = bool(res.get("risk_off", False))
    STATE["_cached_def_held"] = bool((res["target"] != "cash") and (CFG["DEFENSIVE"] in res["target"]))
    STATE["_cached_scores"] = res.get("scores", {}) or {}
    return res["target"]


# ----------------------------- 执行：再平衡 / 止损 -----------------------------
# 交易流水(结构化，JSON行追加)：每次真实下单成功写一条，集中保存 成交时间/方向/代码/
# 价格/数量/金额/触发原因，便于复盘与核证(与 auto_trade.log 的文本日志互补)。
JOURNAL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "trade_journal.jsonl")


def journal(side, code, price, qty, reason=""):
    """追加一条交易流水（成交后调用）。side: buy/sell/stop_sell。失败只告警不影响交易。
    同步双写：本地 JSONL（历史格式）+ DB trade_log（带 param_set_id=当时生效参数，供自动调参）。"""
    try:
        amt = round(float(price or 0) * int(qty or 0), 2)
        rec = {
            "ts": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "side": side, "code": code,
            "price": round(float(price or 0), 3),
            "qty": int(qty or 0), "amount": amt, "reason": reason,
        }
        if _ACTIVE_PARAM:
            rec["param"] = {k: v for k, v in _ACTIVE_PARAM.items()
                            if k in ("SLOPE_WINDOW", "SR_TRAIL_PCT", "DEF_PEAK_STOP",
                                     "MIN_HOLD_DAYS", "DEF_MOM_DAYS", "DEF_MOM_ENTER",
                                     "DEF_MOM_EXIT", "HOLD_N")}
        os.makedirs(os.path.dirname(JOURNAL_FILE), exist_ok=True)
        with open(JOURNAL_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        # 双写到 DB（失败静默，不影响交易）
        try:
            import tuning_db as _tuning
            _tuning.insert_trade(rec, globals().get("_ACTIVE_PARAM_ID"))
        except Exception:
            pass
    except Exception as e:
        logger.warning(f"[流水] 写交易记录失败：{e}")


# ----------------------------- 止损后「再入冷却」（B项） -----------------------------
# 某标的触发止损卖出后，N 个交易日内禁止再买回，防"卖了当天又接回"抖振。
# 冷却按实测交易日历(load_trade_calendar)精确计数，跨重启持久化到本地文件。
REENTER_COOLDOWN_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "reentry_cooldown.json")


def _load_cooldown():
    """{code: 'YYYY-MM-DD' 上次止损卖出日}，跨重启。"""
    if os.path.exists(REENTER_COOLDOWN_FILE):
        try:
            with open(REENTER_COOLDOWN_FILE, "r", encoding="utf-8") as fp:
                return json.load(fp)
        except Exception:
            pass
    return {}


def _save_cooldown(cd):
    try:
        os.makedirs(os.path.dirname(REENTER_COOLDOWN_FILE), exist_ok=True)
        with open(REENTER_COOLDOWN_FILE, "w", encoding="utf-8") as fp:
            json.dump(cd, fp, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"[冷却] 保存失败：{e}")


def _mark_stop_sell(code):
    """记录 code 今天触发止损卖出（下一次买入时被冷却挡住）。"""
    n = CFG.get("REENTER_COOLDOWN_DAYS", 0) or 0
    if n <= 0:
        return
    cd = _load_cooldown()
    cd[code] = _today()
    _save_cooldown(cd)
    logger.info(f"[冷却] {code} 今日止损卖出，{n}个交易日内禁止再买回")


def _trading_days_after(date_str):
    """返回 date_str 当天之后、今天之前(含今天)的交易日天数。date_str in 'YYYY-MM-DD'。"""
    cal = dc.load_trade_calendar()
    if not cal:
        return 0   # 无日历→保守按 0（不冷却，避免误伤买入）
    today = datetime.date.today().strftime("%Y%m%d")
    target = datetime.datetime.strptime(date_str, "%Y-%m-%d").date().strftime("%Y%m%d")
    if target >= today:
        return 0
    count = 0
    # 从卖出日后一天起，数到今天(含)之间的交易日个数
    d = datetime.datetime.strptime(date_str, "%Y-%m-%d").date() + datetime.timedelta(days=1)
    while d.strftime("%Y%m%d") <= today:
        if d.strftime("%Y%m%d") in cal:
            count += 1
        d += datetime.timedelta(days=1)
    return count


def _in_reenter_cooldown(code):
    """code 是否处于止损后再入冷却中。True=本次禁止买回。"""
    n = CFG.get("REENTER_COOLDOWN_DAYS", 0) or 0
    if n <= 0:
        return False
    cd = _load_cooldown()
    last = cd.get(code)
    if not last:
        return False
    elapsed = _trading_days_after(last)
    # 交易日差>=N 且 last记录的卖出日已过去 N 个交易日后才放行
    cool = elapsed < n
    # 冷却已到期：清理记录，避免文件增长
    if not cool:
        cd.pop(code, None)
        _save_cooldown(cd)
    return cool


def _filter_cooldown_targets(target):
    """从买入目标中剔除当前处于止损后再入冷却的标的。"""
    if not target or target == "cash":
        return target
    kept = [c for c in target if not _in_reenter_cooldown(c)]
    removed = [c for c in target if c not in kept]
    if removed:
        logger.info(f"[冷却] 目标 {sorted(removed)} 仍处止损后再入冷却期，本轮跳过不买回")
    return kept if kept else "cash"


def _etf_price(code):
    try:
        h = dc.get_etf_k_history(code, days=5)
        if h is not None and not h.empty:
            return float(h.set_index("date")["close"].iloc[-1])
    except Exception:
        pass
    return None


# 实时价多源节流：记录每个 代码@side 最近一次请求时刻，避免同一周期重复打源。
# 作用 = 天然限流：即使挡损巡检被意外加速，也只会按 _L1_MIN_INTERVAL 实测刷新。
_L1_LAST = {}            # (code, side) -> 最近请求 time.time()
_L1_CACHE = {}           # (code, side) -> 最近一次成功拉到价的实时价
_L1_MIN_INTERVAL = 10.0  # 秒：同一标的同一方向，至少间隔 10s 才重新拉实时价


def _l1_price(code, side):
    """实时对手价（保证成交优先）：买入用卖一、卖出用买一。
    [2026-09 多源护盾] sina→腾讯→东财 顺序轮询，逐源降级；源全失败回落最新收盘价。
    带节流：同(码,方向) 10s 内复用上一次实时价，防重复打源被限流。
    （回测 backtest_sweep_freq：巡检10~120s对收益零影响，故节流不会伤策略。）"""
    import time as _t
    key = (code, side)
    now = _t.time()
    # 节流：同一(码,方向) 10s 内不重复请求实时源，返回上次缓存价
    if _L1_LAST.get(key):
        if now - _L1_LAST[key] < _L1_MIN_INTERVAL:
            last = _L1_CACHE.get(key)
            if last:
                return last
    import requests
    sym = ("sh" if code[0] == "5" else "sz") + code
    prices = {
        "sina": lambda: _sina_quote(sym, side),      # 优先：新狼 hq.sinajs.cn
        "tencent": lambda: _tencent_quote(sym, side),  # 备选：腾讯 qt.gtimg.cn
    }
    for name, fn in prices.items():
        try:
            p = fn()
            if p and p > 0:
                _L1_LAST[key] = now
                _L1_CACHE[key] = p
                return p
        except Exception:
            continue
    return _etf_price(code)


def _sina_quote(sym, side):
    import requests
    r = requests.get("https://hq.sinajs.cn/list=" + sym,
                     headers={"Referer": "https://finance.sina.com.cn"}, timeout=6)
    r.encoding = "gbk"
    f = r.text.split('"')[1].split(",")
    if len(f) >= 22:
        if side == "buy":
            return float(f[21]) if f[21] else float(f[3])
        return float(f[11]) if f[11] else float(f[3])
    return 0.0


def _tencent_quote(sym, side):
    """腾讯实时行情 qt.gtimg.cn 作为备用源。返回对手价或 0.0。"""
    import requests
    url = "https://qt.gtimg.cn/q=" + sym
    r = requests.get(url, timeout=6)
    r.encoding = "gbk"
    txt = r.text
    if '="' not in txt:
        return 0.0
    f = txt.split('"')[1].split("~")
    # 腾讯字段序：f[1]名称 f[3]现价 f[5]买一 f[6]卖一
    if len(f) < 6:
        return 0.0
    if side == "buy":
        v = f[6] if f[6] else f[3]
    else:
        v = f[5] if f[5] else f[3]
    try:
        return float(v)
    except Exception:
        return 0.0


def _live_position_qty(broker, code, field=None):
    """实盘：读取某代码的持仓数量。field=None 时取「可用金额(优先)/可用数量」；读不到返回 0。
    [2026-09 修复] 旧默认字段"可用数量"与 easytrader 实际返回的"可用余额"不符 → 恒为0；改经 _avail_qty 兼容。"""
    try:
        for h in broker.position:
            if h.get("证券代码") == code:
                try:
                    if field is None:
                        return _avail_qty(h)
                    return int(h.get(field, 0) or 0)
                except Exception:
                    return 0
    except Exception as e:
        raise RuntimeError(f"读取持仓失败 {code}：{e}")
    return 0


def _confirm_live_order(broker, code, side, expect_has_position):
    """下单后核对持仓（尽力而为，不阻塞交易）：
    - 委托刚提交时常未立即成交（集合竞价排队 / 客户端持仓刷新滞后 / 深市盘口延时），
      此刻读持仓=0 属正常，故【短延时轮询几次】再判，避免"已提交未成交"被误报 CRITICAL。
    - 轮询窗口内仍未匹配 → 降级为 WARNING（如实标注：已提交、待成交/对账确认），
      不置 confirm_halt（否则读持仓一坏/一时未成交就永久停摆），由 14:50 对账 + 14:57 尾盘追单兜底。
    - 读持仓本身失败（easytrader 依赖OCR未装等）→ 警告+按已提交处理。"""
    for _ in range(3):                       # 最多轮询3次，间隔3秒 ≈ 9秒成交确认窗口
        time.sleep(3)
        try:
            total = _live_position_qty(broker, code, "股票余额") + _live_position_qty(broker, code)
            ok = (total > 0) if expect_has_position else (total == 0)
            if ok:
                logger.info(f"下单后核对通过：{side} {code} 持仓={total}")
                return True
            # 未匹配 → 继续轮询（可能仍未成交）
        except Exception as e:
            # 读持仓本身失败（OCR/tesseract 未装等）→ 无法核对，按已提交处理
            logger.warning(f"{side} {code} 已提交，但持仓读取不可用（{e}），无法核对——按已提交处理，14:50对账兜底")
            return False
    # 轮询窗口内始终未匹配：如实记为"已提交待确认"，不再 as CRITICAL
    logger.warning(f"{side} {code} 已提交，但 {3 * 3}s 内持仓核对未匹配（疑似未成交/客户端刷新滞后）——已按已提交处理，14:50对账兜底")
    return False


def _sell_code(code, broker, price, reason="卖出"):
    if broker is None:
        raise RuntimeError("券商未连接（禁止回退）")
    try:
        _clear_panel_residual(broker)  # 清买卖面板残留
        _pos = _load_owned().get(code) or {}
        if _t1_today_sell_blocked(code, _pos):
            # T+1当日买入：今日卖不动（客户端"可卖数量"本就为0）。抛专用异常由主循环优雅延迟到次日，
            # 不落 _sell_code 失败分支(不报错、不触发 confirm_halt)，且中断后续买入避免双持仓。
            raise T1TradeDeferred(code)
        price = _l1_price(code, "sell")  # 卖出用买一（对手价），保证成交
        qty = (_pos).get("qty", 0)  # 用自有记录（不读客户端=不触发复制验证码）
        if qty <= 0:
            raise RuntimeError(f"自有记录无 {code} 持仓，无法卖出（禁止静默失败）")
        ret = _shortcut_trade(broker, code, "sell", price, qty)  # 转 32 位执行器真实下单
        # 下单成功 → 【立刻】移除自有持仓（防核对/后续64位确认读不到触发重复卖）
        _o = _load_owned()
        _o.pop(code, None)
        _save_owned(_o)
        logger.info(f"[卖出] {code} {qty}股 @ {price:.3f}（{reason}）→ {ret}")
        notify("卖出", f"{code} {qty}股 @ {price:.3f}（{reason}）")
        journal("sell", code, price, qty, reason)
        # 遗留的64位确认/收尾仅为尽力而为，绝不允许在已下单后抛异常(否则误导为失败可能重复卖)
        for _fn in (lambda: _dismiss_pending_prompts(broker, expect_qty=qty),
                    lambda: _confirm_live_order(broker, code, "卖出", expect_has_position=False)):
            try:
                _fn()
            except Exception as _e:
                logger.warning(f"[卖出] 订单已提交，64位确认收尾异常(忽略): {_e}")
    except Exception as e:
        logger.error(f"卖出{code}失败：{e}")
        raise


def _stop_horizon_hint(code, price, qty, avail):
    """买入后监控提示（打印+飞书）：投入/数量/单价/剩余可用/预计盈亏区间/建议几小时后开机检测。
    盈亏下界=触发止损(成本回撤 tr_pct)，上界=近期日均波动×5 的行收盘参考盈点（至少 tr_pct）。
    开机时间：距最坏止损缓冲(成本回撤 tr_pct) ÷ 近期日均波动 ≈ 交易日数，×24≈小时（粗估，非预测）。"""
    try:
        amount = qty * price
        vol = None
        try:
            h = dc.get_etf_k_history(code, days=90)
            if h is not None and not h.empty and "close" in h:
                closes = h["close"].astype(float)
                v = float(closes.pct_change().dropna().abs().tail(20).mean())
                if not pd.isna(v) and v > 0:
                    vol = v
        except Exception:
            pass
        tr_pct = CFG.get("SR_TRAIL_PCT", 0.03) or 0.03
        loss = amount * tr_pct                                  # 触发止损约亏
        up_pct = max(tr_pct, vol * 5) if vol else tr_pct        # 上行参考基本面更高
        up = amount * up_pct
        remaining = max(avail - amount, 0)
        if vol:
            days = tr_pct / vol                                  # 缓冲3% ÷ 日均波动
            xh = int(round(days * 24))
            advise = f"建议约{xh}小时后开机检测" if xh > 1 else "建议在本交易时段尽快开机检测"
        else:
            advise = "暂无波动数据，建议交易时段保持程序运行"
        msg = (f"监控提示：已购买，{code}，{amount:.0f}元，{qty}股 @ 单价{price:.3f}，" +
               f"剩余可用{remaining:.0f}元，预计盈亏[亏约{loss:.0f}～盈约{up:.0f}元]，" +
               f"{advise}，以防止损或换仓失效")
        logger.warning(msg)
        try:
            notify("监控提示", msg)
        except Exception:
            pass
    except Exception as _e:
        logger.debug(f"[监控提示] 生成失败(忽略): {_e}")


def _buy_code(code, broker, price, amount, reason="买入"):
    if price is None or price <= 0 or amount <= 0:
        return
    qty = int(amount / (price * 100)) * 100
    # 买入前按"可用现金"封顶：绝不超账户可买数量/资金（防脏单超买 + 资金不足被拒）
    try:
        avail = CFG["TOTAL_FUND"]
        bal = broker.balance
        avail = float(bal.get("可用金额", bal.get("总资产", CFG["TOTAL_FUND"])))
        max_qty_by_cash = int(avail / (price * 100)) * 100
        qty = min(qty, max_qty_by_cash)
    except Exception:
        pass
    if qty < 100:
        logger.info(f"[买入] {code} 可用资金不足（可买{qty}股<100），跳过买入")
        return
    if broker is None:
        raise RuntimeError("券商未连接（禁止回退）")
    try:
        _clear_panel_residual(broker)  # 清买卖面板残留
        price = _l1_price(code, "buy")  # 买入用卖一（对手价），保证成交
        ret = _shortcut_trade(broker, code, "buy", price, qty)  # 转 32 位执行器真实下单
        # 下单成功 → 【立刻】记入自有持仓（防核对/后续64位确认读不到触发重复下单）
        _o = _load_owned()
        _o[code] = {"qty": qty, "cost": price, "date": _today(), "buy_date": _today(), "peak": price}
        _save_owned(_o)
        logger.info(f"[买入] {code} {qty}股 @ {price:.3f}（{reason}）→ {ret}")
        notify("买入", f"{code} {qty}股 @ {price:.3f}（{reason}）")
        journal("buy", code, price, qty, reason)
        # 买入后监控提示：已购买/数量/单价/剩余可用/预计盈亏/建议开机时间（纯提示，吞错不影响流程）
        _stop_horizon_hint(code, price, qty, avail)
        # 遗留的64位确认/收尾仅为尽力而为，绝不允许在已下单后抛异常(否则误导为失败可能重复下)
        for _fn in (lambda: _dismiss_pending_prompts(broker, expect_qty=qty),
                    lambda: _confirm_live_order(broker, code, "买入", expect_has_position=True)):
            try:
                _fn()
            except Exception as _e:
                logger.warning(f"[买入] 订单已提交，64位确认收尾异常(忽略): {_e}")
    except Exception as e:
        logger.error(f"买入{code}失败：{e}")
        raise


def _alloc_weights(target_set, cur):
    """购买比例：目标内资金分配（对齐回测 backtest_alloc_v4 的 mom2）。
    返回 {code: 权重}，仅对"本次要新增买入"的目标 (target_set - cur) 计算，归一化和为 1。
    - 开启 MOM2_WEIGHT 且新增≥2只：权重 ∝ max(动量分,0)^2（把更多资金压到最强趋势）
    - 否则（等权/单只/防御/动量分无效）：等权。
    """
    new = list(target_set - cur)
    if len(new) <= 1:
        return {c: 1.0 for c in new}
    if CFG.get("MOM2_WEIGHT"):
        scores = STATE.get("scores", {}) or {}
        s = {c: max(float(scores.get(c, 0) or 0), 0.0) ** 2 for c in new}
        tot = sum(s.values())
        if tot > 1e-12:
            return {c: s[c] / tot for c in new}
    w = 1.0 / len(new)
    return {c: w for c in new}


def rebalance_to(target, broker):
    """把组合调整到 target（ETF 代码列表 或 'cash'）。"""
    codes = target if target != "cash" else []
    cur = current_holdings(broker)
    target_set = set(codes)

    # 卖出不在目标中的
    for c in cur - target_set:
        _sell_code(c, broker, _etf_price(c), reason="换仓·卖旧目标")
    # 买入目标中新增的（用"可用现金"而非"总资产"，避免动用被套的个股资金；
    # 购买比例按 mom2 动量加权，把更多现金给强势目标）
    if target_set:
        avail = CFG["TOTAL_FUND"]
        try:
            bal = broker.balance
            avail = float(bal.get("可用金额", bal.get("总资产", CFG["TOTAL_FUND"])))
        except Exception:
            pass
        weights = _alloc_weights(target_set, cur)
        for c, w in weights.items():
            _buy_code(c, broker, _etf_price(c), avail * w, reason="换仓·买新目标")
    logger.info(f"[再平衡完成] 目标={codes} 当前持仓={sorted(current_holdings(broker))}")


# ----------------------------- 实时风控（每决策周期执行，突破1天锁） -----------------------------
def _sina_mtm():
    """用 sina 现价 × 本地持仓记录 估算组合市值（不读客户端=不拉窗口、不触发验证码）。
    依据：单标的轮动、买入量为整仓，本地持仓 qty×现价 即可稳健估算当日盈亏敞口。
    空仓/读不到价格/价格异常 → 返回 None（表示"无法即时估算，跳过即时熔断"）。"""
    try:
        owned = _load_owned()
        if not owned:
            return None                       # 空仓无敞口
        total = 0.0
        for code, pos in owned.items():
            qty = int((pos or {}).get("qty", 0) or 0)
            if qty <= 0:
                continue
            px = _l1_price(code, "sell")      # sina 实时对手价；拉不到回落最新收盘
            if px is None or px <= 0:
                return None                    # 某标的价格取不到 → 本轮没法估，跳过（安全）
            total += px * qty
        return total if total > 0 else None
    except Exception:
        return None


def portfolio_mark_to_market(broker):
    """组合当前市值：读券商账户总资产（读不到返回 None 触发告警而非回退）。"""
    if broker is None:
        raise RuntimeError("券商未连接（禁止回退）")
    try:
        return float(broker.balance["总资产"])
    except Exception as e:
        logger.error(f"读取总资产失败：{e}")
        return None


# ----------------------------- 支撑/阻力(S/R)动态止损 -----------------------------
# 回测验证(2016-2026全历史)：把固定-8%硬止损升级为"最近支撑动态止损"(V2a)更优
# —— 年化 7.44%->10.29%、回撤 -17.34%->-12.30%，且跑赢等权6ETF(9.71%)。
# 用 research/sr_levels.py(scipy-free 纯 numpy) 在近期收盘价窗口算支撑；无有效支撑/数据不足
# 时回退固定-8%(与原行为一致，安全兜底)。逻辑严格对齐 research/backtest_sr_enhance.py 的 V2a。
_SR_HIST_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "sr_history.json")


def _seed_sr_history():
    """首次运行：用 cache/backtest/etf_<code>.csv 的真实ETF日线收盘价给各标的播种近期历史。
    之后只追加本地记录，不依赖网络/客户端。已存在则跳过。"""
    if os.path.exists(_SR_HIST_FILE):
        return
    try:
        hist = {}
        base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "backtest")
        for code in CFG["ETF_UNIVERSE"]:
            fn = os.path.join(base, f"etf_{code}.csv")
            if os.path.exists(fn):
                df = pd.read_csv(fn, dtype={"date": str})
                s = df["close"].astype(float).tolist()
                if len(s) >= 60:
                    hist[code] = s
        if hist:
            os.makedirs(os.path.dirname(_SR_HIST_FILE), exist_ok=True)
            with open(_SR_HIST_FILE, "w", encoding="utf-8") as fp:
                json.dump(hist, fp)
            logger.info(f"[S/R] 已用 etf 日线播种历史：{ {k: len(v) for k, v in hist.items()} }")
    except Exception as e:
        logger.warning(f"[S/R] 历史播种失败(将回退固定-8%)：{e}")


def _sr_recent_closes(code):
    """返回 code 近期收盘价列表(旧->新)；播种文件缺失则回退 None。"""
    try:
        if not os.path.exists(_SR_HIST_FILE):
            _seed_sr_history()
        if os.path.exists(_SR_HIST_FILE):
            with open(_SR_HIST_FILE, "r", encoding="utf-8") as fp:
                return json.load(fp).get(code)
    except Exception:
        pass
    return None


def _append_sr_daily():
    """每日14:50后追加一次各标的当日收盘价(买一对手价)到本地历史，使 S/R 窗口持续滚动。"""
    try:
        hist = {}
        if os.path.exists(_SR_HIST_FILE):
            with open(_SR_HIST_FILE, "r", encoding="utf-8") as fp:
                hist = json.load(fp)
        changed = False
        for code in CFG["ETF_UNIVERSE"]:
            px = _l1_price(code, "buy")
            if not px or px <= 0:
                continue
            s = hist.get(code, [])
            if s and abs(s[-1] - px) < 1e-9:   # 同日不重复追加
                continue
            s.append(px)
            hist[code] = s[-(CFG["SR_WIN"] * 2):]   # 仅保留最近 SR_WIN*2 点，防无限增长
            changed = True
        if changed:
            os.makedirs(os.path.dirname(_SR_HIST_FILE), exist_ok=True)
            with open(_SR_HIST_FILE, "w", encoding="utf-8") as fp:
                json.dump(hist, fp)
    except Exception as e:
        logger.warning(f"[S/R] 日线追加失败：{e}")


def _sr_support_stop_price(code, entry, current_price):
    """给定持仓成本 entry 与当前价 current_price，返回动态止损价。
    无有效支撑/数据不足 -> 回退固定 -8%(与原行为一致，安全兜底)。
    对齐 V2a：支撑距成本 [4%,15%] 用支撑；太近->-8%；太远->封顶-15%；无支撑->-8%。"""
    if not CFG.get("SR_STOP") or _srl is None or entry <= 0 or current_price <= 0:
        return entry * (1 + CFG["HARD_STOP_PCT"])
    closes = _sr_recent_closes(code)
    if closes is None or len(closes) < 60:
        return entry * (1 + CFG["HARD_STOP_PCT"])
    try:
        w = closes[-CFG["SR_WIN"]:]
        df = pd.DataFrame({"close": pd.Series(w, dtype=float)})
        df["open"] = df["close"].shift(1).fillna(df["close"])
        df["high"] = df[["open", "close"]].max(axis=1)
        df["low"] = df[["open", "close"]].min(axis=1)
        levels = _srl.get_levels(df, CFG["SR_ORDER"], CFG["SR_MERGE"])
        sup = []
        for lv in levels:
            if _srl.strength(lv) < CFG["SR_MIN_STRENGTH"]:
                continue
            if lv.level_value < current_price:
                sup.append(lv.level_value)
        nsup = max(sup) if sup else None
        if nsup is None:
            return entry * (1 + CFG["HARD_STOP_PCT"])
        dist = (entry - nsup) / entry
        if dist < CFG["SR_STOP_CAP_LO"]:
            return entry * (1 + CFG["HARD_STOP_PCT"])
        if dist > CFG["SR_STOP_CAP_HI"]:
            return entry * (1 - CFG["SR_STOP_CAP_HI"])
        return nsup
    except Exception as e:
        logger.warning(f"[S/R] 止损价格计算失败({code})，回退固定-8%：{e}")
        return entry * (1 + CFG["HARD_STOP_PCT"])


def _sr_nearest_resistance(code, cur):
    """给定当前价 cur，返回价格上方最近的、强度达标的阻力位；无有效阻力/数据不足 -> None。
    与 _sr_support_stop_price 共用同一套 get_levels 计算，仅取 price 上方的价位取最小。"""
    if not CFG.get("SR_STOP") or _srl is None or cur <= 0:
        return None
    closes = _sr_recent_closes(code)
    if closes is None or len(closes) < 60:
        return None
    try:
        w = closes[-CFG["SR_WIN"]:]
        df = pd.DataFrame({"close": pd.Series(w, dtype=float)})
        df["open"] = df["close"].shift(1).fillna(df["close"])
        df["high"] = df[["open", "close"]].max(axis=1)
        df["low"] = df[["open", "close"]].min(axis=1)
        levels = _srl.get_levels(df, CFG["SR_ORDER"], CFG["SR_MERGE"])
        res = []
        for lv in levels:
            if _srl.strength(lv) < CFG["SR_MIN_STRENGTH"]:
                continue
            if lv.level_value > cur:
                res.append(lv.level_value)
        return min(res) if res else None
    except Exception as e:
        logger.warning(f"[S/R] 阻力计算失败({code})，回退无过滤：{e}")
        return None


def apply_sr_entry_filter(target):
    """V2d 入场过滤：买价落在最近阻力 SR_ENTRY_BUF 以内则跳过该标的(不追高到压力位前)。
    对齐 backtest V2d：entry >= nres*(1-ENTRY_BUF) 即跳过。target='cash' 或 未开启时原样返回。
    若过滤后非空则保留(防御标的始终保留)；若全被剔除(非防御)则转 'cash'(与回测 port=0 一致)。"""
    if not CFG.get("SR_ENTRY_FILTER") or target == "cash":
        return target
    kept, skipped = [], []
    for c in target:
        if c == CFG["DEFENSIVE"]:
            kept.append(c)
            continue
        cur = _l1_price(c, "buy")  # 买价=卖一对手价(真实成交价)
        if cur is None or cur <= 0:
            kept.append(c)          # 取不到价则保守保留，不误杀
            continue
        nres = _sr_nearest_resistance(c, cur)
        if nres is not None and cur >= nres * (1 - CFG["SR_ENTRY_BUF"]):
            skipped.append((c, round(nres, 3)))
            continue
        kept.append(c)
    if skipped:
        logger.info(f"[S/R入场过滤] 跳过贴近阻力标的：{skipped}")
    if not kept:
        return "cash"              # 全部贴近阻力 -> 空仓观望(不追高)
    return kept


def enforce_hard_stop(broker):
    """单只持仓触发止损立即砍，不受 MIN_HOLD_DAYS 限制。
    每决策周期(约120秒)调用一次。止损价 = max(S/R支撑动态止损, 跟踪止损)。
    - S/R 支撑动态止损：无有效支撑/数据不足/未开启时回退固定 -8%。
    - 跟踪止损(SR_TRAIL)：随价格创新高把止损上移，止损=峰值*(1-TRAIL_PCT)，
      只升不降，锁利润少回吐（回测 V3 2026-09：V2d+trail6% 优于 V2d）。
    - 防御资产(黄金)高点回落(DEF_PEAK_STOP)：同跟踪思路、阈值4%，回测 backtest_gold_peak_stop.py
      最优（年化+4.9pp、回撤持平、换手仅+8次）——解决黄金冲高获利后利润守不住。"""
    if broker is None:
        raise RuntimeError("券商未连接（禁止回退）")
    try:
        # 用自有持仓记录做止损（不读客户端持仓=不触发复制验证码）
        _o = _load_owned()
        changed = False
        trail_on = CFG.get("SR_TRAIL") and CFG.get("SR_TRAIL_PCT", 0) > 0
        def_peak_pct = CFG.get("DEF_PEAK_STOP", 0) or 0  # 防御资产(黄金)专用高点回落阈值；0=不叠加
        for code, p in list(_o.items()):
            cost = p.get("cost", 0) or 0
            if cost <= 0:
                continue
            px = _l1_price(code, "sell")  # 止损用买一对手价，确保卖得掉
            if px is None or px <= 0:
                continue
            # 跟踪峰值：防御资产(黄金)用 DEF_PEAK_STOP，普通标的用 SR_TRAIL(有配置才开)
            if code == CFG["DEFENSIVE"]:
                tr_pct = def_peak_pct if def_peak_pct > 0 else None
            else:
                tr_pct = CFG["SR_TRAIL_PCT"] if trail_on else None
            has_trail = tr_pct is not None and tr_pct > 0
            if has_trail:
                peak = p.get("peak", cost) or cost
                if px > peak:
                    peak = px
                    p["peak"] = peak
                    changed = True
                trail_stop = peak * (1 - tr_pct)
            else:
                trail_stop = None
            base_stop = _sr_support_stop_price(code, cost, px)  # S/R 动态止损(含-8%兜底)
            stop = max(base_stop, trail_stop) if trail_stop is not None else base_stop
            if px <= stop:
                mode = "跟踪+S/R止损" if trail_stop is not None and stop == trail_stop else "S/R动态止损" if CFG.get("SR_STOP") else "固定-8%硬止损"
                logger.warning(f"{mode} {code}：现价{px:.3f} ≤ 止损价{stop:.3f}（成本{cost:.3f}，峰值{peak if has_trail else cost:.3f}）")
                if _t1_today_sell_blocked(code, p):
                    # T+1当日买入触发止损，但今日卖不了（客户端可卖数量=0，制度约束）。不如实卖出、
                    # 不抛错打断主循环；次日(buy_date≠today)由常规清扫自然卖出。
                    # 注：现为单标的持仓(HOLD_N=1)、一次一笔，无"多只待卖优先级"，无需额外记录。
                    logger.warning(f"[T+1] {code} 今日刚买入触发止损{stop:.3f}但今日卖不了，保留持仓，次日(可卖)自然止损")
                    return  # 一次一笔原则；其余持仓下一周期再处理
                qty = p.get("qty", 0)  # 用自有记录数量
                if qty > 0:
                    _shortcut_trade(broker, code, "sell", px, qty)  # 快捷键F2止损卖出
                    journal("stop_sell", code, px, qty, mode)
                    _mark_stop_sell(code)  # 记录止损卖出日 → 触发再入冷却(防当天接回抖振)
                    _o.pop(code, None)  # 止损卖出后移出自有持仓（防重复砍）
                    _save_owned(_o)
                    STATE["_stop_today"] = True  # 止损后当日解锁再平衡：让卖出的钱可尽快再配置
                    changed = False
                    return  # 一次止损一笔，下一周期继续处理其余
                else:
                    raise RuntimeError(f"止损时 {code} 记录数量为 {qty}，无法卖出")
        if changed:
            _save_owned(_o)
    except Exception as e:
        logger.error(f"止损失败：{e}")
        raise


# ----------------------------- 主循环 -----------------------------
def _topup_cash(broker, held):
    """补仓：当账上有「闲置现金」而当前持仓已是策略目标时，把闲钱补入已持有目标（"能买就买"）。
    触发前提：调用方保证 current_holdings()==策略目标（即只在无冲突窗口，避免抢跑/双持仓），交易时段内。
    - 阈值：可用现金 ≥ 总资产1% 且 能买≥100股（闲钱太少不值得下单=刷手续费）。
    - 买量：按可用现金 × 权重(单目标=全投)取100股整数；_buy_code 内部再用实时可用金额封顶，绝不超买。
    - 每日最多1次：由调用方用 STATE["_topup_day"]!=today 闸门控制。
    回测依据 research/backtest_cash_deploy.py：每闲置10%资金年化约拖3.5pp，满仓更优。"""
    try:
        bal = broker.balance
        avail = float(bal.get("可用金额", 0) or 0)
        total = float(bal.get("总资产", 0) or 0)
    except Exception as e:
        logger.warning(f"[补仓] 读取账户失败：{e}")
        return
    if total <= 0 or avail <= 0:
        return
    if avail < total * 0.01:                             # 闲钱<总资产1% → 不值得下单
        logger.info(f"[补仓] 闲钱不足总资产1%（可用{avail:.0f}/总{total:.0f}），跳过")
        return
    eff = [c for c in held if c]                          # 只买已持有的目标标的
    if not eff:
        return
    weights = _alloc_weights(set(eff), set())             # 单目标{1.0}，多目标按mom2/等权
    for c in eff:
        px = _etf_price(c)
        if not px or px <= 0:
            continue
        amt = avail * weights.get(c, 0)
        qty = int(amt / (px * 100)) * 100
        if qty < 100:
            logger.info(f"[补仓] {c} 现金不足买100股（可买{qty}），跳过")
            continue
        logger.info(f"[补仓] {c} 可用现金{avail:.0f}≥总资产1%，补入 {qty}股≈{px*qty:.0f}元")
        _buy_code(c, broker, px, amt, reason="补仓·闲钱买入")                       # 对手价+实时可用金额封顶


def main_loop():
    if not acquire_singleton_lock():
        return
    mode = "模拟" if CFG["PAPER"] else "实盘"
    logger.info(f"===== 轮动自动交易启动（初始模式：{mode}）=====")
    # 统一驱动同花顺客户端：attach 后按"客户端登录的账户"自动识别模拟/实盘
    broker = connect_broker()
    if broker is None:
        logger.critical("券商连接失败：拒绝运行（禁止回退）")
        notify("启动失败", "券商连接失败，程序退出。不回退。")
        return
    mode = "模拟盘" if CFG["PAPER"] else "实盘"
    notify("轮动自动交易启动", f"模式：{mode}（按客户端登录账户自动识别；总资产见上一条通知）")
    logger.info(f"===== 模式确认：{mode} =====")
    _seed_sr_history()   # 首次运行用 etf 日线给 S/R 止损播种近期历史（失败仅告警，止损回退固定-8%）
    last_day, last_rebalance = "", ""
    # 从状态文件恢复上次调仓日：关机/开机后 5 天锁依然生效，不因重启而重置
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            last_rebalance = json.load(f).get("last_rebalance", "")
    except Exception:
        pass
    last_decide = None          # 上次决策时刻（节流用）
    last_risk_off = False       # 上次风险-off 状态（检测"新触发"）
    last_stop_sweep = None      # 上次「只读快巡检」峰值/止损判定时刻（A项，独立于120s决策）
    DECIDE_INTERVAL = 120        # 秒：2分钟。日常心跳只用 sina 行情+本地记录（不读客户端=不触发验证码=不抢鼠标），2分钟让止损/防守更及时

    # —— 启动即对账(2026-09)：每次启动都读一次客户端真实持仓，纠正本地记录脱节（如 518880 漏卖类场景）。
    # 低频且内部吞错：读到持仓以「合并式」写入本地(不清空原记录)；被验证码/读持仓阻断时仅告警、沿用本地，
    # 避免"启动时误以为已盘清"导致漏卖/漏止损。当天用 _boot_recon 闸门去重。
    if STATE.get("_boot_recon") != _today():
        STATE["_boot_recon"] = _today()
        try:
            _sync_owned_from_client(broker)
        except Exception as _e:
            logger.error(f"启动对账失败(继续启动)：{_e}")

    try:
        while True:
            if check_kill_switch():
                notify("已停止", "KILL_SWITCH 存在，进程将退出")
                break
            # 盘中热更新：每轮决策边界前重读 DB 现役参数（调参器改库后立即生效，无需重启；
            # 仅影响后续 target/止损，不强迫已持仓立即换仓）。DB 不可用→保持当前参数。
            try:
                _reparam = refresh_active_params(log_change=False)
                # R6 堵漏：参数发生热更时，强制下轮以新参数重新决策并重刷防御迟滞，
                # 避免切换当轮用旧参数遗留的 last_decide/_def_held 做判断（参数/状态不一致）。
                if _reparam:
                    last_decide = None
                    STATE["_def_held"] = CFG["DEFENSIVE"] in set(_load_owned().keys())
            except Exception:
                pass
            now = datetime.datetime.now()
            today = _today()

            if today != last_day:
                last_day = today
                # —— 每交易日结束时自动评估一次（自动调参器）——
                # 用真实成交(默认排除 is_sim 模拟)算各参数组实盘表现，满足护栏才切换现役；
                # 现况样本 <20 交易日 → 维持现役，仅留档审计。异常静默，不影响交易。
                try:
                    import auto_tuner
                    _tuner = auto_tuner.run()
                    # 调参器落库(含分步逼近)后立即热更，消除"DB已切但内存未同步"的错标窗口；
                    # 切换发生时重刷决策状态，保证 next 决策/止损用新参数、状态与参数一致。
                    if _tuner and _tuner.get("changed"):
                        if refresh_active_params(log_change=True):
                            last_decide = None
                            STATE["_def_held"] = CFG["DEFENSIVE"] in set(_load_owned().keys())
                except Exception:
                    pass
                STATE["daily_loss_halt"] = False
                STATE["confirm_halt"] = False  # 新的一天解除核对暂停
                STATE["_stop_today"] = False   # 新的一天：止损解锁标记复位
                STATE.pop("_recon_day", None)  # 重置当日对账闸门（对账只在14:50按点触发，非启动即触发）
                STATE.pop("_tail_day", None)   # 重置当日尾盘追单闸门
                # 迟滞状态种子：以本地自有持仓为准——已持有防御资产则置 True。
                # 关键：防止跨重启 _def_held 从 False 起步，在"已持有518880但动量略低于进入阈值"时
                # 误判为未持有→突变卖出防御资产（应维持持有直到跌破退出阈值-0.8%）。
                STATE["_def_held"] = CFG["DEFENSIVE"] in set(_load_owned().keys())
                # [静默] 空仓不读账户(拉窗口)：无敞口无需当日权益，待持仓时再捕获
                STATE["day_open_equity"] = portfolio_mark_to_market(broker) if _load_owned() else None

            # 每天 14:50 例行核对一次客户端持仓：独立按点触发 + 当日闸门去重。
            # （修复：原实现对账挂在 today!=last_day 分支里，程序若早于14:50启动，
            #   首轮即置位 last_day，导致当天 14:50 对账永远不会执行——正是"委托挂单
            #   未成交却无人兜底"的根因之一。）
            if (now.hour == CFG["REBALANCE_CHECK_TIME"][0]
                    and now.minute >= CFG["REBALANCE_CHECK_TIME"][1]
                    and STATE.get("_recon_day") != today):
                STATE["_recon_day"] = today
                try:
                    _sync_owned_from_client(broker)
                except Exception as _e:
                    logger.warning(f"14:50 对账异常(外层保护，继续主循环)：{_e}")
                # 14:50 对账结果驱动即时追单：本地与目标偏离且仍在交易时段 → 立即拉平，
                # 避免等 14:55 才追、遇验证码慢拖过收盘。失败则交给 14:55 尾盘兜底。
                try:
                    if dc.is_trading_day(now) and now.hour * 100 + now.minute <= 1500:
                        _tgt = decide_target()
                        _ts = set(_tgt) if _tgt != "cash" else set()
                        _cs = current_holdings(broker)
                        if _ts != _cs:
                            logger.warning(f"[对账即追] 持仓 {sorted(_cs)} ≠ 目标 {sorted(_ts)}，立即追平")
                            rebalance_to(_tgt, broker)
                except Exception as _e:
                    logger.warning(f"[对账即追] 失败(跳过，交14:55尾盘兜底)：{_e}")

            # 尾盘(14:55)未成交追单兜底：收盘前拉平仓位，防止委托挂单未成交导致仓位偏离目标。
            # 独立按点触发 + 当日闸门去重；需在交易时段内且未收盘(≤15:00)。
            if (now.hour == CFG["REBALANCE_CHECK_TIME"][0] and now.minute >= CFG["REBALANCE_CHECK_TIME"][1] + 5
                    and now.hour * 100 + now.minute <= 1500
                    and STATE.get("_tail_day") != today
                    and dc.is_trading_day(now)):
                STATE["_tail_day"] = today
                _tail_pursuit(broker)

            # 每日14:50后追加一次各标的收盘价到 S/R 本地历史（滚动窗口）；用 STATE 内的日标志去重
            if now.hour >= CFG["REBALANCE_CHECK_TIME"][0] and now.minute >= CFG["REBALANCE_CHECK_TIME"][1]:
                if STATE.get("_sr_append_day") != today:
                    STATE["_sr_append_day"] = today
                    _append_sr_daily()

            if not dc.is_trading_day(now):
                _sleep_until_next_session()   # 非交易日：睡到下一交易日 09:15，避免每5分钟空转
                continue

            # [时序优化·提前长睡] 非交易时段(午休/盘前/盘后)直接睡到下一交易时段起点。
            # 放在 need_decide 之前、独立判断，避免"午休刚切换时 need_decide=False 走 sleep(30)
            # 空转几轮才入睡"。这解决两个真实低效：
            #   a) 午休 11:30 后约有 2 分钟 30s 空转过渡
            #   b) 下午 13:00 / 次日 9:15 开盘最多延迟 5 分钟才恢复监控
            hm = now.hour * 100 + now.minute
            in_session = (915 <= hm <= 1130) or (1300 <= hm <= 1500)  # 真实可交易时段
            if not in_session:
                # 保留每日一次预热（保证开盘即有最新目标）
                if STATE.get("_prewarm") != today:
                    STATE["_prewarm"] = today
                    target = decide_target()
                    last_risk_off = bool(STATE["risk_off"])
                    logger.info(f"[非交易时段 {now:%H:%M}] 预热目标={target}，不下单")
                _sleep_until_next_session()
                continue

            # ---------- A项：只读快循环（峰值/止损 5~10s 巡检，独立于 120s 决策）----------
            # 交易时段内每 STOP_SWEEP_INTERVAL 秒做一次挡损判定：只读 sina 行情+本地持仓记录，
            # 更新持仓峰值、判断是否触发止损。不读客户端、不点鼠标（不触发复制验证码/抢鼠标）；
            # 仅当真正触发止损时才内部调 _shortcut_trade 操作客户端。价值=从"120s内冲高又回落的
            # 峰值漏更新"中更早离场、更早锁利润。非交易时段不巡检（长睡省请求）。
            hm = now.hour * 100 + now.minute
            if in_session:
                if last_stop_sweep is None or (now - last_stop_sweep).total_seconds() >= CFG["STOP_SWEEP_INTERVAL"]:
                    last_stop_sweep = now
                    # 防爬弹窗存在则跳过本轮(避免重读持续触发)；挡损与决策共用同一闸门
                    if not _captcha_blocked(broker):
                        try:
                            enforce_hard_stop(broker)  # 只读挡损：触发才碰客户端
                        except Exception as _se:
                            logger.error(f"快巡检挡损异常(继续)：{_se}", exc_info=True)
                # —— 盘中自动调参（增量更新）——
                # 交易时段内每 TUNE_INTERVAL 秒评估一次 auto_tuner（满足护栏才切现役）。
                # 切换后由决策边界前 refresh_active_params 即时热更；收紧止损会按峰值重算跟踪止损
                # → 跌破即真实离场（用户确认允许）。异常 / DB 不可用静默，绝不影响交易。
                if (STATE.get("_tune_at") is None
                        or (now - STATE["_tune_at"]).total_seconds() >= CFG["TUNE_INTERVAL"]):
                    STATE["_tune_at"] = now
                    try:
                        import auto_tuner
                        _tuner = auto_tuner.run()
                        # 盘中切换同样"落库即热更"，关闭错标窗口；切换时重刷决策状态(与上面一致)
                        if _tuner and _tuner.get("changed"):
                            if refresh_active_params(log_change=True):
                                last_decide = None
                                STATE["_def_held"] = CFG["DEFENSIVE"] in set(_load_owned().keys())
                    except Exception:
                        pass

            chk = now.replace(hour=CFG["REBALANCE_CHECK_TIME"][0],
                              minute=CFG["REBALANCE_CHECK_TIME"][1], second=0, microsecond=0)
            # 常规：距上次再平衡需够 MIN_HOLD_DAYS（省手续费）；
            # 风险-off：突破 MIN_HOLD_DAYS 立即防守（资本优先）。
            # 止损解锁：当日触发过止损(_stop_today)时跳过持仓锁，让卖出的钱能当天再配置（回测外的事务优化，避免止损后现金长期闲置）。
            enough_hold = (STATE.get("_stop_today") or last_rebalance == "" or
                           (now.date() - _date(last_rebalance)).days >= CFG["MIN_HOLD_DAYS"])

            # 是否需要决策：① 到达检查时刻且(够持有期 或 已知风险-off)；② 周期巡检(检测新风险-off)
            need_decide = False
            if now >= chk and (enough_hold or last_risk_off):
                need_decide = True
            elif last_decide is None or (now - last_decide).total_seconds() >= DECIDE_INTERVAL:
                need_decide = True

            if need_decide:
                # 检测同花顺"正在拷贝数据"防爬弹窗：存在则跳过本轮不读不交易(避免重读持续触发)
                if _captcha_blocked(broker):
                    last_decide = now
                    continue
                try:
                    hm = now.hour * 100 + now.minute
                    in_hours = (915 <= hm <= 1130) or (1300 <= hm <= 1500)  # 交易时段(排除11:30-13:00午休);其余只决策不下单
                    if not in_hours:
                        # 非交易时段不做无意义心跳：每天仅开机预热一次(保证开盘即有最新目标)，
                        # 之后精确睡到下一个交易时段起点，不再 300s 空转刷请求/浪费电。
                        # 9:15进入交易时段后自动恢复 120 秒心跳（此时才需要 2 分钟级止损/防守及时性）。
                        if STATE.get("_prewarm") != today:
                            STATE["_prewarm"] = today
                            target = decide_target()
                            last_risk_off = bool(STATE["risk_off"])
                            logger.info(f"[非交易时段 {now:%H:%M}] 预热目标={target}，不下单")
                        _sleep_until_next_session()
                        continue
                    else:
                        if STATE["confirm_halt"]:
                            logger.warning("下单核对失败已暂停当日交易（等人工核实），本轮不交易")
                            last_decide = now
                            continue
                        enforce_hard_stop(broker)  # 硬止损：用 sina 行情+本地记录，不读客户端、不点鼠标
                        # 单日亏损熔断：盘中每个决策周期用 sina现价×本地持仓 即时估算（不拉窗口）。
                        # 回测(backtest_daily_halt)证明"盘中即时触发"远优于"仅14:50尾盘判"(年化31.75%→51.02%、
                        # 样本外Calmar 2.52→6.55)；用sina估值无需读客户端=不触发复制验证码/不抢鼠标。
                        if not STATE["daily_loss_halt"]:
                            mtm = _sina_mtm()                     # 不读客户端(空仓/取价失败→None)
                            if mtm and STATE.get("day_open_equity"):
                                dloss = (mtm - STATE["day_open_equity"]) / STATE["day_open_equity"]
                                if dloss <= -CFG["MAX_DAILY_LOSS_PCT"]:
                                    logger.warning(f"单日亏损(盘中sina) {dloss*100:.1f}% 超 {CFG['MAX_DAILY_LOSS_PCT']*100:.0f}%，触发当日熔断→转防御")
                                    STATE["daily_loss_halt"] = True
                        # [回退] sina 无法即时估值(取价失败/本地无记录)时，到14:50例行检查点读一次客户端对账判定
                        if not STATE["daily_loss_halt"] and now >= chk:
                            if current_holdings(broker):
                                _mtm_c = portfolio_mark_to_market(broker)
                                if _mtm_c and STATE.get("day_open_equity"):
                                    _dl = (_mtm_c - STATE["day_open_equity"]) / STATE["day_open_equity"]
                                    if _dl <= -CFG["MAX_DAILY_LOSS_PCT"]:
                                        logger.warning(f"单日亏损(对账) {_dl*100:.1f}% 超 {CFG['MAX_DAILY_LOSS_PCT']*100:.0f}%，触发当日熔断→转防御")
                                        STATE["daily_loss_halt"] = True
                        target = decide_target()
                        if STATE["daily_loss_halt"] and target != "cash":
                            target = [CFG["DEFENSIVE"]]  # 当日熔断：直接去防御资产(黄金)避险
                        target = apply_sr_entry_filter(target)  # V2d 入场过滤：贴近阻力的标的跳过
                        target = _filter_cooldown_targets(target)  # 止损后再入冷却：刚止损卖出的标的本轮禁止买回
                        last_decide = now
                        target_set = set(target) if target != "cash" else set()
                        cur = current_holdings(broker)
                        # 目标未变则不重复调仓（仅心跳）。
                        # 注：不再"风险-off翻转日即时防守"——盘中瞬变的risk_off不作为强制调仓触发，
                        # 只在到达定时检查点(chk)且(够持有期 或 risk_off已知)时才按目标切换；
                        # 回测(backtest_daily_close)证明该做法年化 7.44% > 盘中即时防守 6.27%。
                        # 盘中下行保护由 enforce_hard_stop(硬止损) 与 当日熔断 兜底，均未移除。
                        if target_set != cur:
                            try:
                                rebalance_to(target, broker)
                                last_rebalance = today
                                try:  # 落盘，跨开机保留（1天锁与调仓日）
                                    with open(STATE_FILE, "w", encoding="utf-8") as f:
                                        json.dump({"last_rebalance": today}, f)
                                except Exception:
                                    pass
                            except T1TradeDeferred:
                                # T+1 当日买入卖不动：资产未动，**不**记为已完成调仓(否则 last_rebalance=today
                                # 会让次日被 MIN_HOLD_DAYS 锁住、这只 T+1 永远卖不掉)。次日可卖时自然重试换仓。
                                # 注：现为单标的持仓(HOLD_N=1)、一次一笔，无"多只待卖优先级"，无需额外记录标记。
                                logger.warning("[T+1] 当日买入不可卖，本轮调仓未完成，次日(可卖)自然重试换仓")
                            except Exception as e:
                                # 下单失败也暂存调仓日，避免"下单异常→不落盘→每30s重试"的无限循环轰炸账户。
                                last_rebalance = today
                                try:  # 落盘，跨开机保留（1天锁与调仓日）
                                    with open(STATE_FILE, "w", encoding="utf-8") as f:
                                        json.dump({"last_rebalance": today}, f)
                                except Exception:
                                    pass
                                raise
                        else:
                            logger.info(f"[心跳] 目标未变={sorted(target_set)}，跳过调仓")
                            # 补仓：持仓==目标 且 账上有闲钱(≥1%总资产) → 把闲钱补入已持有目标（"能买就买"）。
                            # 每日最多1次（STATE["_topup_day"]!=today 闸门）；仅在目标已持有的无冲突窗口触发。
                            if (target != "cash" and
                                    STATE.get("_topup_day") != today):
                                try:
                                    _topup_cash(broker, target_set)
                                except Exception as _e:
                                    logger.error(f"[补仓] 异常：{_e}", exc_info=True)
                                finally:
                                    STATE["_topup_day"] = today
                        last_risk_off = bool(STATE["risk_off"])
                except Exception as e:
                    # 单周期异常绝不杀死守护进程：记录后继续下一个周期
                    logger.error(f"本周期决策/调仓异常（继续运行）：{e}", exc_info=True)
                    notify("本周期异常", str(e))

            time.sleep(30)
    except KeyboardInterrupt:
        logger.info("用户中断，退出")
    except Exception as e:
        logger.error(f"主循环异常：{e}", exc_info=True)
        notify("程序异常", str(e))


if __name__ == "__main__":
    main_loop()
