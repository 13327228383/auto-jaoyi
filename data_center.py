# -*- coding: utf-8 -*-
"""
data_center.py —— 免费 / 多源 / 带缓存的 A 股数据层
====================================================
解决的问题：
1) "获取不了交易数据"：akshare 为主，efinance / baostock 兜底；
   任何一个源挂了自动切换，失败返回空 DataFrame + 明确日志，绝不返回写死的假数据。
2) "几千只股票里挑不出"：一次性拉全市场快照并本地缓存；涨停筛选改用
   akshare 的「涨停股池」批量接口（一天一次调用返回全部涨停股），
   彻底替代原来「逐只股票拉 K 线」的 N+1 查询。
3) "调休策略不对"：用真实的交易日历（动态获取并缓存）判断交易日，
   不再用 now.weekday()（周一到周五）判断，正确处理调休上班/放假。

全部数据源均免费、开源、无需付费 token / 付费量化接口：
  - akshare  : https://github.com/akfamily/akshare   （全能，爬虫底层）
  - efinance : https://github.com/Micro-sheep/efinance （东财实时，精简单一）
  - baostock : http://baostock.com                  （极简，含交易日历）
安装：pip install akshare efinance baostock pandas numpy
（如只需主源，只装 akshare 也能跑，其余自动跳过）
"""

import os
import json
import datetime
import pandas as pd
import numpy as np

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# 对外暴露的稳定列名（与现有各策略文件约定一致）
STANDARD_COLS = ['股票代码', '股票名称', '涨跌幅(%)', '量比', '换手率(%)',
                 '总市值(亿元)', '最新价', '开盘价', '成交额', '成交量']


def _log(msg):
    print(f"[data_center] {msg}")


def _safe_float(v):
    try:
        if isinstance(v, str):
            return float(v.replace('%', '').replace(',', ''))
        return float(v)
    except Exception:
        return np.nan


def _to_full_symbol(code):
    """A 股代码补全交易所后缀（efinance / 部分接口需要）。"""
    code = str(code)
    if code.startswith(('60', '688', '900')):
        return code + '.SH'
    elif code.startswith(('000', '002', '003', '300', '301')):
        return code + '.SZ'
    return code


# ====================== 交易日历（解决调休） ======================
_CALENDAR_CACHE = os.path.join(CACHE_DIR, "trade_calendar.json")
_CALENDAR = None  # 进程内缓存


def load_trade_calendar(force=False):
    """返回交易日期集合(set of 'YYYYMMDD')，优先缓存，再动态拉取。"""
    global _CALENDAR
    if not force and _CALENDAR is not None:
        return _CALENDAR
    if not force and os.path.exists(_CALENDAR_CACHE):
        try:
            with open(_CALENDAR_CACHE, 'r', encoding='utf-8') as f:
                data = json.load(f)
            s = set(data.get('trade_dates', []))
            if s:
                _CALENDAR = s
                _log(f"交易日历命中缓存：{len(s)} 天")
                return s
        except Exception:
            pass
    s = set()
    # 1) akshare：tool_trade_date_hist_sina 返回历史交易日列表
    try:
        import akshare as ak
        df = ak.tool_trade_date_hist_sina()
        col = 'date' if 'date' in df.columns else df.columns[0]
        for d in df[col].astype(str):
            s.add(d.replace('-', '').replace('/', ''))
        _log(f"akshare 交易日历：{len(s)} 天")
    except Exception as e:
        _log(f"akshare 交易日历获取失败：{e}")
    # 2) baostock 兜底
    if not s:
        try:
            import baostock as bs
            bs.login()
            rs = bs.query_trade_dates()
            while (rs.error_code == '0') & rs.next():
                row = rs.get_row_data()
                if len(row) >= 2 and row[1] == '1':
                    s.add(row[0].replace('-', ''))
            bs.logout()
            _log(f"baostock 交易日历：{len(s)} 天")
        except Exception as e:
            _log(f"baostock 交易日历获取失败：{e}")
    if s:
        try:
            with open(_CALENDAR_CACHE, 'w', encoding='utf-8') as f:
                json.dump({'trade_dates': sorted(s)}, f)
        except Exception:
            pass
        _CALENDAR = s
    return s


def is_trading_day(dt=None):
    """真实交易日判断（含调休）。dt: datetime/date，默认今天。"""
    if dt is None:
        dt = datetime.datetime.now()
    ds = dt.strftime('%Y%m%d')
    cal = load_trade_calendar()
    if cal:
        return ds in cal
    # 兜底：仅排除周末（不含调休，仅应急用，建议保证能拉到真实日历）
    _log("无交易日历，回落到周末判断（不处理调休）")
    return dt.weekday() < 5


# ====================== 全市场实时快照（带缓存） ======================
def get_spot_snapshot(use_cache=True, max_age_minutes=15):
    """一次性获取全市场 A 股实时快照（统一列），带本地缓存。"""
    now = datetime.datetime.now()
    if not is_trading_day(now):
        _log("今日非交易日，无实时行情")
        return pd.DataFrame()
    cache_file = os.path.join(CACHE_DIR, f"spot_{now.strftime('%Y%m%d')}.csv")
    if use_cache and os.path.exists(cache_file):
        age = (datetime.datetime.now().timestamp() - os.path.getmtime(cache_file)) / 60
        if age < max_age_minutes:
            try:
                df = pd.read_csv(cache_file, dtype={'股票代码': str})
                _log(f"快照命中缓存：{len(df)} 只")
                return df
            except Exception:
                pass
    df = _fetch_spot_akshare()
    if df is None or df.empty:
        df = _fetch_spot_efinance()
    if df is None or df.empty:
        _log("所有数据源均获取快照失败")
        return pd.DataFrame()
    try:
        df.to_csv(cache_file, index=False, encoding='utf-8-sig')
    except Exception:
        pass
    _log(f"快照获取成功：{len(df)} 只")
    return df


def _fetch_spot_akshare():
    try:
        import akshare as ak
        raw = ak.stock_zh_a_spot_em()
        if raw is None or raw.empty:
            return None
        mapping = {'代码': '股票代码', '名称': '股票名称', '涨跌幅': '涨跌幅(%)',
                   '量比': '量比', '换手率': '换手率(%)', '总市值': '总市值',
                   '最新价': '最新价', '今开': '开盘价', '成交额': '成交额', '成交量': '成交量'}
        keep = [c for c in mapping if c in raw.columns]
        df = raw[keep].copy()
        df.rename(columns={k: v for k, v in mapping.items() if k in keep}, inplace=True)
        df['涨跌幅(%)'] = df['涨跌幅(%)'].apply(_safe_float)
        df['换手率(%)'] = df['换手率(%)'].apply(_safe_float)
        df['量比'] = pd.to_numeric(df['量比'], errors='coerce').fillna(0)
        # 总市值单位兼容：akshare 历史版本曾用「万元」，新版用「元」。
        # 启发式：>=1e8 视为「元」(÷1e8)，否则视为「万元」(÷1e4)，均能正确得到「亿元」。
        raw_total = pd.to_numeric(df['总市值'], errors='coerce').fillna(0) \
            if '总市值' in df.columns else pd.Series([0.0] * len(df), index=df.index)
        df['总市值(亿元)'] = np.where(raw_total >= 1e8, raw_total / 1e8, raw_total / 1e4)
        df['最新价'] = pd.to_numeric(df['最新价'], errors='coerce').fillna(0)
        df['开盘价'] = pd.to_numeric(df['开盘价'], errors='coerce').fillna(0)
        df['成交额'] = pd.to_numeric(df['成交额'], errors='coerce').fillna(0)
        df['成交量'] = pd.to_numeric(df['成交量'], errors='coerce').fillna(0)
        df = df[(df['涨跌幅(%)'] >= -20) & (df['涨跌幅(%)'] <= 20)]
        for c in STANDARD_COLS:
            if c not in df.columns:
                df[c] = np.nan
        return df[STANDARD_COLS]
    except Exception as e:
        _log(f"akshare 快照失败：{e}")
        return None


def _fetch_spot_efinance():
    try:
        import efinance as ef
        raw = ef.stock.get_realtime_quotes()
        if raw is None or raw.empty:
            return None
        mapping = {'股票代码': '股票代码', '股票名称': '股票名称', '最新价': '最新价',
                   '涨跌幅': '涨跌幅(%)', '换手率': '换手率(%)', '量比': '量比',
                   '今开': '开盘价', '成交额': '成交额', '成交量': '成交量'}
        keep = [c for c in mapping if c in raw.columns]
        df = raw[keep].copy()
        df.rename(columns={k: v for k, v in mapping.items() if k in keep}, inplace=True)
        df['涨跌幅(%)'] = df['涨跌幅(%)'].apply(_safe_float)
        df['换手率(%)'] = df['换手率(%)'].apply(_safe_float)
        df['量比'] = pd.to_numeric(df['量比'], errors='coerce').fillna(0)
        df['最新价'] = pd.to_numeric(df['最新价'], errors='coerce').fillna(0)
        df['开盘价'] = pd.to_numeric(df.get('开盘价', np.nan), errors='coerce').fillna(0)
        df['成交额'] = pd.to_numeric(df['成交额'], errors='coerce').fillna(0)
        df['成交量'] = pd.to_numeric(df['成交量'], errors='coerce').fillna(0)
        df['总市值(亿元)'] = np.nan  # efinance 实时快照不含总市值
        df['股票代码'] = df['股票代码'].astype(str).str.zfill(6)
        df = df[(df['涨跌幅(%)'] >= -20) & (df['涨跌幅(%)'] <= 20)]
        for c in STANDARD_COLS:
            if c not in df.columns:
                df[c] = np.nan
        return df[STANDARD_COLS]
    except Exception as e:
        _log(f"efinance 快照失败：{e}")
        return None


# ====================== 指数行情（大盘风控） ======================
def get_limit_down_pool(date):
    """某交易日跌停股池（akshare 一次返回全部）。date: 'YYYYMMDD' 或 'YYYY-MM-DD'。"""
    d = date.replace('-', '')
    try:
        import akshare as ak
        df = ak.stock_dt_pool_em(date=d)
        if df is None or df.empty:
            return pd.DataFrame(columns=['代码', '名称'])
        return df[['代码', '名称']] if '代码' in df.columns else df
    except Exception as e:
        _log(f"跌停池 {d} 获取失败：{e}")
        return pd.DataFrame(columns=['代码', '名称'])


def get_recent_limit_down_symbols(days=1, end_date=None):
    """近 days 个交易日内出现过跌停的股票代码集合（批量）。"""
    if end_date is None:
        end_dt = datetime.datetime.now()
    else:
        end_dt = datetime.datetime.strptime(end_date, '%Y-%m-%d')
    cal = load_trade_calendar()
    trade_days = []
    d = end_dt
    while len(trade_days) < days and d.year > 2000:
        ds = d.strftime('%Y%m%d')
        if (not cal) or (ds in cal):
            if cal or d.weekday() < 5:
                trade_days.append(ds)
        d -= datetime.timedelta(days=1)
    symbols = set()
    for td in trade_days:
        pool = get_limit_down_pool(td)
        for c in pool['代码'].astype(str):
            symbols.add(c)
    _log(f"近{len(trade_days)}个交易日跌停股共 {len(symbols)} 只")
    return symbols


def get_index_k_history(code, start_date, end_date, adjust=''):
    """指数日线（akshare 东财）。code 如 '000300'。返回含 '收盘' 的 DataFrame。"""
    try:
        import akshare as ak
        sym = code if code.lower().startswith(('sh', 'sz')) else ('sh' + code if code.startswith('6') else 'sz' + code)
        df = ak.stock_zh_index_daily_em(symbol=sym, start_date=start_date.replace('-', ''), end_date=end_date.replace('-', ''))
        if df is None or df.empty:
            return None
        return df
    except Exception as e:
        _log(f"指数K线 {code} 失败：{e}")
        return None


def get_index_spot():
    try:
        import akshare as ak
        return ak.stock_zh_index_spot_em()
    except Exception as e:
        _log(f"指数行情获取失败：{e}")
        return pd.DataFrame()


# ====================== 日 K 线（多源兜底） ======================
def get_k_history(code, start_date, end_date, adjust='qfq'):
    """日 K 线。akshare 优先，efinance 兜底。返回含 '涨跌幅' 列的 DataFrame。"""
    df = _k_akshare(code, start_date, end_date, adjust)
    if df is None or df.empty:
        df = _k_efinance(code, start_date, end_date, adjust)
    if df is None or df.empty:
        _log(f"{code} K线获取失败")
        return pd.DataFrame()
    return df


def _k_akshare(code, start, end, adjust):
    try:
        import akshare as ak
        s = str(start).replace('-', '')
        e = str(end).replace('-', '')
        df = ak.stock_zh_a_hist(symbol=str(code), period='daily',
                                start_date=s, end_date=e, adjust=adjust)
        if df is None or df.empty:
            return None
        df['涨跌幅'] = df['涨跌幅'].apply(
            lambda x: _safe_float(x) if isinstance(x, str) else float(x))
        return df
    except Exception as ex:
        _log(f"akshare K线 {code} 失败：{ex}")
        return None


def _k_efinance(code, start, end, adjust):
    try:
        import efinance as ef
        df = ef.stock.get_quote_history(_to_full_symbol(code),
                                        beg=str(start), end=str(end), adjust=adjust)
        if df is None or df.empty:
            return None
        df['涨跌幅'] = pd.to_numeric(df['涨跌幅'], errors='coerce')
        return df
    except Exception as ex:
        _log(f"efinance K线 {code} 失败：{ex}")
        return None


# ====================== 涨停股池（批量，关键性能优化） ======================
def get_limit_up_pool(date):
    """某交易日涨停股池（akshare 一次返回全部）。date: 'YYYYMMDD' / 'YYYY-MM-DD'。"""
    d = str(date).replace('-', '')
    try:
        import akshare as ak
        df = ak.stock_zt_pool_em(date=d)
        if df is None or df.empty:
            return pd.DataFrame(columns=['代码', '名称'])
        keep = [c for c in ('代码', '名称') if c in df.columns]
        return df[keep].copy()
    except Exception as e:
        _log(f"涨停池 {d} 获取失败：{e}")
        return pd.DataFrame(columns=['代码', '名称'])


def get_recent_limit_up_symbols(days=5, end_date=None):
    """近 days 个交易日内出现过涨停的股票代码集合（批量，替代逐只 K 线循环）。"""
    if end_date is None:
        end_dt = datetime.datetime.now()
    else:
        end_dt = datetime.datetime.strptime(str(end_date), '%Y-%m-%d')
    cal = load_trade_calendar()
    trade_days, d = [], end_dt
    guard = 0
    while len(trade_days) < days and guard < 400:
        guard += 1
        ds = d.strftime('%Y%m%d')
        in_cal = (ds in cal) if cal else (d.weekday() < 5)
        if in_cal:
            trade_days.append(ds)
        d -= datetime.timedelta(days=1)
    symbols = set()
    for td in trade_days:
        pool = get_limit_up_pool(td)
        for c in pool['代码'].astype(str):
            symbols.add(c)
    _log(f"近 {len(trade_days)} 个交易日涨停股共 {len(symbols)} 只")
    return symbols


# ====================== ETF 行情（轮动策略用，仅 6 只，数据量极小） ======================
def _to_bs_etf(code):
    """ETF 代码补全 baostock 前缀：5xxxxx→sh. 1xxxxx→sz."""
    code = str(code)
    return ("sh." + code) if code.startswith("5") else ("sz." + code)


def get_etf_k_history(code, days=130, adjust="qfq"):
    """ETF 日线（最近 days 根）。akshare 东财优先，baostock 兜底。返回含 'date','close' 的 DataFrame。"""
    try:
        import akshare as ak
        df = ak.fund_etf_hist_em(symbol=str(code), period="daily", adjust=adjust)
        if df is None or df.empty:
            raise RuntimeError("空结果")
        df = df[["日期", "收盘"]].copy()
        df.columns = ["date", "close"]
        df["close"] = pd.to_numeric(df["close"], errors="coerce")
        return df.dropna().tail(days)
    except Exception:
        # 东财失败→sina 是常规兜底，静音不打印（用户只要结果，不要过程）
        try:
            import akshare as ak
            sina_sym = ("sh" + str(code)) if str(code).startswith("5") else ("sz" + str(code))
            df = ak.fund_etf_hist_sina(symbol=sina_sym)
            if df is None or df.empty:
                raise RuntimeError("空结果")
            d = df[["date", "close"]].copy()
            d["close"] = pd.to_numeric(d["close"], errors="coerce")
            return d.dropna().tail(days)
        except Exception:
            # sina→baostock 也是常规兜底，静音；只有 baostock 也失败才报
            try:
                import baostock as bs
                bs.login()
                end = datetime.datetime.now().strftime("%Y-%m-%d")
                start = (datetime.datetime.now() - datetime.timedelta(days=days * 2)).strftime("%Y-%m-%d")
                rs = bs.query_history_k_data_plus(_to_bs_etf(code), "date,close",
                                                  start_date=start, end_date=end, frequency="d", adjustflag="2")
                rows = []
                while rs.error_code == "0" and rs.next():
                    rows.append(rs.get_row_data())
                bs.logout()
                d = pd.DataFrame(rows, columns=["date", "close"])
                d["close"] = pd.to_numeric(d["close"], errors="coerce")
                return d.dropna().tail(days)
            except Exception as e3:
                _log(f"ETF K线 {code} 全部数据源失败：{e3}")  # 唯一该报的：真没数据
                return pd.DataFrame(columns=["date", "close"])


# ====================== 自检 ======================
if __name__ == "__main__":
    print("交易日:", is_trading_day())
    snap = get_spot_snapshot()
    print("快照数量:", len(snap))
    if not snap.empty:
        print(snap.head(3).to_string(index=False))
    lu = get_recent_limit_up_symbols(3)
    print("近3日涨停股:", len(lu))
