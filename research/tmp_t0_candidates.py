# -*- coding: utf-8 -*-
"""筛出 T+0 ETF 候选池（跨境/债券/货币/商品），排除A股股票ETF。"""
import sys
sys.path.insert(0, '.')
import akshare as ak
import pandas as pd

# T+0 关键词：跨境(QDII/美股/港股/海外)、债券、货币、商品(黄金白银原油铜)
T0_KW = [
    # 跨境/海外
    "标普", "纳指", "纳斯达克", "恒生", "H股", "港股", "日经", "225",
    "德国", "法国", "中国A50", "中概", "东南亚", "美国50", "美股",
    # 债券
    "债", "可转债", "国债", "政金", "公司债", "信用债",
    # 货币
    "货币", "现金添富",
    # 商品
    "黄金", "白银", "豆粕", "原油", "有色金属", "铜", "能源化工",
]
# A股股票ETF 排除词（含这些=股票类，T+1）
T1_EXCLUDE = [
    "沪深300", "中证500", "创业板", "科创50", "上证50", "中证1000",
    "MSCI", "红利", "消费", "医药", "证券", "银行", "芯片", "半导体",
    "军工", "新能源", "光伏", "智能汽车", "5G", "半导体50", "食品饮料",
    "价值", "成长", "白酒", " ETF", "中证", "上证", "深证", "300",
]

df = ak.fund_etf_category_sina(symbol="ETF基金")
# 归一化名称
df["_n"] = df["名称"].astype(str)

def is_t0(name):
    if any(x in name for x in T1_EXCLUDE):
        return False
    return any(x in name for x in T0_KW)

cand = df[df["_n"].apply(is_t0)][["代码", "名称"]]
# 转标准6位代码
cand["code6"] = cand["代码"].astype(str).str.extract(r"(\d{6})")
cand = cand.dropna(subset=["code6"]).drop_duplicates("code6")
print(f"共筛选出 T+0 候选 {len(cand)} 只：")
print(cand.to_string(index=False))