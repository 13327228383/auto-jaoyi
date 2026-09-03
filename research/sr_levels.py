# -*- coding: utf-8 -*-
"""
sr_levels.py —— 本地 vendored 副本 of alexpantyukhin/support_resistance_levels
(https://github.com/alexpantyukhin/support_resistance_levels  MIT)
仅用于研究/回测，不直接参与实盘下单。

原始算法：用 scipy.argrelextrema 找极值(open/close 的上下沿)，按百分比 delta 把
相近极值聚成支撑/阻力"价位"，每个价位带 touch 次数(强度)。我们只需 _get_level_characteristics
(返回全部候选价位+强度)，自行按"当前价上方=阻力 / 下方=支撑"筛选。
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class LevelCharacteristics:
    mins_indexes: List[int]
    maxs_indexes: List[int]
    channel_mins_indexes: List[int]
    channel_maxs_indexes: List[int]
    channel_delta_spread_percentage: float
    channel_delta_spread_abs_value: float
    level_value: float


def _local_extrema_idx(arr, order, kind):
    """纯 numpy 实现 scipy.signal.argrelextrema 的等价逻辑（避免实盘环境缺 scipy）。
    kind='min' -> 局部极小值索引；'max' -> 局部极大值索引。
    等价于 argrelextrema(..., np.less_equal/greater_equal, order=order, mode='clip')：
    中心点 <= 窗内最小(min) / >= 窗内最大(max) 即判为极值。"""
    n = len(arr)
    if n < 2 * order + 1:
        return np.array([], dtype=int)
    try:
        from numpy.lib.stride_tricks import sliding_window_view
        w = sliding_window_view(arr, 2 * order + 1)          # (n-2*order, 2*order+1)
        sub = arr[order:n - order]
        if kind == "min":
            cond = sub <= w.min(axis=1)
        else:
            cond = sub >= w.max(axis=1)
        return np.nonzero(cond)[0] + order
    except Exception:
        idx = []
        for i in range(order, n - order):
            win = arr[i - order:i + order + 1]
            if kind == "min":
                if arr[i] <= win.min():
                    idx.append(i)
            else:
                if arr[i] >= win.max():
                    idx.append(i)
        return np.array(idx, dtype=int)


def _find_extremas(stock_prices, order):
    min_open_close = stock_prices[["close", "open"]].min(axis=1).to_numpy()
    max_open_close = stock_prices[["close", "open"]].max(axis=1).to_numpy()
    min_extrema = _local_extrema_idx(min_open_close, order, "min")
    max_extrema = _local_extrema_idx(max_open_close, order, "max")
    return (
        min_extrema,
        max_extrema,
        list(min_open_close[min_extrema]),
        list(max_open_close[max_extrema]),
    )


def _find_left_right_indexes_in_delta(sorted_array, delta_absolute):
    left_bound = 0
    right_bound = 0
    len_sorted_array = len(sorted_array)
    result = []
    for i in range(len_sorted_array):
        value = sorted_array[i]
        for j in range(left_bound, i + 1):
            if sorted_array[j] >= value - delta_absolute:
                break
        left_bound = j
        for k in range(right_bound, len_sorted_array):
            if sorted_array[k] > value + delta_absolute:
                break
        right_bound = k - 1
        result.append((left_bound, right_bound))
    return result


def _get_value_indexes(keys, values):
    dct = {}
    for i in range(len(keys)):
        if values[i] not in dct:
            dct[values[i]] = []
        dct[values[i]].append(keys[i])
    return dct


def _get_level_characteristics(stocks, order, percentage):
    mins, maxs, mins_vals, maxs_vals = _find_extremas(stocks, order)
    extremas = sorted(mins_vals + maxs_vals)
    len_extremas = len(extremas)
    max_value = stocks[["close", "low", "high", "open"]].max(axis=1).to_numpy().max()
    percent_abs_value = max_value * percentage / 100
    left_right_indexes = _find_left_right_indexes_in_delta(extremas, percent_abs_value)
    mins_values_indexes = _get_value_indexes(mins, mins_vals)
    maxs_values_indexes = _get_value_indexes(maxs, maxs_vals)
    result = []
    for i in range(len_extremas):
        level = extremas[i]
        left_index, right_index = left_right_indexes[i]
        mins_indexes_full = []
        maxs_indexes_full = []
        channel_mins_indexes_full = []
        channel_maxs_indexes_full = []
        for j in range(left_index, right_index + 1):
            extrema = extremas[j]
            if extrema in mins_values_indexes:
                if j == i:
                    mins_indexes_full += mins_values_indexes[extrema]
                else:
                    channel_mins_indexes_full += mins_values_indexes[extrema]
            if extrema in maxs_values_indexes:
                if j == i:
                    maxs_indexes_full += maxs_values_indexes[extrema]
                else:
                    channel_maxs_indexes_full += maxs_values_indexes[extrema]
        result.append(
            LevelCharacteristics(
                mins_indexes=mins_indexes_full,
                maxs_indexes=maxs_indexes_full,
                channel_mins_indexes=channel_mins_indexes_full,
                channel_maxs_indexes=channel_maxs_indexes_full,
                channel_delta_spread_percentage=percentage,
                channel_delta_spread_abs_value=percent_abs_value,
                level_value=level,
            )
        )
    return result


def _get_level_points(level):
    return (len(level.mins_indexes) + len(level.maxs_indexes)
            + len(level.channel_mins_indexes) + len(level.channel_maxs_indexes))


def strength(level):
    """价位的'touch'次数 = 强度，越高越可信。"""
    return _get_level_points(level)


def get_levels(df, order=3, merge_pct=2.0):
    """返回全部候选支撑/阻力价位(LevelCharacteristics 列表)。"""
    return _get_level_characteristics(df, order, merge_pct)
