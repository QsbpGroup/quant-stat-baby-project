from crash import crash
import pandas as pd
from horizontal_area import find_horizontal_area
from high_low_xuejie_zuhui import find_high_low


def open_position(df, threshold=0.4, len_of_sideway=30, gap=30):
    """
    计算开仓点，算法逻辑如下：
    1. 找出在三轮下跌(要求每次反弹不超过上次下跌的50%)内跌幅超过threshold的点
    2. 在这些点中, 找出在随后出现横盘1-2个月的点, 这些点为开仓点

    Parameters
    ----------
    df : DataFrame
        股票数据，列：['date', 'price']

    threshold : float
        跌幅阈值

    len_of_sideway : int
        横盘的长度 (days)

    gap : int
        开仓点与横盘的间隔 (days)

    Returns
    -------
    result : list
        开仓点
    """
    # 忽略链式赋值的警告
    pd.options.mode.chained_assignment = None
    result = []
    df_cache = df.copy()
    df_cache.reset_index(drop=True, inplace=True)
    crash_info = crash(df_cache, threshold=threshold)
    df_cache.columns = ['date', 'price']
    crash_end_dates = crash_info['end_date'].tolist()
    highs, lows = find_high_low(df_cache, draw=False)

    # 对于从每个crash_end_date开始的(len_of_sideway + gap)天利用长度为len_of_sideway的滑动窗口计算，
    for crash_end_date in crash_end_dates:
        temp_start_date = crash_end_date - pd.Timedelta(days=len_of_sideway-2)
        temp_end_date = crash_end_date + \
            pd.Timedelta(days=len_of_sideway*2 + gap)
        df_temp = df_cache[(df_cache['date'] >= crash_end_date)
                           & (df_cache['date'] <= temp_end_date)]
        judge = find_horizontal_area(df_temp, highs, lows, gamma=0.1, max_len_of_window=60, min_len_of_window=len_of_sideway, only_past=True)
        if True:
            result.append(crash_end_date)
    return result
