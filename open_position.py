from crash import crash
import pandas as pd
from high_low_xuejie_zuhui import find_hl_MACD_robust


def open_position(df, threshold=0.4, len_of_sideway=30, gap=30):
    """
    计算开仓点，算法逻辑如下：
    1. 找出在三轮下跌(要求每次反弹不超过上次下跌的50%)内跌幅超过threshold的点
    2. 在这些点中, 找出在随后出现横盘1-2个月的点, 这些点为开仓点
    3. 确认下跌结束点到买入点之间不存在新的下跌区间

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
    if len(crash_info) == 0:
        return result
    crash_end_dates = crash_info['end_date'].tolist()
    # highs, lows = find_hl_MACD_robust(df_cache, draw=False)

    # 对于从每个crash_end_date开始的(len_of_sideway + gap)天利用长度为len_of_sideway的滑动窗口计算:
    # 1. 滑动窗口内的最大值crt_max
    # 2. 滑动窗口内的最小值crt_min
    # 3. crash_end_date到滑动窗口末尾的最大值ttl_max
    # 4. crash_end_date到滑动窗口末尾的最小值ttl_min
    # 如果crt_max <= ttl_max*1.1 或者 crt_min >= ttl_min*0.9， 则记录当前滑动窗口末尾的日期为买入日期
    for crash_end_date in crash_end_dates:
        temp_start_date = crash_end_date
        temp_end_date = crash_end_date + \
            pd.Timedelta(days=len_of_sideway + gap)
        df_temp = df_cache[(df_cache['date'] >= temp_start_date)
                           & (df_cache['date'] <= temp_end_date)]
        df_temp.reset_index(drop=True, inplace=True)
        for i in range(gap):
            crt_end_date = crash_end_date+pd.Timedelta(days=len_of_sideway)
            # find the index of the latest line whose date is earlier or equal to than crt_end_date
            crt_end_index = df_temp[df_temp['date'] <= crt_end_date].index.tolist()[-1]
            crt_max = df_temp.iloc[i+1:crt_end_index+1]['price'].max()
            crt_min = df_temp.iloc[i+1:crt_end_index+1]['price'].min()
            ttl_max = df_temp.iloc[0:crt_end_index+1]['price'].max()
            ttl_min = df_temp.iloc[0:crt_end_index+1]['price'].min()
            if crt_max <= ttl_max*1.1 or crt_min >= ttl_min*0.9:
                result.append([crash_end_date, crash_end_date+pd.Timedelta(days=crt_end_index)])
                break

    # 若result每个区间中包含任一crash_end_date，去除这个区间
    for i in range(len(result)):
        for j in range(len(crash_end_dates)):
            if result[i][0] < crash_end_dates[j] <= result[i][1]:
                result[i] = 0
                break
    result = [x for x in result if x != 0]

    # 计算result中的日期
    for i in range(len(result)):
        result[i] = result[i][1]

    # 去除重复的开仓点
    if len(result) > 0:
        result = list(set(result))
        result.sort()

    return result
