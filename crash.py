from pandas import DataFrame
import matplotlib.pyplot as plt
from high_low_xuejie_zuhui import find_hl_MACD_robust


def crash(df, threshold=0.4):
    """
    识别股票在三轮下跌(要求每次反弹不超过上次下跌的50%)内跌幅超过threshold的点的情况

    Parameters
    ----------
    df : DataFrame
        股票数据，列：['日期', '价格']

    threshold : float
        跌幅阈值

    Returns
    -------
    result = DataFrame
        跌幅超过threshold的点, 列：['start_date', 'end_date', 'type'], 
        type=1表示单次暴跌, type=2表示双次内暴跌, type=3表示三次内暴跌
    """
    # 找出高低点
    highs, lows = find_hl_MACD_robust(df, draw=False)
    if len(highs) == 0 or len(lows) == 0:
        return DataFrame()
    # 计算每个低点到下一个高点的跌幅
    result = DataFrame(columns=['start_date', 'end_date', 'type'])
    # 如果第一个低点比第一个高点靠前，删除第一个低点
    if lows[0]['low_date'] < highs[0]['high_date']:
        lows = lows[1:]
    max_len = min(len(highs), len(lows))
    for i in range(max_len):
        high = highs[i]
        low = lows[i]
        if len(result) > 0 and (high['high_date'] < result['end_date'].max() or low['low_date'] < result['end_date'].max()):
            continue
        # 计算跌幅
        crash = (high['high_price'] - low['low_price']) / high['high_price']
        # 如果跌幅超过阈值，记录
        if crash > threshold:
            result = result.append(
                {'start_date': high['high_date'], 'end_date': low['low_date'], 'type': 1}, ignore_index=True)
        elif i < max_len - 1:
            # 如果跌幅没有超过阈值and下一个高点不回升超过50%，但是这个高点到下一个低点的跌幅超过阈值，记录
            high_shake = 0
            if high['high_price'] > low['low_price']:
                high_shake = (high['high_price'] - highs[i + 1]['high_price']
                                        ) / (high['high_price'] - low['low_price'])
            crash = (high['high_price'] - lows[i+1]
                     ['low_price']) / high['high_price']
            if high_shake > 0.5 and crash > threshold:
                result = result.append(
                    {'start_date': high['high_date'], 'end_date': lows[i+1]['low_date'], 'type': 2}, ignore_index=True)
            elif i < max_len - 2:
                # 如果跌幅没有超过阈值and下两个高点不回升超过50%，但是这个高点到下两个低点的跌幅超过阈值，记录
                high_shake_twice = 0
                if highs[i+1]['high_price'] > lows[i+1]['low_price']:
                    high_shake_twice = (highs[i + 1]['high_price'] - highs[i + 2]['high_price']
                                        ) / (highs[i+1]['high_price'] - lows[i+1]['low_price'])
                crash_2 = (high['high_price'] - lows[i+2]
                            ['low_price']) / high['high_price']
                if high_shake > 0.5 and high_shake_twice > 0.5 and crash_2 > threshold:
                    result = result.append(
                        {'start_date': high['high_date'], 'end_date': lows[i+2]['low_date'], 'type': 3}, ignore_index=True)

    return result


def draw_crash(df, crash, fig_start_date, fig_end_date):
    """
    画出fig_start_date, fig_end_date之间的crash情况
    """
    # 初始化df_cache，避免浅拷贝导致的原始df被修改
    df_cache = df.copy()
    highs, lows = find_hl_MACD_robust(df_cache, draw=False)
    df_cache.columns = ['date', 'price']
    highs = DataFrame(highs)
    lows = DataFrame(lows)
    # 截取需要的数据
    df_cache = df_cache[(df_cache['date'] >= fig_start_date)
                        & (df_cache['date'] <= fig_end_date)]
    highs = highs[(highs['high_date'] >= fig_start_date)
                  & (highs['high_date'] <= fig_end_date)]
    lows = lows[(lows['low_date'] >= fig_start_date)
                & (lows['low_date'] <= fig_end_date)]
    # 筛选crash中的[start_date, end_date]，确保在df_cache中
    crash = crash[(crash['start_date'] >= fig_start_date)
                  & (crash['start_date'] <= fig_end_date)]

    plt.rcParams['figure.figsize'] = [10, 5]
    plt.rcParams['font.sans-serif']=['Arial Unicode MS']
    
    # 绘制df的折线图, 颜色浅蓝色
    plt.plot(df_cache['date'], df_cache['price'])
    # 将highs中的高点绘制成红色的星星
    plt.scatter(highs['high_date'], highs['high_price'],
                color='red', marker='*', s=80)
    # 将lows中的低点绘制成绿色的星星
    plt.scatter(lows['low_date'], lows['low_price'],
                color='green', marker='*', s=80)
    # 将crash中的[start_date, end_date]绘制成橙色区间
    for index, row in crash.iterrows():
        if row['type'] == 1:
            plt.axvspan(row['start_date'], row['end_date'],
                        color='orange', alpha=0.19)
        elif row['type'] == 2:
            plt.axvspan(row['start_date'], row['end_date'],
                        color='orange', alpha=0.25)
        elif row['type'] == 3:
            plt.axvspan(row['start_date'], row['end_date'],
                        color='orange', alpha=0.45)
    # 设置正确的label和title
    plt.xlabel('date')
    plt.ylabel('price')
    plt.title('crash')
    # 利用handle设置和图像中形状格式一致的legand
    handle = [plt.plot([], [], color='orange', alpha=0.15)[0], plt.plot(
        [], [], color='orange', alpha=0.3)[0], plt.plot([], [], color='orange', alpha=0.5)[0]]
    label = ['单次暴跌', '双次内暴跌', '三次内暴跌']
    plt.legend(handle, label)
    plt.show()
    plt.close()
