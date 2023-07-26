import pandas as pd
from high_low_xuejie_zuhui import find_high_low


def waves_price_change(df):
    """
    计算了df中每个波段的涨跌幅
    """
    df_cache = df.copy()
    df_cache.reset_index(drop=True, inplace=True)
    highs, lows = find_high_low(df_cache, draw=False)
    high_price = [item['high_price'] for item in highs]
    low_price = [item['low_price'] for item in lows]
    # 下跌的高点序列突然上升，说明下跌结束
    # 上升的低点序列突然下降，说明上涨结束
    high_chang_point_index = []
    low_chang_point_index = []
    for i in range(1, len(high_price)-1):
        if high_price[i+1] > high_price[i] and high_price[i-1] > high_price[i]:
            high_chang_point_index.append(i)
        if low_price[i+1] < low_price[i] and low_price[i-1] < low_price[i]:
            low_chang_point_index.append(i)
    # 从highs, lows中提取第index个高点/低点并存进同一个dataframe
    df_highs = pd.DataFrame(highs)
    df_lows = pd.DataFrame(lows)
    df_highs.reset_index(drop=True, inplace=True)
    df_lows.reset_index(drop=True, inplace=True)
    df_highs = df_highs.iloc[high_chang_point_index]
    df_lows = df_lows.iloc[low_chang_point_index]
    df_highs = df_highs[['index', 'high_date', 'high_price']]
    df_lows = df_lows[['index', 'low_date', 'low_price']]
    df_highs.columns = ['index', 'date', 'price']
    df_lows.columns = ['index', 'date', 'price']
    df_highs['type'] = 'high'
    df_lows['type'] = 'low'
    df_highs_lows = pd.concat([df_highs, df_lows])
    df_highs_lows.sort_values(by=['index'], inplace=True)
    df_highs_lows.reset_index(drop=True, inplace=True)
    # 计算每个波段的涨跌幅
    df_highs_lows['price_change'] = df_highs_lows['price'].pct_change()
    df_highs_lows['price_change'] = df_highs_lows['price_change'].shift(-1)
    df_highs_lows['days'] = df_highs_lows['date'].diff()
    df_highs_lows['days'] = df_highs_lows['days'].shift(-1)
    df_highs_lows['days'] = df_highs_lows['days'].dt.days
    df_highs_lows['price_change_per_day'] = df_highs_lows['price_change'] / \
        df_highs_lows['days']
    df_highs_lows.dropna(inplace=True)
    return df_highs_lows
