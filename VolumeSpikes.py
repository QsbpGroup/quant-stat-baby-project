import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from high_low_xuejie_zuhui import df_init, find_hl_MACD_robust
from wave_price_change import wave_identify
from horizontal_area import _single_ha
import seaborn as sns


def Volume_Spikes_up(df, days):
    '''
    Output:
    ------------
    vs_up = [{'vs_date': '2019-01-01', 'vs_price': 10.0, 'max_gain': 10.00, 'max_loss': 5.00, 'max_swing': 15.00, 'drawdown': 8.00,
              'return': 10.0, 'upper_shadow': 5.0, 'avg': 5.0}, ...]
    'vs_date': 放量上影线日期
    'vs_price': 放量上影线当日收盘价
    'max_gain': 周期内的最大涨幅（以收益为正）
    'max_loss': 周期内的最大跌幅（以亏损为正）
    'max_swing': 周期内的最大振幅（用于判断是否是横盘）
    'drawdown': 周期内的回撤【（最大值-周期结束日收盘价）/最大值】
    'return': 周期内的收益【（周期第一天开盘价-周期结束日收盘价）/周期第一天开盘价】
    'upper_shadow': 放量上影线长度（用与实体长度比值衡量）
    'avg': 周期内收盘价均值与当日收盘价差距（百分比）
    'upper_shadow_ratio':上影线与实体长度比值

    Example:
    ------------
    >>> df = pd.read_csv('000001.SZ.csv')
    >>> df['TRADE_DT'] = pd.to_datetime(df['TRADE_DT'], format='%Y%m%d')
    >>> df = df[['TRADE_DT', 'S_DQ_OPEN', 'S_DQ_HIGH', 'S_DQ_LOW', 'S_DQ_CLOSE']]
    >>> vs_up_5 = Volume_Spikes_up(df, 5)    
    '''
    # 上影线定义：上影线长度至少2倍实体长度）
    vs_up = []
    for i in range(len(df)-days):
        upper_shadow_length = df['S_DQ_HIGH'][i] - \
            max(df['S_DQ_OPEN'][i], df['S_DQ_CLOSE'][i])
        lower_shadow_length = min(
            df['S_DQ_OPEN'][i], df['S_DQ_CLOSE'][i]) - df['S_DQ_LOW'][i]
        body_length = max(df['S_DQ_OPEN'][i], df['S_DQ_CLOSE'][i]) - \
            min(df['S_DQ_OPEN'][i], df['S_DQ_CLOSE'][i])
        if body_length != 0 and upper_shadow_length >= 2 * body_length:
            upper_shadow_ratio = upper_shadow_length/body_length
            max_price = df[i:i+days+1]['S_DQ_CLOSE'].max()
            min_price = df[i:i+days+1]['S_DQ_CLOSE'].min()
            max_gain = (max_price - df['S_DQ_CLOSE']
                        [i]) / df['S_DQ_CLOSE'][i] * 100
            max_loss = (df['S_DQ_CLOSE'][i] - min_price) / \
                df['S_DQ_CLOSE'][i] * 100
            max_swing = max_gain + max_loss
            drawdown = (max_price - df['S_DQ_CLOSE'][i+days]) / max_price * 100
            returns = (df['S_DQ_OPEN'][i+1] - df['S_DQ_CLOSE']
                       [i+days]) / df['S_DQ_OPEN'][i+1] * 100
            upper_shadow = upper_shadow_length / body_length
            avg = (df[i:i+days+1]['S_DQ_CLOSE'].mean() -
                   df['S_DQ_CLOSE'][i]) / df['S_DQ_CLOSE'][i] * 100
            vs_up.append(
                {'vs_date': df['TRADE_DT'][i], 'vs_price': df['S_DQ_CLOSE'][i], 'max_gain': max_gain, 'max_loss': max_loss,
                 'max_swing': max_swing, 'drawdown': drawdown, 'return': returns, 'upper_shadow': upper_shadow, 'avg': avg,
                 'upper_shadow_ratio': upper_shadow_ratio})
        elif body_length == 0 and upper_shadow_length > 0 and upper_shadow_length > lower_shadow_length:
            max_price = df[i:i+days+1]['S_DQ_CLOSE'].max()
            min_price = df[i:i+days+1]['S_DQ_CLOSE'].min()
            max_gain = (max_price - df['S_DQ_CLOSE']
                        [i]) / df['S_DQ_CLOSE'][i] * 100
            max_loss = (df['S_DQ_CLOSE'][i] - min_price) / \
                df['S_DQ_CLOSE'][i] * 100
            max_swing = max_gain + max_loss
            drawdown = (max_price - df['S_DQ_CLOSE'][i+days]) / max_price * 100
            returns = (df['S_DQ_OPEN'][i+1] - df['S_DQ_CLOSE']
                       [i+days]) / df['S_DQ_OPEN'][i+1] * 100
            upper_shadow = 'infinite'
            avg = (df[i:i+days+1]['S_DQ_CLOSE'].mean() -
                   df['S_DQ_CLOSE'][i]) / df['S_DQ_CLOSE'][i] * 100
            vs_up.append(
                {'vs_date': df['TRADE_DT'][i], 'vs_price': df['S_DQ_CLOSE'][i], 'max_gain': max_gain, 'max_loss': max_loss,
                 'max_swing': max_swing, 'drawdown': drawdown, 'return': returns, 'upper_shadow': upper_shadow, 'avg': avg,
                 'upper_shadow_ratio': 100})
    return pd.DataFrame(vs_up)


def Volume_Spikes_low(df, days):
    '''
    Output:
    ------------
    vs_low = [{'vs_date': '2019-01-01', 'vs_price': 10.0, 'max_gain': 10.00, 'max_loss': 5.00, 'max_swing': 15.00, 'drawdown': 8.00,
              'return': 10.0, 'lower_shadow': 5.0, 'avg': 5.0}, ...]
    'vs_date': 放量下影线日期
    'vs_price': 放量下影线当日收盘价
    'max_gain': 周期内的最大涨幅（以收益为正）
    'max_loss': 周期内的最大跌幅（以亏损为正）
    'max_swing': 周期内的最大振幅（用于判断是否是横盘）
    'drawdown': 周期内的回撤【（最大值-周期结束日收盘价）/最大值】
    'return': 周期内的收益【（周期第一天开盘价-周期结束日收盘价）/周期第一天开盘价】
    'lower_shadow': 放量下影线长度（用与实体长度比值衡量）
    'avg': 周期内收盘价均值与当日收盘价差距（百分比）
    'low_shadow_ratio':下影线与实体长度比值

    Example:
    ------------
    >>> df = pd.read_csv('000001.SZ.csv')
    >>> df['TRADE_DT'] = pd.to_datetime(df['TRADE_DT'], format='%Y%m%d')
    >>> df = df[['TRADE_DT', 'S_DQ_OPEN', 'S_DQ_HIGH', 'S_DQ_LOW', 'S_DQ_CLOSE']]
    >>> vs_low_5 = Volume_Spikes_low(df, 5)    
    '''
    # 下影线定义：下影线长度至少2倍实体长度）
    vs_low = []
    for i in range(len(df)-days):
        upper_shadow_length = df['S_DQ_HIGH'][i] - \
            max(df['S_DQ_OPEN'][i], df['S_DQ_CLOSE'][i])
        lower_shadow_length = min(
            df['S_DQ_OPEN'][i], df['S_DQ_CLOSE'][i]) - df['S_DQ_LOW'][i]
        body_length = max(df['S_DQ_OPEN'][i], df['S_DQ_CLOSE'][i]) - \
            min(df['S_DQ_OPEN'][i], df['S_DQ_CLOSE'][i])
        if body_length != 0 and lower_shadow_length >= 2 * body_length:
            lower_shadow_ratio = lower_shadow_length/body_length
            max_price = df[i:i+days+1]['S_DQ_CLOSE'].max()
            min_price = df[i:i+days+1]['S_DQ_CLOSE'].min()
            max_gain = (max_price - df['S_DQ_CLOSE']
                        [i]) / df['S_DQ_CLOSE'][i] * 100
            max_loss = (df['S_DQ_CLOSE'][i] - min_price) / \
                df['S_DQ_CLOSE'][i] * 100
            max_swing = max_gain + max_loss
            drawdown = (max_price - df['S_DQ_CLOSE'][i+days]) / max_price * 100
            returns = (df['S_DQ_OPEN'][i+1] - df['S_DQ_CLOSE']
                       [i+days]) / df['S_DQ_OPEN'][i+1] * 100
            lower_shadow = lower_shadow_length / body_length
            avg = (df[i:i+days+1]['S_DQ_CLOSE'].mean() -
                   df['S_DQ_CLOSE'][i]) / df['S_DQ_CLOSE'][i] * 100
            vs_low.append(
                {'vs_date': df['TRADE_DT'][i], 'vs_price': df['S_DQ_CLOSE'][i], 'max_gain': max_gain, 'max_loss': max_loss,
                 'max_swing': max_swing, 'drawdown': drawdown, 'return': returns, 'lower_shadow': lower_shadow, 'avg': avg,
                 'lower_shadow_ratio': lower_shadow_ratio})
        elif body_length == 0 and lower_shadow_length > 0 and upper_shadow_length < lower_shadow_length:
            max_price = df[i:i+days+1]['S_DQ_CLOSE'].max()
            min_price = df[i:i+days+1]['S_DQ_CLOSE'].min()
            max_gain = (max_price - df['S_DQ_CLOSE']
                        [i]) / df['S_DQ_CLOSE'][i] * 100
            max_loss = (df['S_DQ_CLOSE'][i] - min_price) / \
                df['S_DQ_CLOSE'][i] * 100
            max_swing = max_gain + max_loss
            drawdown = (max_price - df['S_DQ_CLOSE'][i+days]) / max_price * 100
            returns = (df['S_DQ_OPEN'][i+1] - df['S_DQ_CLOSE']
                       [i+days]) / df['S_DQ_OPEN'][i+1] * 100
            lower_shadow = 'infinite'
            avg = (df[i:i+days+1]['S_DQ_CLOSE'].mean() -
                   df['S_DQ_CLOSE'][i]) / df['S_DQ_CLOSE'][i] * 100
            vs_low.append(
                {'vs_date': df['TRADE_DT'][i], 'vs_price': df['S_DQ_CLOSE'][i], 'max_gain': max_gain, 'max_loss': max_loss,
                 'max_swing': max_swing, 'drawdown': drawdown, 'return': returns, 'lower_shadow': lower_shadow, 'avg': avg,
                 'lower_shadow_ratio': 100})
    return pd.DataFrame(vs_low)


def negative_ratio_result(df, bins, result_up, name):
    try:
        # Attempt to perform the cut operation
        df["upper_shadow_ratio_bin"] = pd.cut(df["upper_shadow_ratio"], bins)
    except ValueError:
        # Handle the exception by setting "upper_shadow_ratio" to 0
        df["upper_shadow_ratio"] = 0
        df["upper_shadow_ratio_bin"] = pd.cut(df["upper_shadow_ratio"], bins)

    # 进行分箱操作
    df["upper_shadow_ratio_bin"] = pd.cut(df["upper_shadow_ratio"], bins)
    # 计算每个分箱区间内"return"为负的比例
    result_group = df.groupby("upper_shadow_ratio_bin")["return"].apply(
        lambda x: (x < 0).mean()).reset_index(name="negative_return_ratio")
    # 该股回撤均值
    drawdown_mean = df['drawdown'].mean()
    # 该股出现上影线后下跌的概率
    negative_return_prob = (df["return"] < 0).mean()
    result_up.append({'Stock_name': name,
                      'drawdown_mean': drawdown_mean,
                      '跌的概率': negative_return_prob,
                      'upper_shadow_ratio（2，5]': result_group['negative_return_ratio'][0],
                      'upper_shadow_ratio（5，10]': result_group['negative_return_ratio'][1],
                      'upper_shadow_ratio（10，20]': result_group['negative_return_ratio'][2],
                      'upper_shadow_ratio（20，100]': result_group['negative_return_ratio'][3]})
    return(result_up)


def positive_ratio_result(df, bins, result_up, name):
    try:
        # Attempt to perform the cut operation
        df["lower_shadow_ratio_bin"] = pd.cut(df["lower_shadow_ratio"], bins)
    except ValueError:
        # Handle the exception by setting "upper_shadow_ratio" to 0
        df["lower_shadow_ratio"] = 0
        df["lower_shadow_ratio_bin"] = pd.cut(df["lower_shadow_ratio"], bins)

    # 进行分箱操作
    df["lower_shadow_ratio_bin"] = pd.cut(df["lower_shadow_ratio"], bins)
    # 计算每个分箱区间内"return"为正的比例
    result_group = df.groupby("lower_shadow_ratio_bin")["return"].apply(
        lambda x: (x > 0).mean()).reset_index(name="positive_return_ratio")
    # 该股回撤均值
    drawdown_mean = df['drawdown'].mean()
    # 该股出现下影线后上涨的概率
    positive_return_prob = (df["return"] > 0).mean()
    result_up.append({'Stock_name': name,
                      'drawdown_mean': drawdown_mean,
                      '涨的概率': positive_return_prob,
                      'lower_shadow_ratio（2，5]': result_group['positive_return_ratio'][0],
                      'lower_shadow_ratio（5，10]': result_group['positive_return_ratio'][1],
                      'lower_shadow_ratio（10，20]': result_group['positive_return_ratio'][2],
                      'lower_shadow_ratio（20，100]': result_group['positive_return_ratio'][3]})
    return(result_up)


def save_result_to_csv(result_data, result_filename, results_directory):
    # 将结果保存到CSV文件（指定编码为UTF-8）
    result_file_path = os.path.join(results_directory, result_filename)
    result_df = pd.DataFrame(result_data)
    result_df.to_csv(result_file_path, index=False, encoding='utf-8-sig')


def wave_identify_with_high_ha(filename='000001.SZ.csv', alpha=0.05, min_ha_length=4, max_ha_length=10):
    '''
    识别高位横盘并返回waves with high sideways，在waves所有edo_rise日期附近找到横盘区间 (若有)。
    具体来说，我们找到以每两个edo_fall之间的区域为窗口，在窗口内以edo_rise为中心点，利用_single_ha识别横盘区间

    Input:
    ------------
    filename : str
        why use it if you have no idea what the filename is? in case u do not know, it is the sotck code that ends with .csv

    alpha : float
        quantile number used to drop some points from original waves, the larger alpha is, the more hl points will be dropped.

    min_ha_length : int
        the minimum length of horizontal area

    max_ha_length : int
        the maximum length of horizontal area

    Output:
    ------------
    waves_with_high_ha: DataFrame
        columns=['date', 'price', 'type'], type can be 'edo_fall', 'edo_rise', 'start_of_ha', 'end_of_ha'.
    '''
    df = df_init(filename)
    df.columns = ['date', 'close']
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    waves_with_high_ha = wave_identify(filename, alpha)
    waves_with_high_ha['date'] = pd.to_datetime(
        waves_with_high_ha['date'], format='%Y-%m-%d')
    if waves_with_high_ha.empty:
        return waves_with_high_ha
    # make sure every edo_rise has a edo_fall before it and after it
    if waves_with_high_ha['type'].iloc[0] == 'edo_rise':
        waves_with_high_ha = waves_with_high_ha.iloc[1:]
    if waves_with_high_ha.empty:
        return waves_with_high_ha
    if waves_with_high_ha['type'].iloc[-1] == 'edo_rise':
        waves_with_high_ha = waves_with_high_ha.iloc[:-1]
    # edo_rise need to drop
    edo_rise_drop_list = []
    for i in range(int((len(waves_with_high_ha)-1)/2)):
        tmp_edo_fall_date = waves_with_high_ha['date'].iloc[2*i]
        tmp_edo_rise_date = waves_with_high_ha['date'].iloc[2*i+1]
        tmp_edo_fall_date_after = waves_with_high_ha['date'].iloc[2*i+2]
        df_tmp = df[(df['date'] >= tmp_edo_fall_date) & (
            df['date'] <= tmp_edo_fall_date_after)]
        tmp_edo_rise_data = waves_with_high_ha[waves_with_high_ha['date'] == tmp_edo_rise_date]
        tmp_edo_rise_data_point = tmp_edo_rise_data.iloc[0]
        df_tmp = df_tmp.reset_index(drop=True)
        if df_tmp.empty or tmp_edo_rise_data.empty:
            continue
        tmp_ha = _single_ha(
            df_tmp, tmp_edo_rise_data_point, alpha)
        if not tmp_ha.empty:
            tmp_start_of_ha_date = tmp_ha['start_date'].iloc[0]
            tmp_end_of_ha_date = tmp_ha['end_date'].iloc[0]
            # cal the days difference between tmp_start_of_ha_date and tmp_end_of_ha_date
            tmp_start_of_ha_date_index = df_tmp[df_tmp['date']
                                                == tmp_start_of_ha_date].index[0]
            tmp_end_of_ha_date_index = df_tmp[df_tmp['date']
                                                == tmp_end_of_ha_date].index[0]
            tmp_days_diff = tmp_end_of_ha_date_index - \
                tmp_start_of_ha_date_index
            if tmp_days_diff < min_ha_length or tmp_days_diff > max_ha_length:
                continue
            edo_rise_drop_list.append(tmp_edo_rise_date)
            # find their price from df_tmp
            tmp_start_ha_price = df_tmp[df_tmp['date']
                                        == tmp_start_of_ha_date]['close'].iloc[0]
            tmp_end_ha_price = df_tmp[df_tmp['date']
                                      == tmp_end_of_ha_date]['close'].iloc[0]
            waves_with_high_ha = waves_with_high_ha.append({'date': tmp_start_of_ha_date, 'price': tmp_start_ha_price,
                                                            'type': 'start_of_ha'}, ignore_index=True)
            waves_with_high_ha = waves_with_high_ha.append({'date': tmp_end_of_ha_date, 'price': tmp_end_ha_price,
                                                            'type': 'end_of_ha'}, ignore_index=True)
    # 去掉waves-with-high-ha中date在edo_rise_drop_list中且type为edo_rise的行
    waves_with_high_ha = waves_with_high_ha[~((waves_with_high_ha['date'].isin(
        edo_rise_drop_list)) & (waves_with_high_ha['type'] == 'edo_rise'))]
    
    # sort by date and reset index
    waves_with_high_ha = waves_with_high_ha.sort_values(by='date')
    waves_with_high_ha = waves_with_high_ha.reset_index(drop=True)

    return waves_with_high_ha


def draw_waves_with_high_ha(df, real_waves, fig_start_date, fig_end_date):
    """
    画出fig_start_date, fig_end_date之间的crash情况
    """
    # 初始化df_cache，避免浅拷贝导致的原始df被修改
    df_cache = df.copy()
    highs, lows = find_hl_MACD_robust(df_cache, draw=False)
    df_cache.columns = ['date', 'price']
    highs = pd.DataFrame(highs)
    lows = pd.DataFrame(lows)
    # 截取需要的数据
    df_cache = df_cache[(df_cache['date'] >= fig_start_date)
                        & (df_cache['date'] <= fig_end_date)]
    highs = highs[(highs['high_date'] >= fig_start_date)
                  & (highs['high_date'] <= fig_end_date)]
    lows = lows[(lows['low_date'] >= fig_start_date)
                & (lows['low_date'] <= fig_end_date)]
    # 筛选crash中的[start_date, end_date]，确保在df_cache中
    real_waves = real_waves[(real_waves['date'] >= fig_start_date)
                            & (real_waves['date'] <= fig_end_date)]

    plt.rcParams['figure.figsize'] = [10, 5]
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

    # 绘制df的折线图, 颜色浅蓝色
    plt.plot(df_cache['date'], df_cache['price'], label='stock price')
    # 将highs中的高点绘制成红色的星星
    plt.scatter(highs['high_date'], highs['high_price'],
                color='red', marker='*', s=80, label='high points')
    # 将lows中的低点绘制成绿色的星星
    plt.scatter(lows['low_date'], lows['low_price'],
                color='green', marker='*', s=80, label='low points')

    for i in range(len(real_waves)-1):
        if real_waves.iloc[i]['type'] == 'edo_fall':
            plt.axvspan(real_waves.iloc[i]['date'], real_waves.iloc[i+1]['date'],
                        facecolor='red', alpha=0.15, label='rising waves')
        elif real_waves.iloc[i]['type'] == 'start_of_ha':
            plt.axvspan(real_waves.iloc[i]['date'], real_waves.iloc[i+1]['date'],
                        facecolor='yellow', alpha=0.15, label='high sideways')
        else:
            plt.axvspan(real_waves.iloc[i]['date'], real_waves.iloc[i+1]['date'],
                        facecolor='green', alpha=0.15, label='falling waves')
    if real_waves.iloc[0]['type'] == 'edo_fall':
        plt.axvspan(fig_start_date, real_waves.iloc[0]['date'],
                    facecolor='green', alpha=0.15, label='falling waves')
    elif real_waves.iloc[0]['type'] == 'end_of_ha':
        plt.axvspan(fig_start_date, real_waves.iloc[0]['date'],
                    facecolor='yellow', alpha=0.15, label='high sideways')
    else:
        plt.axvspan(fig_start_date, real_waves.iloc[0]['date'],
                    facecolor='red', alpha=0.15, label='rising waves')
    if real_waves.iloc[-1]['type'] == 'edo_fall':
        plt.axvspan(real_waves.iloc[-1]['date'], fig_end_date,
                    facecolor='red', alpha=0.15, label='rising waves')
    elif real_waves.iloc[-1]['type'] == 'start_of_ha':
        plt.axvspan(real_waves.iloc[-1]['date'], fig_end_date,
                    facecolor='yellow', alpha=0.15, label='high sideways')
    else:
        plt.axvspan(real_waves.iloc[-1]['date'], fig_end_date,
                    facecolor='green', alpha=0.15, label='falling waves')

    # 设置正确的label和title
    plt.xlabel('date')
    plt.ylabel('price')
    plt.title('Waves')
    handles, labels = plt.gca().get_legend_handles_labels()
    # legand去重
    set(labels)
    new_labels, new_handles = ['stock price', 'high points',
                               'low points', 'rising waves', 'falling waves', 'high sideways'], []
    for i in range(len(new_labels)):
        for j in range(len(labels)):
            if new_labels[i] == labels[j]:
                new_handles.append(handles[j])
                break
    plt.legend(new_handles, new_labels)
    plt.show()
    plt.close()


if __name__ == '__main__':
    # 获取股票数据文件夹的路径
    #data_directory = '/Users/zhaochenxi/Desktop/quant/data_csv1'
    data_directory = '/Users/zhaochenxi/Desktop/quant/data_csv_distinct'
    # 设置结果保存文件夹的路径
    results_directory_up_1 = r'/Users/zhaochenxi/Desktop/quant/Volume_Spikes_result_up_1'
    results_directory_up_5 = r'/Users/zhaochenxi/Desktop/quant/Volume_Spikes_result_up_5'
    results_directory_up_22 = r'/Users/zhaochenxi/Desktop/quant/Volume_Spikes_result_up_22'

    results_directory_low_1 = r'/Users/zhaochenxi/Desktop/quant/Volume_Spikes_result_low_1'
    results_directory_low_5 = r'/Users/zhaochenxi/Desktop/quant/Volume_Spikes_result_low_5'
    results_directory_low_22 = r'/Users/zhaochenxi/Desktop/quant/Volume_Spikes_result_low_22'

    # 创建结果保存文件夹
    if not os.path.exists(results_directory_up_1):
        os.makedirs(results_directory_up_1)
    if not os.path.exists(results_directory_up_5):
        os.makedirs(results_directory_up_5)
    if not os.path.exists(results_directory_up_22):
        os.makedirs(results_directory_up_22)

    if not os.path.exists(results_directory_low_1):
        os.makedirs(results_directory_low_1)
    if not os.path.exists(results_directory_low_5):
        os.makedirs(results_directory_low_5)
    if not os.path.exists(results_directory_low_22):
        os.makedirs(results_directory_low_22)

    result_up_1 = []
    result_up_5 = []
    result_up_22 = []
    result_low_1 = []
    result_low_5 = []
    result_low_22 = []

    # 遍历目录中的所有文件
    for filename in tqdm(os.listdir(data_directory)):

        if filename.endswith('.csv'):
            try:

                file_path = os.path.join(data_directory, filename)
                stock_name = filename[:-4]
                # 读取数据并创建DataFrame
                data = pd.read_csv(file_path)
                df = pd.DataFrame(data)
                # 转换日期列为日期时间类型
                df['TRADE_DT'] = pd.to_datetime(df['TRADE_DT'], format='%Y%m%d')
                df = df[['TRADE_DT', 'S_DQ_OPEN',
                        'S_DQ_HIGH', 'S_DQ_LOW', 'S_DQ_CLOSE']]
                # 定义分箱区间
                bins = [2, 5, 10, 20, 100]

                vs_up_1 = Volume_Spikes_up(df, 1)
                output_filename_up_1 = os.path.splitext(
                    filename)[0] + 'Volume_Spikes_up_1.csv'
                output_file_path_up_1 = os.path.join(
                    results_directory_up_1, output_filename_up_1)
                vs_up_1.to_csv(output_file_path_up_1,
                            index=False, encoding='utf-8-sig')
                resut_up_1 = negative_ratio_result(
                    vs_up_1, bins, result_up_1, stock_name)

                vs_up_5 = Volume_Spikes_up(df, 5)
                output_filename_up_5 = os.path.splitext(
                    filename)[0] + 'Volume_Spikes_up_5.csv'
                output_file_path_up_5 = os.path.join(
                    results_directory_up_5, output_filename_up_5)
                vs_up_5.to_csv(output_file_path_up_5,
                            index=False, encoding='utf-8-sig')
                resut_up_5 = negative_ratio_result(
                    vs_up_5, bins, result_up_5, stock_name)

                vs_up_22 = Volume_Spikes_up(df, 22)
                output_filename_up_22 = os.path.splitext(
                    filename)[0] + 'Volume_Spikes_up_22.csv'
                output_file_path_up_22 = os.path.join(
                    results_directory_up_22, output_filename_up_22)
                vs_up_22.to_csv(output_file_path_up_22,
                                index=False, encoding='utf-8-sig')
                resut_up_22 = negative_ratio_result(
                    vs_up_22, bins, result_up_22, stock_name)

                vs_low_1 = Volume_Spikes_low(df, 1)
                output_filename_low_1 = os.path.splitext(
                    filename)[0] + 'Volume_Spikes_low_1.csv'
                output_file_path_low_1 = os.path.join(
                    results_directory_low_1, output_filename_low_1)
                vs_low_1.to_csv(output_file_path_low_1,
                                index=False, encoding='utf-8-sig')
                resut_low_1 = positive_ratio_result(
                    vs_low_1, bins, result_low_1, stock_name)

                vs_low_5 = Volume_Spikes_low(df, 5)
                output_filename_low_5 = os.path.splitext(
                    filename)[0] + 'Volume_Spikes_low_5.csv'
                output_file_path_low_5 = os.path.join(
                    results_directory_low_5, output_filename_low_5)
                vs_low_5.to_csv(output_file_path_low_5,
                                index=False, encoding='utf-8-sig')
                resut_low_5 = positive_ratio_result(
                    vs_low_5, bins, result_low_5, stock_name)

                vs_low_22 = Volume_Spikes_low(df, 22)
                output_filename_low_22 = os.path.splitext(
                    filename)[0] + 'Volume_Spikes_low_22.csv'
                output_file_path_low_22 = os.path.join(
                    results_directory_low_22, output_filename_low_22)
                vs_low_22.to_csv(output_file_path_low_22,
                                index=False, encoding='utf-8-sig')
                resut_low_22 = positive_ratio_result(
                    vs_low_22, bins, result_low_22, stock_name)

            except Exception as e:
                # Handle the exception (print an error message, log it, etc.)
                print(f"Error processing {filename}: {e}")
                continue  # Skip to the next file

    # 将结果保存到CSV文件（指定编码为UTF-8）
    save_result_to_csv(resut_up_1, 'result_up_1.csv', results_directory_up_1)
    save_result_to_csv(resut_up_5, 'result_up_5.csv', results_directory_up_5)
    save_result_to_csv(resut_up_22, 'result_up_22.csv', results_directory_up_22)
    save_result_to_csv(resut_low_1, 'result_low_1.csv', results_directory_low_1)
    save_result_to_csv(resut_low_5, 'result_low_5.csv', results_directory_low_5)
    save_result_to_csv(resut_low_22, 'result_low_22.csv', results_directory_low_22)

    result_low_1 = pd.read_csv(
        '/Users/zhaochenxi/Desktop/quant/Volume_Spikes_result_low_1/result_low_1.csv')

    # 处理缺失值，使用 dropna() 方法来删除包含 NaN 值的行
    result_low_1.dropna(subset=["lower_shadow_ratio（2，5]", "lower_shadow_ratio（5，10]",
                                "lower_shadow_ratio（10，20]", "lower_shadow_ratio（20，100]"], inplace=True)

    # 提取lower_shadow_ratio列数据
    lower_shadow_ratio_1 = result_low_1["lower_shadow_ratio（2，5]"]
    lower_shadow_ratio_5 = result_low_1['lower_shadow_ratio（5，10]']
    lower_shadow_ratio_10 = result_low_1['lower_shadow_ratio（10，20]']
    lower_shadow_ratio_20 = result_low_1['lower_shadow_ratio（20，100]']

    # 绘制箱线图
    data_to_plot1 = [lower_shadow_ratio_1, lower_shadow_ratio_5,
                    lower_shadow_ratio_10, lower_shadow_ratio_20]
    labels = ['(2,5]', '(5,10]', '(10,20]', '(20,100]']

    plt.boxplot(data_to_plot1, labels=labels)
    plt.title('Lower Shadow Ratio Box Plot for Different Ratio')
    plt.ylabel('The percentage of the rise')
    plt.xlabel('Ratio')
    plt.show()
