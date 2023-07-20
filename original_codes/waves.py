import os
import pandas as pd
#import matplotlib.pyplot as plt
import talib
import numpy as np
#from matplotlib.legend_handler import HandlerLine2D, HandlerPathCollection
#import matplotlib.lines as mlines
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
'''
# 获取股票数据文件夹的路径
data_directory = '/Users/zhaochenxi/Desktop/quant/data_csv_distinct'
# 设置结果保存文件夹的路径
results_directory = '/Users/zhaochenxi/Desktop/quant/data_csv_distinct_results'
'''
# 获取股票数据文件夹的路径
data_directory = '/Users/zhaochenxi/Desktop/quant/data_csv1'
# 设置结果保存文件夹的路径
results_directory = '/Users/zhaochenxi/Desktop/quant/data_csv_results1'

# 创建结果保存文件夹
if not os.path.exists(results_directory):
    os.makedirs(results_directory)

result_data_waves = []
# 遍历目录中的所有文件
for filename in tqdm(os.listdir(data_directory)):
    if filename.endswith('.csv'):
        # 构建完整的文件路径
        file_path = os.path.join(data_directory, filename)
        stock_name = filename[:-4]

        # 读取数据并创建DataFrame
        data = pd.read_csv(file_path)
        df = pd.DataFrame(data)

        # 转换日期列为日期时间类型
        df['TRADE_DT'] = pd.to_datetime(df['TRADE_DT'], format='%Y%m%d')

        # 找出MACD
        macd, macd_signal, _ = talib.MACD(df['S_DQ_CLOSE'].values)
        df['macd'] = macd
        df['macd_signal'] = macd_signal

        # 找出金叉和死叉
        df['golden_cross'] = np.where((df['macd'] > df['macd_signal']) & (
            df['macd'].shift() < df['macd_signal'].shift()), 1, 0)
        df['death_cross'] = np.where((df['macd'] < df['macd_signal']) & (
            df['macd'].shift() > df['macd_signal'].shift()), 1, 0)

        # 选出金叉和死叉
        # * df1保存了金叉、死叉，shift_date是下一个x的日期
        df1 = df[(df['golden_cross'] == 1) | (df['death_cross'] == 1)]
        df1['shift_date'] = df1['TRADE_DT'].shift(-1)
        df1 = df1.dropna(subset=['shift_date'])
        df2 = pd.DataFrame()
        df5 = pd.DataFrame()
        for index, row in df1.iterrows():
            # df3保存了两个x之间的df
            df3 = df[(df['TRADE_DT'] >= row['TRADE_DT']) &
                     (df['TRADE_DT'] <= row['shift_date'])]
            if df3.iloc[0]['golden_cross'] == 1:
                # * 金叉->死叉，之间是高点
                df4 = df3[df3['S_DQ_CLOSE'].values == df3['S_DQ_CLOSE'].max()]
                df4 = df4.head(1)
                df2 = pd.concat([df2, df4])
            elif df3.iloc[0]['death_cross'] == 1:
                # * 死叉->金叉，之间是低点
                df4 = df3[df3['S_DQ_CLOSE'].values == df3['S_DQ_CLOSE'].min()]
                df4 = df4.head(1)
                df5 = pd.concat([df5, df4])

        # 按日期排序
        df.sort_values('TRADE_DT', inplace=True)
        '''
        # 初始化变量
        peaks = []
        valleys = []

        # 找到峰值和谷值
        for i in range(1, len(df) - 1):
            if df['S_DQ_CLOSE'][i] > df['S_DQ_CLOSE'][i-1] and df['S_DQ_CLOSE'][i] > df['S_DQ_CLOSE'][i+1]:
                peaks.append({'date': df['TRADE_DT'][i], 'price': df['S_DQ_CLOSE'][i]})
            elif df['S_DQ_CLOSE'][i] < df['S_DQ_CLOSE'][i-1] and df['S_DQ_CLOSE'][i] < df['S_DQ_CLOSE'][i+1]:
                valleys.append({'date': df['TRADE_DT'][i], 'price': df['S_DQ_CLOSE'][i]})

        # 计算波动周期的时间间隔和涨幅
        result_data = []
        for i in range(len(valleys) - 1):
            valley_date = valleys[i]['date']
            valley_price = valleys[i]['price']
            next_valley_date = valleys[i+1]['date']
            next_valley_price = valleys[i+1]['price']
            
            # 寻找最高点价格
            highest_price = df[(df['TRADE_DT'] > valley_date) & (df['TRADE_DT'] < next_valley_date)]['S_DQ_CLOSE'].max()
            
            price_change = (highest_price - valley_price) / valley_price * 100
            interval = (next_valley_date - valley_date).days

            result_data.append({'波动周期起始日期': valley_date,
                                '波动周期终止日期': next_valley_date,
                                '最低点价格': valley_price,
                                '最高点价格': highest_price,
                                '涨幅': price_change,
                                '时间间隔（天）': interval})
        '''

        # 计算高点和低点的时间间隔和涨幅
        result_data2 = []
        for i in range(min(len(df2), len(df5))):
            high_date = df2.iloc[i]['TRADE_DT']
            low_date = df5.iloc[i]['TRADE_DT']
            high_price = df2.iloc[i]['S_DQ_CLOSE']
            low_price = df5.iloc[i]['S_DQ_CLOSE']
            if df2.iloc[0]['TRADE_DT'] < df5.iloc[0]['TRADE_DT']:
                interval = (low_date - high_date).days
                price_change = (low_price - high_price) / high_price * 100
                result_data2.append({'高点日期': high_date, '低点日期': low_date, '最高点价格': high_price,
                                    '最低点价格': low_price, '时间间隔（天）': interval, '变化': price_change})
            else:
                interval = (high_date - low_date).days
                price_change = (high_price - low_price) / low_price * 100
                result_data2.append({'低点日期': low_date, '高点日期': high_date, '最低点价格': low_price,
                                    '最高点价格': high_price, '时间间隔（天）': interval, '变化': price_change})

        # 将结果保存到CSV文件（指定编码为UTF-8）
        output_filename = os.path.splitext(filename)[0] + '_result.csv'
        output_file_path = os.path.join(results_directory, output_filename)
        result_df = pd.DataFrame(result_data2)
        result_df.to_csv(output_file_path, index=False, encoding='utf-8-sig')

        # 统计波数和每波反弹幅度

        # high曲线的谷值，找低点
        highs_valleys = []

        for i in range(2, len(df2)-1):
            if df2.iloc[i]['S_DQ_CLOSE'] < df2.iloc[i-1]['S_DQ_CLOSE'] and df2.iloc[i]['S_DQ_CLOSE'] < df2.iloc[i+1]['S_DQ_CLOSE']:
                highs_valleys.append(
                    {'date': df2.iloc[i]['TRADE_DT'], 'price': df2.iloc[i]['S_DQ_CLOSE']})
        # 下降的波数
        waves_fall = len(highs_valleys)

        # low曲线的峰值，找高点
        lows_peaks = []

        for i in range(2, len(df5)-1):
            if df5.iloc[i]['S_DQ_CLOSE'] > df5.iloc[i-1]['S_DQ_CLOSE'] and df5.iloc[i]['S_DQ_CLOSE'] > df5.iloc[i+1]['S_DQ_CLOSE']:
                lows_peaks.append(
                    {'date': df5.iloc[i]['TRADE_DT'], 'price': df5.iloc[i]['S_DQ_CLOSE']})
        # 上升的波数
        waves_rise = len(lows_peaks)

        lower_high_count = 0
        satisfied_highs_count = 0  # 计数满足条件的高点数量
        for i in range(1, len(df2)):
            # 判断当前high点是否比上一个high点低
            if df2.iloc[i]['S_DQ_CLOSE'] < df2.iloc[i-1]['S_DQ_CLOSE']:
               #!! 找到当前high点前的low点
                t1 = df2.iloc[i]['TRADE_DT']
                t2 = df2.iloc[i-1]['TRADE_DT']
                low_points_before_high = df5[(
                    df5['TRADE_DT'] <= t1) & (df5['TRADE_DT'] >= t2)]

                if len(low_points_before_high) > 0:
                    lower_high_count += 1
                    # 计算当前high点前的low点到当前high点的涨幅
                    low_price = low_points_before_high['S_DQ_CLOSE'].iloc[-1]
                    high_price = df2.iloc[i]['S_DQ_CLOSE']
                    price_change = (high_price - low_price)

                    # 计算上一个high点到上一个low点的跌幅的一半
                    price_change_prev = (
                        df2.iloc[i-1]['S_DQ_CLOSE'] - low_price)
                    # 判断涨幅是否达到了跌幅的一半
                    price_change_prev_half = price_change_prev / 2
                    if (price_change >= price_change_prev_half):
                        satisfied_highs_count += 1
        percentage = 0
        if lower_high_count > 0:
            percentage = "{:.2f}%".format(
                (satisfied_highs_count / lower_high_count) * 100)
        mean_waves_fall = 0
        if waves_fall > 0:
            mean_waves_fall = lower_high_count/waves_fall

        higher_low_count = 0
        satisfied_lows_count = 0  # 计数满足条件的低点数量
        for i in range(1, len(df5)):
            # 判断当前low点是否比上一个low点高
            if df5.iloc[i]['S_DQ_CLOSE'] > df5.iloc[i-1]['S_DQ_CLOSE']:
                # 找到当前low点前的high点
                t1 = df5.iloc[i]['TRADE_DT']
                t2 = df5.iloc[i-1]['TRADE_DT']
                high_points_before_low = df2[(
                    df2['TRADE_DT'] <= t1) & (df2['TRADE_DT'] >= t2)]

                if len(high_points_before_low) > 0:
                    higher_low_count += 1
                    # 计算当前low点前的high点到当前low点的跌幅
                    high_price = high_points_before_low['S_DQ_CLOSE'].iloc[-1]
                    low_price = df5.iloc[i]['S_DQ_CLOSE']
                    price_change = (high_price - low_price)

                    # 计算上一个high点到上一个low点的跌幅的一半
                    price_change_prev = - \
                        (df5.iloc[i-1]['S_DQ_CLOSE'] - high_price)
                    # 判断涨幅是否达到了跌幅的一半
                    price_change_prev_half = price_change_prev / 2
                    if (price_change <= price_change_prev_half):
                        satisfied_lows_count += 1
        percentage_lows = 0
        if higher_low_count > 0:
            percentage_lows = "{:.2f}%".format(
                (satisfied_lows_count / higher_low_count) * 100)
        mean_waves_rise = 0
        if waves_rise > 0:
            mean_waves_rise = higher_low_count/waves_rise

        result_data_waves.append({"Stock:": stock_name,
                                  "下跌次数：": waves_fall,
                                  "下跌波数均值：": mean_waves_fall,
                                  "下跌过程涨幅超过中位线的highs占比：": percentage,
                                  "上涨次数：": waves_rise,
                                  "上涨波数均值：": mean_waves_rise,
                                  "上涨过程跌幅超过中位线的lows占比：": percentage_lows})

# 将结果保存到CSV文件（指定编码为UTF-8）
waves_output_filename = 'waves_result.csv'
waves_output_file_path = os.path.join(results_directory, waves_output_filename)
waves_result_df = pd.DataFrame(result_data_waves)
waves_result_df.to_csv(waves_output_file_path,
                       index=False, encoding='utf-8-sig')
