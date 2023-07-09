import os
import time
import pandas as pd
import matplotlib.pyplot as plt

# 获取股票数据文件夹的路径
data_directory = '/Users/kai/Desktop/qs/data_csv_distinct_0606'
# 设置结果保存文件夹的路径
results_directory = os.path.join(
    data_directory, '_results_'+time.strftime("%h%d_%H%M"))


def df_init(filename='000001.SZ.csv'):
    # 构建完整的文件路径
    file_path = os.path.join(data_directory, filename)
    print(file_path)
    df = pd.read_csv(file_path)
    df['TRADE_DT'] = pd.to_datetime(df['TRADE_DT'], format='%Y%m%d')
    # 仅用到日期和收盘价两列
    df = df[['TRADE_DT', 'S_DQ_CLOSE']]
    return df


def dic_init():
    # 创建结果保存文件夹
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)


def find_high_low_xuejieZuhui(df, filename='000001.SZ.csv', n_days=200, draw=True):
        
    # 初始化变量
    peaks = []
    valleys = []

    # 找到峰值和谷值
    for i in range(1, len(df) - 1):
        if df['S_DQ_CLOSE'][i] > df['S_DQ_CLOSE'][i-1] and df['S_DQ_CLOSE'][i] > df['S_DQ_CLOSE'][i+1]:
            peaks.append({'date': df['TRADE_DT'][i], 'price': df['S_DQ_CLOSE'][i]})
        elif df['S_DQ_CLOSE'][i] < df['S_DQ_CLOSE'][i-1] and df['S_DQ_CLOSE'][i] < df['S_DQ_CLOSE'][i+1]:
            valleys.append({'date': df['TRADE_DT'][i],
                        'price': df['S_DQ_CLOSE'][i]})

    # 计算波动周期的时间间隔和涨幅
    result_data = []
    for i in range(len(valleys) - 1):
        valley_date = valleys[i]['date']
        valley_price = valleys[i]['price']
        next_valley_date = valleys[i+1]['date']
        next_valley_price = valleys[i+1]['price']

        # 寻找最高点价格
        highest_price = df[(df['TRADE_DT'] > valley_date) & (
            df['TRADE_DT'] < next_valley_date)]['S_DQ_CLOSE'].max()

        price_change = (highest_price - valley_price) / valley_price * 100
        interval = (next_valley_date - valley_date).days

        result_data.append({'波动周期起始日期': valley_date,
                            '波动周期终止日期': next_valley_date,
                            '最低点价格': valley_price,
                            '最高点价格': highest_price,
                            '涨幅': price_change,
                            '时间间隔（天）': interval})
    # 峰值曲线的谷值，找低点
    peaks_valleys = []

    for i in range(2, len(peaks)-1):
        if peaks[i-1]['price'] < peaks[i-2]['price'] and peaks[i]['price'] < peaks[i-1]['price'] and peaks[i]['price'] < peaks[i+1]['price']:
            peaks_valleys.append(
                {'date': peaks[i]['date'], 'price': peaks[i]['price']})

    low_points = []
    for pv in peaks_valleys:
        low_date_front = pv['date']
        low_index = valleys.index(
            next(v for v in valleys if v['date'] > low_date_front))
        low_price = valleys[low_index]['price']
        low_date = valleys[low_index]['date']
        low_points.append({'low_date': low_date, 'low_price': low_price})

    # 谷值曲线的峰值，找高点
    valleys_peaks = []

    for i in range(2, len(valleys)-1):
        if valleys[i-1]['price'] > valleys[i-2]['price'] and valleys[i]['price'] > valleys[i-1]['price'] and valleys[i]['price'] > valleys[i+1]['price']:
            valleys_peaks.append(
                {'date': valleys[i]['date'], 'price': valleys[i]['price']})

    high_points = []
    for vp in valleys_peaks:
        high_date_front = vp['date']
        high_index = peaks.index(
            next(p for p in peaks if p['date'] > high_date_front))
        high_price = peaks[high_index]['price']
        high_date = peaks[high_index]['date']
        high_points.append({'high_date': high_date, 'high_price': high_price})

    # 将结果保存到CSV文件（指定编码为UTF-8）
    output_filename = os.path.splitext(filename)[0] + '_result.csv'
    output_file_path = os.path.join(results_directory, output_filename)
    result_df = pd.DataFrame(result_data)
    result_df.to_csv(output_file_path, index=False, encoding='utf-8-sig')
    
    if(draw):
        # 获取最后100天的数据
        last_hundred_days_df = df.tail(n_days)
        # 绘制折线图
        plt.plot(last_hundred_days_df['TRADE_DT'],
                last_hundred_days_df['S_DQ_CLOSE'], color='royalblue', label='stock price', alpha=0.8)
        # 将last_hundred_days_df['TRADE_DT']转换为与peaks中日期格式相同的字符串格式
        last_hundred_days_dates = last_hundred_days_df['TRADE_DT'].dt.strftime(
            '%Y-%m-%d')
        # 提取最后100天内的峰值和谷值
        last_hundred_days_peaks = [peak for peak in peaks if peak['date'].strftime(
            '%Y-%m-%d') in last_hundred_days_dates.values]
        last_hundred_days_valleys = [valley for valley in valleys if valley['date'].strftime(
            '%Y-%m-%d') in last_hundred_days_dates.values]

        # 提取最后100天内的高点和低点
        last_hundred_days_high = [high_point for high_point in high_points if high_point['high_date'].strftime(
            '%Y-%m-%d') in last_hundred_days_dates.values]
        last_hundred_days_low = [low_point for low_point in low_points if low_point['low_date'].strftime(
            '%Y-%m-%d') in last_hundred_days_dates.values]


        # 标记峰值和谷值
        for peak in last_hundred_days_peaks:
            plt.scatter(peak['date'], peak['price'], color='red',
                        marker='^', label='Peak', alpha=0.3)
        for valley in last_hundred_days_valleys:
            plt.scatter(valley['date'], valley['price'],
                        color='green', marker='v', label='Valley', alpha=0.3)

        # 标记高点和低点
        for high_point in last_hundred_days_high:
            plt.scatter(high_point['high_date'], high_point['high_price'],
                        color='red', marker='*', label='high', s=80)
        for low_point in last_hundred_days_low:
            plt.scatter(low_point['low_date'], low_point['low_price'],
                        color='green', marker='*', label='low', s=80)

        # 设置图形标题和标签
        plt.title('Stock Price')
        plt.xlabel('Date')
        plt.ylabel('Price')
        # 获取当前图形中的所有句柄和标签
        handles, labels = plt.gca().get_legend_handles_labels()

        # 去除重复的标签
        unique_labels = set(labels)

        # 创建新的标签和句柄列表, 其中元素按照'stock price', 'Peak', 'Valley', 'high', 'low'的顺序排列
        new_labels = ['stock price', 'Peak', 'Valley', 'high', 'low']
        new_handles = []
        for new_label in new_labels:
            for i in range(len(labels)):
                if labels[i] == new_label:
                    new_handles.append(handles[i])
                    break

        plt.legend(handles=new_handles, labels=new_labels)
        plt.show()
        
    return (peaks, valleys, high_points, low_points)

# TODO: 暂存分析数据，自定义画图天数n，画出最近n天的图
# tODO：找到趋势中部的局部横盘；联系学妹看他的进度