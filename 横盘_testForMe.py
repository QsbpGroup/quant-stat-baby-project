import pandas as pd
import os


def find_sideways_movement(data, date, alpha, beta):
    sideways_movement = []
    start_date = data.iloc[0]['TRADE_DT']
    end_date = start_date + pd.DateOffset(weeks=date)  # 初始的八周

    while end_date <= data.iloc[-1]['TRADE_DT']:
        range_data = data[(data['TRADE_DT'] >= start_date)
                          & (data['TRADE_DT'] <= end_date)]
        high_price = range_data['S_DQ_HIGH'].max()
        low_price = range_data['S_DQ_LOW'].min()
        range_amplitude = (high_price - low_price) / low_price
        range_return = (range_data.iloc[-1]['S_DQ_CLOSE'] - range_data.iloc[0]
                        ['S_DQ_OPEN']) / range_data.iloc[0]['S_DQ_OPEN']

        if range_amplitude <= alpha and abs(range_return) <= beta:
            current_end_date = end_date
            while current_end_date < data.iloc[-1]['TRADE_DT']:
                current_end_date += pd.DateOffset(days=1)
                current_range_data = data[(data['TRADE_DT'] >= start_date) & (
                    data['TRADE_DT'] <= current_end_date)]
                current_range_return = (
                    current_range_data.iloc[-1]['S_DQ_CLOSE'] - current_range_data.iloc[0]['S_DQ_OPEN']) / current_range_data.iloc[0]['S_DQ_OPEN']
                current_high_price = current_range_data['S_DQ_HIGH'].max()
                current_low_price = current_range_data['S_DQ_LOW'].min()
                current_range_amplitude = (current_high_price - current_low_price) / current_low_price

                if abs(current_range_return) > beta or current_range_amplitude > alpha:
                    break

            sideways_movement.append({
                'start_date': start_date,
                'end_date': current_end_date,
                'range_amplitude': range_amplitude,
                'range_return': range_return
            })

            start_date = current_end_date + \
                pd.DateOffset(days=1)  # 开始日期更新为当前区间结束日期的下一天
            end_date = start_date + pd.DateOffset(weeks=8)  # 新的结束日期为开始日期后的八周
        else:
            end_date += pd.DateOffset(days=1)  # 结束日期向后移动一天

    return sideways_movement


# 文件路径
# data_directory = '/Users/zhaochenxi/Desktop/quant/data_csv1'
# results_directory = '/Users/zhaochenxi/Desktop/quant/data_csv_results2'
# 获取股票数据文件夹的路径
data_directory = '/Users/kai/Desktop/qs/data_csv_distinct_0606'
# 设置结果保存文件夹的路径
results_directory = '/Users/kai/Desktop/qs/data_csv_distinct_0606/__results'

# 遍历目录中的所有文件
# for filename in os.listdir(data_directory):
    # if filename.endswith('.csv'):
filename = '000001.SZ.csv'
        # 构建完整的文件路径
file_path = os.path.join(data_directory, filename)

# 读取数据并创建DataFrame
data = pd.read_csv(file_path)
df = pd.DataFrame(data)
df['up_flag'] = df['S_DQ_PCTCHANGE'].apply(lambda x: 1 if x > 0 else 0)
df['up_flag'] = df['up_flag'].astype('bool')

# 转换日期列为日期时间类型
df['TRADE_DT'] = pd.to_datetime(df['TRADE_DT'], format='%Y%m%d')

# 按日期排序
df.sort_values('TRADE_DT', inplace=True)

# 提取股票代码（文件名中的数字部分）
stock_code = os.path.splitext(filename)[0]

# 计算横盘区间
date = 24  # 横盘时间24周
alpha = 0.3  # 最大区间振幅30%
beta = 0.1  # 最大区间涨跌幅10%
sideways_movement = find_sideways_movement(df, date, alpha, beta)

# 保存结果文件
results_filename = f'{stock_code}_trading_ranges.csv'
results_file_path = os.path.join(results_directory, results_filename)
results_df = pd.DataFrame(sideways_movement)
results_df.to_csv(results_file_path, index=False)
