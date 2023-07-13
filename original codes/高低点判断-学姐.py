import os
import pandas as pd
import matplotlib.pyplot as plt
import datetime

# 获取股票数据文件夹的路径
data_directory = '/Users/zhaochenxi/Desktop/quant/data_csv1'
# 设置结果保存文件夹的路径
results_directory = '/Users/zhaochenxi/Desktop/quant/data_csv_results1'

# 创建结果保存文件夹
if not os.path.exists(results_directory):
    os.makedirs(results_directory)

# 遍历目录中的所有文件
for filename in os.listdir(data_directory):
    if filename.endswith('.csv'):
        # 构建完整的文件路径
        file_path = os.path.join(data_directory, filename)
        
        # 读取数据并创建DataFrame
        data = pd.read_csv(file_path)
        df = pd.DataFrame(data)
        
        # 转换日期列为日期时间类型
        df['TRADE_DT'] = pd.to_datetime(df['TRADE_DT'], format='%Y%m%d')
        
        # 按日期排序
        df.sort_values('TRADE_DT', inplace=True)
        
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

        # 将结果保存到CSV文件（指定编码为UTF-8）
        output_filename = os.path.splitext(filename)[0] + '_result.csv'
        output_file_path = os.path.join(results_directory, output_filename)
        result_df = pd.DataFrame(result_data)
        result_df.to_csv(output_file_path, index=False, encoding='utf-8-sig')
        
        # 获取最后100天的数据
        last_hundred_days_df = df.tail(100)

        # 绘制折线图
        plt.plot(last_hundred_days_df['TRADE_DT'], last_hundred_days_df['S_DQ_CLOSE'], color='blue')
        
        
        # 将last_hundred_days_df['TRADE_DT']转换为与peaks中日期格式相同的字符串格式
        last_hundred_days_dates = last_hundred_days_df['TRADE_DT'].dt.strftime('%Y-%m-%d')
        # 提取最后100天内的峰值和谷值
        last_hundred_days_peaks = [peak for peak in peaks if peak['date'].strftime('%Y-%m-%d') in last_hundred_days_dates.values]
        last_hundred_days_valleys = [valley for valley in valleys if valley['date'].strftime('%Y-%m-%d') in last_hundred_days_dates.values]


        # 标记峰值和谷值
        for peak in last_hundred_days_peaks:
            plt.scatter(peak['date'], peak['price'], color='red', marker='^', label='Peak')
        for valley in last_hundred_days_valleys:
            plt.scatter(valley['date'], valley['price'], color='green', marker='v', label='Valley')

        # 设置图形标题和标签
        plt.title('Stock Price')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend(['Stock Price', 'Peak', 'Valley'])

        # 显示图形
        plt.show()




        


