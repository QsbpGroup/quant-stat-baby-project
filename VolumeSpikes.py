import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


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
