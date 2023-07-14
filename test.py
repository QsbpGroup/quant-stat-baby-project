import datetime
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt


def find_horizontal_area1(df, high_points, low_points, var=20, max_len_of_window=30, min_len_of_window=10, gamma=0.4, view_coe=1, ignore_hl=True, draw_hist=False):
    """
    此函数的输出为一个dataframe，包含以下列：start_date, end_date, start_price, end_price, price_change, interval
    横盘的定义是：此段区间内价格变化不超过滑动窗口内最大价格变化的40%；滑动窗口的定义是从start_date向前数10天，到end_date向后数10天
    此函数的输出是一支股票所有的横盘区域
    """

    print('当前参数组合: max_len_of_window = {}, min_len_of_window = {}, gamma = {}'.format(
        max_len_of_window, min_len_of_window, gamma))

    result = pd.DataFrame(columns=[
                          'start_date', 'end_date', 'start_price', 'end_price', 'price_change', 'interval'])

    # 循环遍历每个高点和低点，确定横盘区间的起止日期，起始价格和结束价格
    index = 0
    for i in tqdm(range(len(df))):
        if i < index:
            continue
        for j in range(min_len_of_window+1, max_len_of_window+1):
            if i+j >= len(df):
                break
            start_date = df.iloc[i]['TRADE_DT']
            end_date = df.iloc[i+j]['TRADE_DT']

            # 判断横盘区间是否包含 high_points 和 low_points 中的点
            if ignore_hl:
                if any([high['high_date'] >= start_date and high['high_date'] <= end_date for high in high_points]):
                    continue
                if any([low['low_date'] >= start_date and low['low_date'] <= end_date for low in low_points]):
                    continue

            # 使用start date, end date, start peice, end price计算滑动窗口内的最大价格变化（窗口内所有价格的最高点减去最低点）
            start_date_window = start_date - \
                datetime.timedelta(days=max_len_of_window*view_coe)
            end_date_window = end_date + \
                datetime.timedelta(days=max_len_of_window*view_coe)
            window_price = df[(start_date_window <= df['TRADE_DT']) & (df['TRADE_DT'] <= end_date_window)]['S_DQ_CLOSE']
            # 找到window_price的分位数
            window_price_100 = window_price.quantile(1)
            window_price_75 = window_price.quantile(0.75)
            window_price_25 = window_price.quantile(0.25)
            window_price_0 = window_price.quantile(0)
            variance = window_price.var()
            max_change = window_price_100 - window_price_0

            # 计算横盘区间的价格变化和区间长度
            # price_change 是横盘区间内所有价格中最大价格减去最小价格，你需要遍历所有价格
            price_change_50 = window_price_75 - window_price_25
            interval = (end_date - start_date).days

            # 判断区间是否为横盘，如果是则将信息添加到 DataFrame 中
            if (price_change_50 >= gamma * max_change) & (variance < var):
                print('variance:', variance)
                result = result.append({'start_date': start_date,
                                        'end_date': end_date,
                                        'interval': interval}, ignore_index=True)
                index = i+j
    print('len of result:', len(result))
    print('mean of interval:', result['interval'].mean())
    print('--------------------------------------------------')
    if draw_hist:
        # 输出横盘区间的长度分布直方图然后关闭画布
        plt.hist(result['interval'], bins=100)
        plt.show()
        plt.close()
    return result
