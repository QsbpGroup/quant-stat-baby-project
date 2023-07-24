import pandas as pd
from decimal import Decimal


def close_position(df, threshold=0.1):
    """ 
    计算平仓日期、价格、类别 (止盈/止损) 的函数

    Parameters
    ----------
    df : DataFrame
        列：[日期，价格]. 从买入当天到最后一天的数据, 包含了买入当天的价格.
    threshold : float, optional
        止盈/止损的阈值, by default 0.1.

    Returns
    -------
    close_position_info : DataFrame
        列：[卖出日期，卖出价格，类别 (止盈=1, 止损=0)], i.e. ['date', 'price', 'type']. 
        返回空DataFrame表示没有平仓操作.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError('Input df must be a DataFrame.')
    if not isinstance(threshold, float):
        raise TypeError('Input threshold must be a float.')
    if threshold < 0:
        raise ValueError('Input threshold must be positive.')
    if len(df) < 2:
        raise ValueError('Input df must contain at least 2 rows.')

    # 初始化变量
    df.reset_index(drop=True, inplace=True)
    df.columns = ['date', 'price']
    buy_date = df['date'][0]
    buy_price = df['price'][0]
    # 使用 Decimal 计算上下界来避免浮点数精度误差   e.g. 100 * 1.1 = 110.00000000000001
    buy_price = Decimal(str(buy_price))
    upper_bound = float(buy_price * (1+Decimal(threshold)))
    lower_bound = float(buy_price * (1-Decimal(threshold)))

    # 遍历df
    for i in range(1, len(df)):
        if df['price'][i] >= upper_bound:
            return pd.DataFrame([[df['date'][i], df['price'][i], 1]], columns=['date', 'price', 'status'])
        elif df['price'][i] <= lower_bound:
            return pd.DataFrame([[df['date'][i], df['price'][i], 0]], columns=['date', 'price', 'status'])

    # 若没有平仓操作, 返回空DataFrame
    return pd.DataFrame(columns=['date', 'price', 'status'])
