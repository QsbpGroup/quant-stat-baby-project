import matplotlib.pyplot as plt
import pandas as pd
from decimal import Decimal
import numpy as np
import os
from tqdm import tqdm
import warnings
import datetime
warnings.filterwarnings("ignore")

path_raw = '/Users/kai/Desktop/qs'
data_directory = '/Users/kai/Desktop/qs/data_csv_distinct_0606'
# buy_close_dir = r'D:\360MoveData\Users\86158\Desktop\小组作业\codes\buy_close_data\buy_close_data'
# 合并买入卖出文件并保存
# data_buy_sell=pd.DataFrame()
# for file in tqdm(os.listdir(buy_close_dir)):
#    data=pd.read_csv(buy_close_dir+'/'+file)
#   data['code']=file.split('.')[0]+'.'+file.split('.')[1]
#   data_buy_sell=pd.concat([data_buy_sell,data])

# data_buy_sell.to_csv(r'D:\360MoveData\Users\86158\Desktop\小组作业\codes\汇总buy_sell.csv',index=False)

data_buy_sell = pd.read_csv('/Users/kai/Desktop/qs/quant-stat-baby-project/temp/total_buy_sell.csv')
data_buy_sell.columns = ['TRADE_DT', 'type', 'code']

# 导入ST以及退市相关股票并去除
df_st = pd.read_excel('/Users/kai/Desktop/qs/实施ST.xlsx')
df_st2 = pd.read_excel('/Users/kai/Desktop/qs/退市资料.xlsx')
df_st = df_st.iloc[:, :2]
df_st2 = df_st2.iloc[:, :2]
df_st.columns = ['ind', 'code']
df_st2.columns = ['ind', 'code']
df_st = pd.concat([df_st, df_st2], axis=0)
df_st = df_st.drop_duplicates(subset=['code'])
df_st = df_st.dropna(subset=['ind'])

# 和买入卖出
data_buy_sell = pd.merge(data_buy_sell, df_st, how='outer')
data_buy_sell = data_buy_sell[~(data_buy_sell['ind'] > 0)]
data_buy_sell['tmp'] = data_buy_sell['code'].apply(lambda x:x.startswith('8'))
data_buy_sell=data_buy_sell[data_buy_sell['tmp']==False]
data_buy_sell.drop(columns=['tmp'],inplace=True)
# 交易日数据
trading_date = pd.read_csv(path_raw+'/工作日.csv')
trading_date['TRADE_DT'] = trading_date['TRADE_DT'].apply(lambda x: pd.to_datetime(x, format='%Y%m%d'))
start_date='2005-01-01'

def performance(nav_df,freq='D'):
    mapping={'D':252,'W':52,'M':12,'Q':4}
    n=mapping.get(freq)
    '''
    log1p(number)函数：
    返回log(1+number)自然对数的结果
    '''
    ##part1:根据回测净值计算相关指标的数据准备（日度数据）
    if len(nav_df)>1:
        nav_next=nav_df.shift(1)
        return_df=(nav_df-nav_next)/nav_next  #计算净值变化率，即为日收益率,包含组合与基准
        return_df=return_df.iloc[1:]  #在计算净值变化率时，首日得到的是缺失值，需将其删除
        analyze=pd.DataFrame() if type(nav_df)==pd.core.frame.DataFrame else  pd.DataFrame(index=[0])#用于存储计算的指标
        #len(return_df)/n表示有多少年，倒过来相当于就是求这个值的开方，求（累计收益率+1）年数的开方再减1，就是年化收益率
        analyze['累计收益率'] = nav_df.iloc[-1]/nav_df.iloc[0] - 1
        analyze['年化收益率']=(nav_df.iloc[-1,:]/nav_df.iloc[0,:])**(n/(len(nav_df)-1))-1  if type(nav_df)==pd.core.frame.DataFrame else  (nav_df.iloc[-1]/nav_df.iloc[0])**(n/(len(nav_df)-1))-1##将年化收益率的Series赋值给数据框
        #part3:计算收益波动率（以年为基准）
        analyze['收益波动率']=return_df.std()*np.sqrt(n) #return_df中的收益率为日收益率，所以计算波动率转化为年时，需要乘上np.sqrt(n)
        max_return=nav_df.cummax()
        analyze['最大回撤']=nav_df.sub(max_return).div(max_return).min()  #最大回撤一般小于0，越小，说明离1越远，各时间点与最大收益的差距越大
        risk_free=0
    #    analyze['夏普比率']=((1+return_df.mean())**n-1-risk_free)/(return_df.std()*np.sqrt(n)) #由此计算的是年化的!非年化则不需*np.sqrt(n)
    #    analyze['夏普比率']=np.sqrt(n)*(return_df.mean()-risk_free)/return_df.std()    
        analyze['夏普比率']=analyze['年化收益率']/analyze['收益波动率']
        analyze['卡玛比率']=analyze['年化收益率']/abs(analyze['最大回撤'])
    else:
        return None
    return analyze 


def get_combine_data(df,index_code="000300.SH",start_date='2005-01-01'):
    '''
    

    Parameters
    ----------
    df : pd.DataFrame
        收益率序列数据,包括日期与净值两列.
    index_code : str, optional
        需要对比的指数。The default is "000300.SH".
    start_date : str, optional
        开始日期. The default is '2005-01-01'.

    Returns
    -------
    df_combine : pd.DataFrame
        与指数合并后的数据.

    '''
    
    #导入沪深300的数据
    today=datetime.date.today().strftime(format='%Y-%m-%d')
    HS300_df =pd.read_csv('/Users/kai/Desktop/qs/HS300.csv')
    HS300_df['TRADE_DT'] = pd.to_datetime(HS300_df['TRADE_DT'])
    HS300_df.dropna(inplace=True)
    HS300_df=HS300_df[HS300_df['TRADE_DT']>=pd.to_datetime(start_date)]
    HS300_df['HS300']=HS300_df['close']/HS300_df.iloc[0,1]
    HS300_df = HS300_df[['TRADE_DT', 'HS300']]
    df['TRADE_DT'] = pd.to_datetime(df['TRADE_DT'])
    df_combine=pd.merge(HS300_df,df)
    return df_combine

def max_draw(df):
    '''
    

    Parameters
    ----------
    df : pd.DataFrame
        净值序列。

    Returns
    -------
    所有日期最大回撤与最大回撤。

    '''
    max_draw=(df-df.cummax())/df.cummax()
    return -np.min(max_draw),max_draw



        
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
        return pd.DataFrame(columns=['date', 'price', 'status'])

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
            df0 = df.iloc[:i+1, :]
            df0['status'] = 1
            # return pd.DataFrame([[df['date'][i], df['price'][i], 1]], columns=['date', 'price', 'status'])
            return df0
        elif df['price'][i] <= lower_bound:
            df0 = df.iloc[:i+1, :]
            df0['status'] = -1
            # return pd.DataFrame([[df['date'][i], df['price'][i], 0]], columns=['date', 'price', 'status'])
            return df0

    # 若没有平仓操作, 返回空DataFrame
    df0 = df.copy()
    df0['status'] = 0
    return df0


def cal_net_value(df, capital=1):
    '''
    根据当只股票持仓数据获得日度净值数据，第一天的净值为1

    Parameters
    ----------
    df : DataFrame
        列：[日期，价格，类别]. 从买入当天到卖出当天的数据, 包含了买入当天的价格.
    capital : float
        本金。

    Returns
    -------
    第一个结果返回df, [净值数据, date]
    第二个结果返回收益率

    '''
    #print('capital is',capital)
    assert capital > 0, '本金capital必须是一个正数'
    df_copy = df.copy()
    df_copy['S_DQ_CLOSE'] = capital*df_copy['S_DQ_CLOSE'] / \
        float(df_copy.iloc[0]['S_DQ_CLOSE'])
    return df_copy, float(df_copy.iloc[-1]['S_DQ_CLOSE'])


def df_init(filename='000001.SZ.csv'):
    # 构建完整的文件路径
    file_path = os.path.join(data_directory, filename)
    # print('Current file path is', file_path)
    df = pd.read_csv(file_path)
    df['TRADE_DT'] = pd.to_datetime(df['TRADE_DT'], format='%Y%m%d')
    # 仅用到日期和收盘价两列
    df = df[['TRADE_DT', 'S_DQ_CLOSE']]
    return df
z=df_init()

def tackle_buy(df_buy_this_day, df_nv, date_change, buy_money, buy_num, ret_dict, index_dict, num_index, buy_df):
    '''

    买入一只股票，寻找这只股票的买入卖出点位，更新还能够购买的股票数量与现金与每天净值，并更新buy_df这张表

    Parameters
    ----------
    df_buy_this_day : pd.DataFrame
        当天买入的股票信息，包括日期、买入或者卖出类别以及代码
    df_nv : pd.DataFrame
        每个交易日现金以及持仓股票净值信息。第一列为时间，第二列为现金，之后的列每一列代表一次买入和卖出的数据，其他区间的数据为na。
    date_change : str
        买入股票的日期.
    buy_money : float
        本金.
    buy_num : int
        最多还能买入股票的数量.
    ret_dict : dict
        没啥用.
    index_dict : dict
        买入股票在df_nv中的列名代表的股票代码，例如{1：‘000001.SZ'}.
    num_index : int
        df_nv中的列名，每买入一次增加1.
    buy_df : pd.DataFrame
        对应日期买入和卖出的股票。

    Returns
    -------
    同Parameters中的变量

    '''
    # 取代码与相应股价数据
    code = df_buy_this_day['code'].values[0]
    df_code = df_init(filename=code+'.csv')
    # print(code,df_code)
    df_code = df_code[df_code['TRADE_DT'] >= date_change]
    # 计算股票买入日到卖出日的净值
    df_close = close_position(df_code, threshold=0.1)
    df_close.columns = ['TRADE_DT', 'S_DQ_CLOSE', 'status']
    df_close = df_close.iloc[:, :2]
    df_close, ret = cal_net_value(df_close, capital=buy_money/buy_num)
    df_close = df_close.rename(columns={'S_DQ_CLOSE': num_index})
    # 与每日净值表合并
    df_nv = pd.merge(df_nv, df_close, how='outer')
    # 更新需要的参数
    cash_buy = buy_money/buy_num
    buy_money -= cash_buy
    buy_num -= 1
    ret_dict[code] = ret
    index_dict[code] = num_index
    num_index += 1
    df_nv.loc[df_nv['TRADE_DT'] == date_change, 'cash'] = buy_money
    try:
        if len(buy_df.loc[date_change, 'buy']) > 0:
            buy_df.loc[date_change, 'buy'].append(code)
    except Exception:
        buy_df.loc[date_change, 'buy'] = [code]
    return df_nv, buy_money, buy_num, ret_dict, index_dict, num_index, buy_df


def tackle_sell(df_sell_this_day, df_nv, date_change, buy_money, buy_num, index_dict, buy_df,commission):
    '''


    Parameters
    ----------
    df_sell_this_day : pd.DataFrame
        当天卖出的股票信息，包括日期、买入或者卖出类别以及代码
    df_nv : pd.DataFrame
        每个交易日现金以及持仓股票净值信息。第一列为时间，第二列为现金，之后的列每一列代表一次买入和卖出的数据，其他区间的数据为na。
    buy_money : float
        本金.
    buy_num : int
        最多还能买入股票的数量.
    index_dict : dict
        买入股票在df_nv中的列名代表的股票代码，例如{1：‘000001.SZ'}.
    buy_df : pd.DataFrame
        对应日期买入和卖出的股票。

    Returns
    -------
    同Parameters中的变量

    '''
    # 如果是卖出信号，更新我需要的值
    code = df_sell_this_day['code'].values[0]
    index_date = df_nv[df_nv['TRADE_DT'] == date_change].index[0]
    df_nv.iloc[:index_date+1, 1] = df_nv.loc[df_nv['TRADE_DT']<= date_change, 'cash'].fillna(method='pad', axis=0)
    df_nv.loc[df_nv['TRADE_DT'] == date_change, index_dict[code]]=float(df_nv.loc[df_nv['TRADE_DT'] == date_change, index_dict[code]])*(1-commission)
    df_nv.loc[df_nv['TRADE_DT'] == date_change, 'cash'] = df_nv.loc[df_nv['TRADE_DT'] ==date_change, 'cash']+float(df_nv.loc[df_nv['TRADE_DT'] == date_change, index_dict[code]])
    buy_money = float(df_nv.loc[df_nv['TRADE_DT'] == date_change, 'cash'])
    df_nv.loc[df_nv['TRADE_DT'] == date_change, index_dict[code]] = np.nan
    buy_num += 1
    index_dict.pop(code)
    try:
        if len(buy_df.loc[date_change, 'sell']) > 0:
            buy_df.loc[date_change, 'sell'].append(code)
    except Exception:
        buy_df.loc[date_change, 'sell'] = [code]
    # pd.isna(buy_df.loc[date_change,'sell']):
    #     buy_df.loc[date_change,'sell']=[[code]]
    # else:
    #   buy_df.loc[date_change,'sell'].append(code)
    buy_df.to_csv(path_raw + '/buy_df.csv')
    return df_nv, buy_money, buy_num, index_dict, buy_df


# 假设有所有交易日的dataframe,只有一列是时间
# 暂时将最后一天还没卖出的当作日期也写上了


def net_value_all(df, df_t, max_num=20, init_money=1,commission=0.002):
    '''


    Parameters
    ----------
    df : pd.DataFrame
        需要买入和卖出的所有股票信息.包括日期、买入或者卖出类别以及代码
    df_t : pd.DataFrame
        交易日数据.
    max_num : int, optional
        最多能够买的股票数量. The default is 20.
    init_money : float, optional
        初始资金. The default is 1.

    Returns
    -------
    df_nv_output : pd.DataFrame
        最终净值.
    ret : float
        累计净值.

    '''
    num_index = 0
    df_nv = df_t.copy()
    df_nv['cash'] = np.nan
    df_nv.iloc[0, 1] = init_money
    # xiangyixiang设成0还是20
    # 还可以买的数量
    buy_num = max_num
    # 初始资金
    buy_money = init_money
    index_dict = {}
    ret_dict = {}
    # 用来记录每天买入卖出股票的buy_df
    buy_df = pd.DataFrame(columns=['TRADE_DT', 'buy', 'sell'])
    buy_df['TRADE_DT'] = df_t['TRADE_DT']
    for z in range(len(buy_df)):
        buy_df.iloc[z, 1] = np.nan
        buy_df.iloc[z, 2] = np.nan
    buy_df.set_index('TRADE_DT', inplace=True)
    for date_change in tqdm(np.sort(df['TRADE_DT'].unique())):
        # 这里加try是因为数据里面有一条在我们测试范围最后一天卖出的数据，会报错，但是不影响结果，所以不管
        try:
            if 'close' in df[df['TRADE_DT'] == date_change]['type'].values:
                # 因为如果超出数量我们会随机选择股票，所以可能在df出现卖出股票信息但实际没有卖掉
                df_sell_this_day = df[(df['TRADE_DT'] == date_change) & (df['type'] == 'close')]
                df_sell_this_day = df_sell_this_day[df_sell_this_day['code'].isin(
                    list(index_dict.keys()))]
            # 因为如果只有一行不能iterrows，所以分情况分析
                if len(df_sell_this_day) == 1:
                    df_nv, buy_money, buy_num, index_dict, buy_df = tackle_sell(
                        df_sell_this_day, df_nv, date_change, buy_money, buy_num, index_dict, buy_df,commission)  # 1
                else:
                    for _, row in df_sell_this_day.iterrows():
                        df_nv, buy_money, buy_num, index_dict, buy_df = tackle_sell(pd.DataFrame(
                            row).T, df_nv, date_change, buy_money, buy_num, index_dict, buy_df,commission)
            if 'buy' in df[df['TRADE_DT'] == date_change]['type'].values:
                if buy_num == 0:
                    continue
                df_buy_this_day = df[(df['TRADE_DT'] == date_change) & (df['type'] == 'buy')]
                if buy_num-len(df_buy_this_day) < 0:
                    df_buy_this_day = df_buy_this_day.sample(n=buy_num,random_state=1231213)
                if len(df_buy_this_day) == 1:
                    df_nv, buy_money, buy_num, ret_dict, index_dict, num_index, buy_df = tackle_buy(
                        df_buy_this_day, df_nv, date_change, buy_money, buy_num, ret_dict, index_dict, num_index, buy_df)
                else:
                    for _, row in df_buy_this_day.iterrows():
                        df_nv, buy_money, buy_num, ret_dict, index_dict, num_index, buy_df = tackle_buy(pd.DataFrame(
                            row).T, df_nv, date_change, buy_money, buy_num, ret_dict, index_dict, num_index, buy_df)
        except Exception:
            continue
    # 补全净值
    df_nv['cash'] = df_nv['cash'].fillna(method='pad', axis=0)
    # df_nv.to_excel(r'D:\360MoveData\Users\86158\Desktop\df_nv.xlsx')
    # cc=df_nv[(df_nv['TRADE_DT']<pd.to_datetime('2007-08-30'))&(df_nv['TRADE_DT']>pd.to_datetime('2006-08-22'))]

    df_nv_output = df_t.copy()
    df_nv_output['cash'] = df_nv.sum(axis=1)
    ret = df_nv_output.iloc[-1]['cash']-1
    # ann_ret=(1+ret)**(252/len(df_nv_output))-1
    return df_nv_output, ret


# df=data_buy_sell

df_nv_output, ret = net_value_all(df=data_buy_sell, df_t=trading_date)
df_nv_output_05=df_nv_output[df_nv_output['TRADE_DT']>=start_date]

df_nv_output_05['cash']=df_nv_output_05['cash']/df_nv_output_05.iloc[0,1]
df_nv_output_05['TRADE_DT']=df_nv_output_05['TRADE_DT'].apply(lambda x:datetime.datetime.strftime(x,'%Y-%m-%d'))
df_combine=get_combine_data(df_nv_output_05,index_code="000300.SH",start_date=start_date)
#画图并保存
plt.plot(df_combine['TRADE_DT'], df_combine['cash'])

plt.plot(df_combine['TRADE_DT'], df_combine['HS300'])
#plt.yticks(list(np.arange(0,25)))
plt.legend(['strategy','HS300'])
plt.savefig(path_raw+'/net_value.jpg', dpi=200, bbox_inches='tight',figsize=(16,8))

#输出评价指标
df_sta=performance(df_combine['cash'])
df_sta=pd.concat([df_sta,performance(df_combine['HS300'])])
df_sta.index=['strategy','HS300']
df_sta.to_excel(path_raw+'/performance.xlsx')
df_nv_output.to_excel(path_raw+'/net_value_raw.xlsx',index=False)
df_combine.to_excel(path_raw+'/net_value.xlsx',index=False)

# 保存交易历史数据
nv = pd.read_excel('/Users/kai/Desktop/qs/net_value.xlsx')
nv.columns = ['TRADE_DT', 'HS300', 'ours']
trade_history = pd.read_csv('/Users/kai/Desktop/qs/buy_df.csv')
# 在buy添加nv中的两列，按照TRADE_DT匹配
trade_history['HS300'] = trade_history['TRADE_DT'].map(nv.set_index('TRADE_DT')['HS300'])
trade_history['ours'] = trade_history['TRADE_DT'].map(nv.set_index('TRADE_DT')['ours'])
trade_history.to_csv('/Users/kai/Desktop/qs/trade_history.csv', index=False)
