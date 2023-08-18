from pandas import DataFrame, concat
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors
from high_low_xuejie_zuhui import find_hl_MACD_robust, df_init


def wave_identify(filename='000001.SZ.csv', alpha=0.05):
    """
    this goddamn function is used to identify the TRUE waves (with perfectly no overlap!) in the stock price

    * What makes it different and robust?
        * we ignored small fluctuations near zero on the MACD
        * we also ignored Extremely small fluctuations in the series of highs and lows(by using quantile methos)
        * We skillfully added split points inside the overlapping waves in the original waves so that each wave is independent



    Parameters
    ----------
    filename : str
        why use it if you have no idea what the filename is? in case u do not know, it is the sotck code that ends with .csv

    alpha : float
        quantile number used to drop some points from original waves, the larger alpha is, the more hl points will be dropped.

    Returns
    -------
    real_waves : DataFrame
        columns=['date', 'price', 'type'], type can be 'edo_fall'(a low point) and 'edo_rise'(a high point).
    """

    df = df_init(filename)
    high_points, low_points = find_hl_MACD_robust(df, filename, draw=False)
    hd = [item['high_date'] for item in high_points]
    hp = [item['high_price'] for item in high_points]
    ld = [item['low_date'] for item in low_points]
    lp = [item['low_price'] for item in low_points]
    h_cgdt = []
    l_cgdt = []
    for i in range(1, min(len(hp), len(lp))-1):
        if hp[i+1] > hp[i] and hp[i-1] > hp[i]:
            h_cgdt.append(
                {'date': hd[i], 'price': hp[i], 'pcg': max(hp[i+1], hp[i-1])-hp[i]})
        if lp[i+1] < lp[i] and lp[i-1] < lp[i]:
            l_cgdt.append(
                {'date': ld[i], 'price': lp[i], 'pcg': lp[i]-min(lp[i+1], lp[i-1])})
    h_cgdt = DataFrame(h_cgdt)
    l_cgdt = DataFrame(l_cgdt)
    if len(h_cgdt) == 0 or len(l_cgdt) == 0:
        return DataFrame(columns=['date', 'price', 'type'])
    h_cgdt = h_cgdt[h_cgdt['pcg'] > h_cgdt['pcg'].quantile(alpha)]
    h_cgdt['type'] = 'edo_fall'
    l_cgdt = l_cgdt[l_cgdt['pcg'] > l_cgdt['pcg'].quantile(alpha)]
    l_cgdt['type'] = 'edo_rise'
    all_cgdt = concat([h_cgdt, l_cgdt])
    all_cgdt.sort_values(by=['date'], inplace=True)
    all_cgdt.reset_index(drop=True, inplace=True)
    all_cgdt['true_date'] = None
    type_lst = all_cgdt['type'].tolist()
    dt_lst = all_cgdt['date'].tolist()
    prs_lst = all_cgdt['price'].tolist()
    real_waves = DataFrame(columns=['date', 'price', 'type'])
    h_df = DataFrame(high_points)
    l_df = DataFrame(low_points)
    for i in range(1, len(all_cgdt)):
        if type_lst[i] == type_lst[i-1]:
            if type_lst[i] == 'edo_fall':
                # 总之，这个地方的思想是在两个相邻的end of fall（高点）中找到一个end of rise（高点），vise versa
                tmp_highs = h_df[(h_df['high_date'] >= dt_lst[i-1])
                                 & (h_df['high_date'] <= dt_lst[i])]
                tmp_h = tmp_highs[tmp_highs['high_price']
                                  == tmp_highs['high_price'].max()]
                tmp_h = tmp_h[tmp_h['high_date'] == tmp_h['high_date'].max()]
                real_waves = real_waves.append(
                    {'date': tmp_h['high_date'].iloc[0], 'price': tmp_h['high_price'].iloc[0], 'type': 'edo_rise'}, ignore_index=True)
                # 找到low_points中在dt_lst[i]之前的最后一个low_point和之后的第一个low_point
                if ld[-1] <= dt_lst[i]:
                    real_waves = real_waves.append(
                        {'date': ld[-1], 'price': lp[-1], 'type': 'edo_fall'}, ignore_index=True)
                    continue
                low_dt1 = [item for item in ld if item < dt_lst[i]][-1]
                low_dt2 = [item for item in ld if item > dt_lst[i]][0]
                low_p1 = lp[ld.index(low_dt1)]
                low_p2 = lp[ld.index(low_dt2)]
                if low_p1 < low_p2:
                    real_waves = real_waves.append(
                        {'date': low_dt1, 'price': low_p1, 'type': 'edo_fall'}, ignore_index=True)
                else:
                    real_waves = real_waves.append(
                        {'date': low_dt2, 'price': low_p2, 'type': 'edo_fall'}, ignore_index=True)
            if type_lst[i] == 'edo_rise':
                tmp_lows = l_df[(l_df['low_date'] >= dt_lst[i-1])
                                & (l_df['low_date'] <= dt_lst[i])]
                tmp_l = tmp_lows[tmp_lows['low_price']
                                 == tmp_lows['low_price'].min()]
                tmp_l = tmp_l[tmp_l['low_date'] == tmp_l['low_date'].max()]
                real_waves = real_waves.append(
                    {'date': tmp_l['low_date'].iloc[0], 'price': tmp_l['low_price'].iloc[0], 'type': 'edo_fall'}, ignore_index=True)
                # 找到high_points中在dt_lst[i]之前的最后一个high_point和之后的第一个high_point
                if hd[-1] <= dt_lst[i]:
                    real_waves = real_waves.append(
                        {'date': hd[-1], 'price': hp[-1], 'type': 'edo_rise'}, ignore_index=True)
                    continue
                high_dt1 = [item for item in hd if item < dt_lst[i]][-1]
                high_dt2 = [item for item in hd if item > dt_lst[i]][0]
                high_p1 = hp[hd.index(high_dt1)]
                high_p2 = hp[hd.index(high_dt2)]
                if high_p1 > high_p2:
                    real_waves = real_waves.append(
                        {'date': high_dt1, 'price': high_p1, 'type': 'edo_rise'}, ignore_index=True)
                else:
                    real_waves = real_waves.append(
                        {'date': high_dt2, 'price': high_p2, 'type': 'edo_rise'}, ignore_index=True)
        else:
            # 这里要找到end of fall（高点）两侧更低的那个低点，vise versa
            if type_lst[i] == 'edo_fall':
                # 找到low_points中在dt_lst[i]之前的最后一个low_point和之后的第一个low_point
                if ld[-1] <= dt_lst[i]:
                    real_waves = real_waves.append(
                        {'date': ld[-1], 'price': lp[-1], 'type': 'edo_fall'}, ignore_index=True)
                    continue
                low_dt1 = [item for item in ld if item < dt_lst[i]][-1]
                low_dt2 = [item for item in ld if item > dt_lst[i]][0]
                low_p1 = lp[ld.index(low_dt1)]
                low_p2 = lp[ld.index(low_dt2)]
                if low_p1 < low_p2:
                    real_waves = real_waves.append(
                        {'date': low_dt1, 'price': low_p1, 'type': 'edo_fall'}, ignore_index=True)
                else:
                    real_waves = real_waves.append(
                        {'date': low_dt2, 'price': low_p2, 'type': 'edo_fall'}, ignore_index=True)
            if type_lst[i] == 'edo_rise':
                # 找到high_points中在dt_lst[i]之前的最后一个high_point和之后的第一个high_point
                if hd[-1] <= dt_lst[i]:
                    real_waves = real_waves.append(
                        {'date': hd[-1], 'price': hp[-1], 'type': 'edo_rise'}, ignore_index=True)
                    continue
                high_dt1 = [item for item in hd if item < dt_lst[i]][-1]
                high_dt2 = [item for item in hd if item > dt_lst[i]][0]
                high_p1 = hp[hd.index(high_dt1)]
                high_p2 = hp[hd.index(high_dt2)]
                if high_p1 > high_p2:
                    real_waves = real_waves.append(
                        {'date': high_dt1, 'price': high_p1, 'type': 'edo_rise'}, ignore_index=True)
                else:
                    real_waves = real_waves.append(
                        {'date': high_dt2, 'price': high_p2, 'type': 'edo_rise'}, ignore_index=True)

    return real_waves


def draw_waves(df, real_waves, fig_start_date, fig_end_date):
    """
    画出fig_start_date, fig_end_date之间的crash情况
    """
    # 初始化df_cache，避免浅拷贝导致的原始df被修改
    df_cache = df.copy()
    highs, lows = find_hl_MACD_robust(df_cache, draw=False)
    df_cache.columns = ['date', 'price']
    highs = DataFrame(highs)
    lows = DataFrame(lows)
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
        else:
            plt.axvspan(real_waves.iloc[i]['date'], real_waves.iloc[i+1]['date'],
                        facecolor='green', alpha=0.15, label='falling waves')
    if real_waves.iloc[0]['type'] == 'edo_fall':
        plt.axvspan(fig_start_date, real_waves.iloc[0]['date'],
                    facecolor='green', alpha=0.15, label='falling waves')
    else:
        plt.axvspan(fig_start_date, real_waves.iloc[0]['date'],
                    facecolor='red', alpha=0.15, label='rising waves')
    if real_waves.iloc[-1]['type'] == 'edo_fall':
        plt.axvspan(real_waves.iloc[-1]['date'], fig_end_date,
                    facecolor='red', alpha=0.15, label='rising waves')
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
                               'low points', 'rising waves', 'falling waves'], []
    for i in range(len(new_labels)):
        for j in range(len(labels)):
            if new_labels[i] == labels[j]:
                new_handles.append(handles[j])
                break
    plt.legend(new_handles, new_labels)
    plt.show()
    plt.close()


def cal_price_change_rate(filename='000001.SZ.csv', alpha=0.05):
    '''return a DataFrame of price change of each wave, columns=['price_change_rate', 'type']'''
    waves = wave_identify(filename, alpha)
    # calculate the price change of each wave
    pcg = []
    for i in range(len(waves)-1):
        pcg.append((waves.iloc[i+1]['price']-waves.iloc[i]['price'])*100/waves.iloc[i]['price'])
    pcg = DataFrame(pcg, columns=['price_change_rate'])
    # add sign if >0 or not
    pcg['type'] = pcg['price_change_rate'].apply(lambda x: 'rise' if x > 0 else 'fall')
    
    return pcg


def plot_pcg_rate(pcg, cut=0.99):
    pcg.reset_index(drop=True, inplace=True)
    pcg_cache = pcg.copy()
    pcg_cache = pcg_cache[pcg_cache['price_change_rate'] < pcg_cache['price_change_rate'].quantile(cut)]
    
    red = colors.hex2color('#fd5a45')
    green = colors.hex2color('#3bc66d') 
    plt.rcParams['figure.figsize'] = [5, 8]
    sns.boxplot(x='type', y='price_change_rate', data=pcg_cache, order=['rise', 'fall'], palette=[red, green])
    plt.text(1, pcg_cache.groupby('type')['price_change_rate'].median()[0], 
                round(pcg_cache.groupby('type')['price_change_rate'].median()[0], 2), 
                ha='center', va='bottom', fontsize=12)
    plt.text(0, pcg_cache.groupby('type')['price_change_rate'].median()[1], 
                round(pcg_cache.groupby('type')['price_change_rate'].median()[1], 2), 
                ha='center', va='bottom', fontsize=12)
    plt.show()

    plt.rcParams['figure.figsize'] = [7,5]
    sns.histplot(data=pcg_cache, x='price_change_rate', bins=40, hue='type', palette=[green, red])
    plt.show()

    pcg_cache['price_change_rate'] = abs(pcg_cache['price_change_rate'])
    plt.rcParams['figure.figsize'] = [5, 8]
    sns.boxplot(x='type', y='price_change_rate', data=pcg_cache, order=['rise', 'fall'], palette=[red, green])
    for i in range(len(pcg_cache.groupby('type'))):
        plt.text(1-i, pcg_cache.groupby('type')['price_change_rate'].median()[i], 
                round(pcg_cache.groupby('type')['price_change_rate'].median()[i], 2), 
                ha='center', va='bottom', fontsize=12)
    plt.show()

    plt.rcParams['figure.figsize'] = [7,5]
    sns.histplot(data=pcg_cache, x='price_change_rate', bins=40, hue='type', palette=[green, red])
    plt.show()
