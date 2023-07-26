import pandas as pd
from high_low_xuejie_zuhui import find_hl_MACD_robust, df_init


def wave_idnetify(filename='000001.SZ.csv', alpha=0.05):
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
    real_waves : pd.DataFrame
        columns=['date', 'price', 'type'], type can be 'edo_fall'(a low point) and 'edo_rise'(a high point).
    """

    df = df_init(filename)
    high_points, low_points = find_hl_MACD_robust(
        df, filename, draw=False, fig_start_date='2019-7', fig_end_date='2021-3')
    hd = [item['high_date'] for item in high_points]
    hp = [item['high_price'] for item in high_points]
    ld = [item['low_date'] for item in low_points]
    lp = [item['low_price'] for item in low_points]
    h_cgdt = []
    l_cgdt = []
    for i in range(2, len(hd)-1):
        if hp[i+1] > hp[i] and hp[i-1] > hp[i]:
            h_cgdt.append(
                {'date': hd[i], 'price': hp[i], 'pcg': max(hp[i+1], hp[i-1])-hp[i]})
        if lp[i+1] < lp[i] and lp[i-1] < lp[i]:
            l_cgdt.append(
                {'date': ld[i], 'price': lp[i], 'pcg': lp[i]-min(lp[i+1], lp[i-1])})
    h_cgdt = pd.DataFrame(h_cgdt)
    h_cgdt = h_cgdt[h_cgdt['pcg'] > h_cgdt['pcg'].quantile(alpha)]
    h_cgdt['type'] = 'edo_fall'
    l_cgdt = pd.DataFrame(l_cgdt)
    l_cgdt = l_cgdt[l_cgdt['pcg'] > l_cgdt['pcg'].quantile(alpha)]
    l_cgdt['type'] = 'edo_rise'
    all_cgdt = pd.concat([h_cgdt, l_cgdt])
    all_cgdt.sort_values(by=['date'], inplace=True)
    all_cgdt.reset_index(drop=True, inplace=True)
    all_cgdt['true_date'] = None
    type_lst = all_cgdt['type'].tolist()
    dt_lst = all_cgdt['date'].tolist()
    prs_lst = all_cgdt['price'].tolist()
    real_waves = pd.DataFrame(columns=['date', 'price', 'type'])
    h_df = pd.DataFrame(high_points)
    l_df = pd.DataFrame(low_points)
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
