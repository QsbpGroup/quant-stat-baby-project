import os
import pandas as pd
from tqdm import tqdm
import VolumeSpikes as vs
import wave_price_change as wpc
import high_low_xuejie_zuhui as hl

def find_volsurge_limitup_days(csv_file_name):
    # 读取CSV文件的股票数据
    df = pd.read_csv(csv_file_name)

    # 计算平均成交量，采用过去30天的历史数据
    df['average_volume'] = df['S_DQ_VOLUME'].rolling(window=30).mean()

    # 初始化结果列表
    volsurge_limitup_days = []

    # 将日期列转换为从1开始的数组
    date_array = list(range(1, len(df) + 1))
    df['date_array'] = date_array

    # 查找放量涨停日
    # 需满足三个条件（不确定）：
    # 1.当日成交量明显高于平均成交量，设置为了1.5倍
    # 2.当日的涨跌幅达到或超过10%
    # 3.当日最高价等于涨停价，设置成了大于等于前一交易日的收盘价乘以1.1
    for index, row in df.iterrows():
        if row[ 'S_DQ_VOLUME'] > 1.5 * row['average_volume'] \
                and ((row['S_DQ_CLOSE'] - row['S_DQ_OPEN']) / row['S_DQ_OPEN']) * 100 >= 10 \
                and row['S_DQ_HIGH'] >= df.at[index - 1, 'S_DQ_CLOSE'] * 1.1:
            volsurge_limitup_days.append(row['date_array'])

    # 删除日期数组列
    df.drop('date_array', axis=1, inplace=True)

    # 合并短时间内多个放量涨停
    if len(volsurge_limitup_days) == 0:
        return []
    volsurge_limitup_days_merged = [volsurge_limitup_days[0]]
    for i in range(1, len(volsurge_limitup_days)):
        if volsurge_limitup_days[i] - volsurge_limitup_days[i - 1] >3:
            volsurge_limitup_days_merged.append(volsurge_limitup_days[i]) #即将连续子串合并为该子串的第一个元素


    return volsurge_limitup_days_merged


def cal_wave_info(filename, vol_days_index):
    """
    找到某只股票所有放量涨停后的波段，计算波段终点与放量涨停收盘价的价差信息
    
    Input:
    ----------------
    filename: str
        e.g. '000001.SZ.csv'
    vol_days_index: list
        e.g. [17, 29, 84, 31] 代表第17, 29, 84, 31个交易日是放量涨停日
    
    Output:
    ----------------
    result: DataFrame
        date: 放量涨停日
        wave_type: 波段类型
        price_change: 波段终点 - 放量涨停收盘价
        price_change_rate: (波段终点 - 放量涨停收盘价) / 放量涨停收盘价
    """
    
    
    df = hl.df_init(filename)
    vol_days_index = [x-1 if x==len(df) else x for x in vol_days_index]
    vol_days = df['TRADE_DT'].values[vol_days_index].tolist()
    waves = wpc.wave_identify(filename)
    position = []
    # 找到每个放量涨停日对应的波段终点
    for day in vol_days:
        for i in range(len(waves)):
            if waves['date'][i] >= pd.to_datetime(day):
                position.append(i)
                break
            elif i == len(waves) - 1:
                position.append(i)

    if len(position) == 0:
        return pd.DataFrame({'date': [], 'wave_type': [], 'price_change': [], 'price_change_rate': []})
    wave_length = []
    for i in position:
        if i == 0:
            wave_length.append((waves['date'][i] - pd.to_datetime(df['TRADE_DT'][0])).days)
        else:
            wave_length.append((waves['date'][i] - waves['date'][i-1]).days)
    future_length = [(waves['date'][x] - pd.to_datetime(vol_days[y])).days for x,
                    y in zip(position, range(len(vol_days)))]
    wave_type = []
    price_change = []
    price_change_rate = []
    pops = []
    for i in range(len(vol_days)):
        # 判断当前处于什么周期，剩余周期长度过短说明此次放量涨停应当进入下一个周期
        if (future_length[i]/wave_length[i]) > 0.3:
            wave_type.append(waves['type'][position[i]])
            price_change.append(
                waves['price'][position[i]] - df['S_DQ_CLOSE'][vol_days_index[i]])
        else:
            # 如果报错则continue，报错原因是放量涨停处于截止到今日（20230606）的周期中
            try:
                tmp_price_change = waves['price'][position[i]+1] - df['S_DQ_CLOSE'][vol_days_index[i]]
                tmp_wave_type = waves['type'][position[i]-1]
            except:
                # 将此日期从vol_days中删除
                pops.append(vol_days[i])
                continue
            price_change.append(tmp_price_change)
            wave_type.append(tmp_wave_type)
        price_change_rate.append(price_change[-1]/df['S_DQ_CLOSE'][vol_days_index[i]])
    wave_type = [x[-4:] for x in wave_type]
    for i in pops:
        vol_days.remove(i)
    result = pd.DataFrame({'date': vol_days, 'wave_type': wave_type,
                        'price_change': price_change, 'price_change_rate': price_change_rate})
    
    return result

if __name__ == '__main__':
    # 调用函数并读取数据
    results = pd.DataFrame(columns=['date', 'wave_type', 'price_change', 'price_change_rate'])
    for filename in tqdm(os.listdir('/Users/kai/Desktop/qs/data_csv_distinct_0606')):
        if filename[-4:] != '.csv':
            continue
        filename = '/Users/kai/Desktop/qs/data_csv_distinct_0606/' + filename
        volsurge_limitup_days = find_volsurge_limitup_days(filename)
        cals = cal_wave_info(filename, volsurge_limitup_days)
        results = results.append(cals, ignore_index=True)

    # 保存结果
    results.to_csv('volsurge_limitup_wave_info.csv', index=False)