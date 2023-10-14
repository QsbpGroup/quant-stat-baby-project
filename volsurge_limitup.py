import pandas as pd
import VolumeSpikes as vs
import wave_price_change as wpc
import high_low_xuejie_zuhui as hl


def cal_wave_info(filename, vol_days):
    df = hl.df_init(filename)
    waves = wpc.wave_identify(filename)
    position = []
    for day in vol_days:
        for i in range(len(waves)):
            if waves['date'][i] >= day:
                position.append(i)
                break

    wave_length = [(waves['date'][x] - waves['date'][x-1]).days for x in position]
    future_length = [(waves['date'][x] - vol_days[y]).days for x,
                    y in zip(position, range(len(vol_days)))]
    wave_type = []
    price_change = []
    price_change_rate = []
    for i in range(len(vol_days)):
        # 判断当前处于什么周期，剩余周期长度过短说明此次放量涨停应当进入下一个周期
        if (future_length[i]/wave_length[i]) > 0.3:
            wave_type.append(waves['type'][position[i]])
            price_change.append(
                waves['price'][position[i]] - df['S_DQ_CLOSE'][days_index[i]])
        else:
            wave_type.append(waves['type'][position[i]+1])
            price_change.append(
                waves['price'][position[i]+1] - df['S_DQ_CLOSE'][days_index[i]])
        price_change_rate.append(price_change[i]/df['S_DQ_CLOSE'][days_index[i]])
    wave_type = [x[-4:] for x in wave_type]
    result = pd.DataFrame({'date': vol_days, 'wave_type': wave_type,
                        'price_change': price_change, 'price_change_rate': price_change_rate})
