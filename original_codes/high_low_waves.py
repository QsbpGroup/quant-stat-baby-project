
# ---------------------------------------------------------------------------- #
# *                   在两个金叉间找到high point， 两个死叉间找到low point           #
# *                           MACD定义：有两条线，DIFF和DEA                       #
# *          DIFF是短期EMA(12日指数滑动平均)减去EMA26，DEA是DIFF的9日均线             #
# ---------------------------------------------------------------------------- #
## 
import os
import pandas as pd
import matplotlib.pyplot as plt
import talib
import numpy as np
from matplotlib.legend_handler import HandlerLine2D, HandlerPathCollection
import matplotlib.lines as mlines
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# 获取股票数据文件夹的路径
data_directory = '/Users/zhaochenxi/Desktop/quant/data_csv1'
# 设置结果保存文件夹的路径
results_directory = '/Users/zhaochenxi/Desktop/quant/data_csv_results1'

# 创建结果保存文件夹
if not os.path.exists(results_directory):
    os.makedirs(results_directory)

# 遍历目录中的所有文件
for filename in tqdm(os.listdir(data_directory)):
    if filename.endswith('.csv'):
        # 构建完整的文件路径
        #filename='000001.SZ.csv'
        file_path = os.path.join(data_directory, filename)
        
        # 读取数据并创建DataFrame
        data = pd.read_csv(file_path)
        df = pd.DataFrame(data)
        
        # 转换日期列为日期时间类型
        df['TRADE_DT'] = pd.to_datetime(df['TRADE_DT'], format='%Y%m%d')
        
        
        #找出MACD
        macd,macd_signal,_=talib.MACD(df['S_DQ_CLOSE'].values)
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        
        #找出金叉和死叉
        df['golden_cross'] = np.where((df['macd'] > df['macd_signal']) & (df['macd'].shift() < df['macd_signal'].shift()), 1, 0)
        df['death_cross'] = np.where((df['macd'] < df['macd_signal']) & (df['macd'].shift() > df['macd_signal'].shift()), 1, 0)
        
        #选出金叉和死叉
        df1=df[(df['golden_cross']==1)|(df['death_cross']==1)]
        df1['shift_date']=df1['TRADE_DT'].shift(-1)
        df1=df1.dropna(subset=['shift_date'])
        df2=pd.DataFrame()
        df5=pd.DataFrame()
        for index,row in df1.iterrows():
            df3=df[(df['TRADE_DT']>=row['TRADE_DT'])&(df['TRADE_DT']<=row['shift_date'])]
            if df3.iloc[0]['golden_cross']==1:
                df4=df3[df3['S_DQ_CLOSE'].values==df3['S_DQ_CLOSE'].max()]
                df4=df4.head(1)
                df2=pd.concat([df2,df4])
            elif df3.iloc[0]['death_cross']==1:
                df4=df3[df3['S_DQ_CLOSE'].values==df3['S_DQ_CLOSE'].min()]
                df4=df4.head(1)
                df5=pd.concat([df5,df4])
        
        
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
        '''
        #峰值曲线的谷值，找低点
        peaks_valleys = [] 
         
        for i in range(2,len(peaks)-1):
            if peaks[i-1]['price'] < peaks[i-2]['price'] and peaks[i]['price'] < peaks[i-1]['price'] and peaks[i]['price'] < peaks[i+1]['price']:
                peaks_valleys.append({'date': peaks[i]['date'], 'price': peaks[i]['price']})
        
        low_points = []
        for pv in peaks_valleys:
            low_date_front = pv['date']
            low_index = valleys.index(next(v for v in valleys if v['date'] > low_date_front))
            low_price = valleys[low_index]['price']
            low_date = valleys[low_index]['date']
            low_points.append({'low_date': low_date, 'low_price': low_price})
            
        #谷值曲线的峰值，找高点
        valleys_peaks = [] 
         
        for i in range(3,len(valleys)-3):
            if valleys[i]['price'] > valleys[i-2]['price'] and valleys[i]['price'] > valleys[i-1]['price'] and valleys[i]['price'] > valleys[i+1]['price'] and valleys[i]['price'] > valleys[i+2]['price']:
                valleys_peaks.append({'date': valleys[i]['date'], 'price': valleys[i]['price']})
        
        high_points = []
        for vp in valleys_peaks:
            high_date_front = vp['date']
            high_index = peaks.index(next(p for p in peaks if p['date'] > high_date_front))
            #print(high_date_front,high_index)
            #break
            high_price = peaks[high_index]['price']
            high_date = peaks[high_index]['date']
            high_points.append({'high_date': high_date, 'high_price': high_price})
            
         '''    

        # 将结果保存到CSV文件（指定编码为UTF-8）
        output_filename = os.path.splitext(filename)[0] + '_result.csv'
        output_file_path = os.path.join(results_directory, output_filename)
        result_df = pd.DataFrame(result_data)
        result_df.to_csv(output_file_path, index=False, encoding='utf-8-sig')
        
        print(output_filename)
        # 获取最后100天的数据
        last_hundred_days_df = df.tail(200)
        #last_hundred_days_df = df[1520:1550]
    
        # 创建包含两个子图的图形
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True,gridspec_kw={'height_ratios': [3, 1]})
        # 绘制折线图
        ax1.plot(last_hundred_days_df['TRADE_DT'], last_hundred_days_df['S_DQ_CLOSE'], color='royalblue',alpha=0.8)
        
        # 将last_hundred_days_df['TRADE_DT']转换为与peaks中日期格式相同的字符串格式
        last_hundred_days_dates = last_hundred_days_df['TRADE_DT'].dt.strftime('%Y-%m-%d')
        # 提取最后100天内的峰值和谷值
        last_hundred_days_peaks = [peak for peak in peaks if peak['date'].strftime('%Y-%m-%d') in last_hundred_days_dates.values]
        last_hundred_days_valleys = [valley for valley in valleys if valley['date'].strftime('%Y-%m-%d') in last_hundred_days_dates.values]
        
        # 提取最后100天内的高点和低点
        #last_hundred_days_high = [high_point for high_point in high_points if high_point['high_date'].strftime('%Y-%m-%d') in last_hundred_days_dates.values]
        #last_hundred_days_low = [low_point for low_point in low_points if low_point['low_date'].strftime('%Y-%m-%d') in last_hundred_days_dates.values]

        # 提取最后100天内的高点和低点
        last_hundred_days_high1=[]
        for index,row in df2.iterrows():
            last_hundred_days_high1.append({'high_date': row['TRADE_DT'], 'high_price': row['S_DQ_CLOSE']})


        last_hundred_days_low1=[]
        for index,row in df5.iterrows():
            last_hundred_days_low1.append({'low_date': row['TRADE_DT'], 'low_price': row['S_DQ_CLOSE']})

        last_hundred_days_high2 = [high_point1 for high_point1 in last_hundred_days_high1 if high_point1['high_date'].strftime('%Y-%m-%d') in last_hundred_days_dates.values]
        last_hundred_days_low2 = [high_point1 for high_point1 in last_hundred_days_low1 if high_point1['low_date'].strftime('%Y-%m-%d') in last_hundred_days_dates.values]



        # 标记峰值和谷值
        for peak in last_hundred_days_peaks:
            line2=ax1.scatter(peak['date'], peak['price'], color='red', marker='^', label='Peak',alpha=0.3)
        for valley in last_hundred_days_valleys:
            line3=ax1.scatter(valley['date'], valley['price'], color='green', marker='v', label='Valley',alpha=0.3)
        # 标记高点和低点
        for high_point in last_hundred_days_high2:
            line4=ax1.scatter(high_point['high_date'], high_point['high_price'], color='red', marker='*', label='high',s=80)
        for low_point in last_hundred_days_low2:
            line5=ax1.scatter(low_point['low_date'], low_point['low_price'], color='green', marker='*', label='low',s=80)
            

        l1 = mlines.Line2D([], [], color='royalblue', marker='.',
                          markersize=5, label='Stock Price')
        l2 = mlines.Line2D([], [], color='red', marker='^',
                          markersize=8, label='Peak',alpha=0.3)     
        l3 = mlines.Line2D([], [], color='green', marker='v',
                          markersize=8, label='Valley',alpha=0.3)      
        l4 = mlines.Line2D([], [], color='red', marker='*',
                          markersize=10, label='high')
        l5 = mlines.Line2D([], [], color='green', marker='*',
                          markersize=10, label='low')
        # 设置图形标题和标签
        #plt.legend(labels=['Stock Price', 'Peak', 'Valley','High',"Low"])
        ax1.legend(handles=[l1,l2,l3,l4,l5],fontsize="x-small")
        ax1.set_title('Stock Price')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price')
        #ax1.xticks(rotation=45)
        # 显示图形
        
        # 绘制MACD曲线

        # 绘制折线图
        ax2.plot(last_hundred_days_df['TRADE_DT'], last_hundred_days_df['macd'], color='royalblue',alpha=0.8)
        #ax2.plot(df['TRADE_DT'], df['macd'], color='blue', label='MACD')
        ax2.plot(last_hundred_days_df['TRADE_DT'], last_hundred_days_df['macd_signal'], color='red', label='MACD Signal')

        # 设置子图2的标题和标签
        ax2.set_title('MACD and MACD Signal')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('MACD')

        # 调整子图之间的间距
        plt.subplots_adjust(hspace=0.3)

        # 显示图形
        plt.show()

        # 计算高点和低点的时间间隔和涨幅
        result_data2 = []
        for i in range(min(len(df2),len(df5))):
            high_date = df2.iloc[i]['TRADE_DT']
            low_date = df5.iloc[i]['TRADE_DT']
            high_price = df2.iloc[i]['S_DQ_CLOSE']
            low_price = df5.iloc[i]['S_DQ_CLOSE']
            if df2.iloc[0]['TRADE_DT'] < df5.iloc[0]['TRADE_DT']:
                interval = (low_date - high_date).days
                price_change = (low_price - high_price) / high_price * 100
                result_data2.append({'高点日期': high_date, '低点日期': low_date, '时间间隔（天）': interval, '变化': price_change})
            else:
                interval = (high_date - low_date).days
                price_change = (high_price - low_price) / low_price * 100
                result_data2.append({ '低点日期': low_date,'高点日期': high_date, '时间间隔（天）': interval, '变化': price_change})
    
        # 将结果保存到DataFrame
        result_df = pd.DataFrame(result_data2)
        
        # 打印结果
        print(result_df)
        
        #统计波数和每波反弹幅度
        
        #high曲线的谷值，找低点
        highs_valleys = [] 
         
        for i in range(2,len(df2)-1):
            if df2.iloc[i]['S_DQ_CLOSE'] < df2.iloc[i-1]['S_DQ_CLOSE'] and df2.iloc[i]['S_DQ_CLOSE'] < df2.iloc[i+1]['S_DQ_CLOSE']:
                highs_valleys.append({'date': df2.iloc[i]['TRADE_DT'], 'price': df2.iloc[i]['S_DQ_CLOSE']})
        #下降的波数
        waves_fall = len(highs_valleys)

        #low曲线的峰值，找高点
        lows_peaks = [] 
         
        for i in range(2,len(df5)-1):
            if df5.iloc[i]['S_DQ_CLOSE'] > df5.iloc[i-1]['S_DQ_CLOSE'] and df5.iloc[i]['S_DQ_CLOSE'] > df5.iloc[i+1]['S_DQ_CLOSE']:
                lows_peaks.append({'date': df5.iloc[i]['TRADE_DT'], 'price': df5.iloc[i]['S_DQ_CLOSE']})
        #上升的波数
        waves_rise =len(lows_peaks)


        lower_high_count = 0
        satisfied_highs_count = 0  # 计数满足条件的高点数量
        for i in range(1, len(df2)):      
            # 判断当前high点是否比上一个high点低
            if df2.iloc[i]['S_DQ_CLOSE'] < df2.iloc[i-1]['S_DQ_CLOSE']:
                # 找到当前high点前的low点
                low_points_before_high = df5[df5['S_DQ_CLOSE'] < df2.iloc[i]['S_DQ_CLOSE']]
                if len(low_points_before_high) > 0:
                    lower_high_count +=1
                    # 计算当前high点前的low点到当前high点的涨幅
                    low_price = low_points_before_high['S_DQ_CLOSE'].iloc[-1]
                    high_price = df2.iloc[i]['S_DQ_CLOSE']
                    price_change = (high_price - low_price)
                    
                    
                    # 计算上一个high点到上一个low点的跌幅的一半
                    price_change_prev = (df2.iloc[i-1]['S_DQ_CLOSE'] - low_price)
                    # 判断涨幅是否达到了跌幅的一半
                    price_change_prev_half = price_change_prev / 2
                    if (price_change <= price_change_prev_half):
                        satisfied_highs_count += 1

        percentage = (satisfied_highs_count / lower_high_count) * 100
        
        print("下跌次数：",waves_fall)
        print("下跌波数均值：",lower_high_count/waves_fall)
        print("满足条件的highs占比：", percentage, "%")

        
        higher_low_count = 0
        satisfied_lows_count = 0  # 计数满足条件的高点数量
        for i in range(1, len(df5)):      
            # 判断当前low点是否比上一个low点高
            if df5.iloc[i]['S_DQ_CLOSE'] > df5.iloc[i-1]['S_DQ_CLOSE']:
                # 找到当前low点前的high点
                high_points_before_low = df2[df2['S_DQ_CLOSE'] < df5.iloc[i]['S_DQ_CLOSE']]
                if len(high_points_before_low) > 0:
                    higher_low_count +=1
                    # 计算当前low点前的high点到当前low点的跌幅
                    high_price = high_points_before_low['S_DQ_CLOSE'].iloc[-1]
                    low_price = df5.iloc[i]['S_DQ_CLOSE']
                    price_change = (high_price - low_price)
                    
                    # 计算上一个high点到上一个low点的跌幅的一半
                    price_change_prev = (df5.iloc[i-1]['S_DQ_CLOSE'] - high_price)
                    # 判断涨幅是否达到了跌幅的一半
                    price_change_prev_half = price_change_prev / 2
                    if (price_change <= price_change_prev_half):
                        satisfied_lows_count += 1

        percentage_lows = (satisfied_lows_count / higher_low_count) * 100
        
        print("上涨次数：",waves_rise)
        print("上涨波数均值：",higher_low_count/waves_rise)
        print("满足条件的lows占比：", percentage_lows, "%")

