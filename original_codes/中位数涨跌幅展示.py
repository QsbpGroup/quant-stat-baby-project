import pandas as pd
import horizontal_area as ha
import high_low_xuejie_zuhui as hl
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os

count = 0
cr_all = pd.DataFrame()
data_directory = '/Users/kai/Desktop/qs/data_csv_distinct_0606'
for filename in tqdm(os.listdir(data_directory)):
    if filename.endswith('.csv') & (count<100):
        df = hl.df_init(filename)

        high_points, low_points = hl.find_high_low(
            df, filename, save_data=False, draw=False)
        result = ha.find_ha_near_hl_median(df, high_points, low_points, 0.03)
        # ha.draw_horizontal_area(df, result, high_points, low_points, filename,
        #                         n_days=400, print_result=False, show_plot=True, save_plot=False)
        change_rate = pd.DataFrame(ha.cal_diff_rate_between_medians(result))
        if change_rate.empty:
            continue
        change_rate.columns = ['change_rate']
        change_rate['flag'] = change_rate['change_rate'].apply(lambda x: '上涨' if x >= 0 else '下跌')
        change_rate['change_rate'] = change_rate['change_rate']*100
        cr_all.append(change_rate)
        count+=1

# 重置index
cr_all.reset_index(drop=True, inplace=True)

plt.rcParams['figure.figsize'] = [5, 8]
plt.rcParams['font.sans-serif']=['Arial Unicode MS']
# 绘制boxplot并在图中每个box中位数上用文字写出数值
sns.boxplot(x='flag', y='change_rate', data=cr_all)
for i in range(len(cr_all.groupby('flag'))):
    plt.text(i, cr_all.groupby('flag')['change_rate'].median()[i], 
             round(cr_all.groupby('flag')['change_rate'].median()[i], 2), 
             ha='center', va='bottom', fontsize=12)
plt.show()
# 绘制histogram
plt.rcParams['figure.figsize'] = [7,5]
sns.histplot(data=cr_all, x='change_rate', hue='flag', bins=40)
plt.show()

cr_all_abs = cr_all.copy()
cr_all_abs['change_rate'] = cr_all_abs['change_rate'].apply(lambda x: abs(x))
plt.rcParams['figure.figsize'] = [5, 8]
plt.rcParams['font.sans-serif']=['Arial Unicode MS']
# 绘制boxplot并在图中每个box中位数上用文字写出数值
sns.boxplot(x='flag', y='change_rate', data=cr_all)
for i in range(len(cr_all.groupby('flag'))):
    plt.text(i, cr_all.groupby('flag')['change_rate'].median()[i], 
             round(cr_all.groupby('flag')['change_rate'].median()[i], 2), 
             ha='center', va='bottom', fontsize=12)
plt.show()
# 绘制histogram
plt.rcParams['figure.figsize'] = [7,5]
sns.histplot(data=cr_all, x='change_rate', hue='flag', bins=40)
plt.show()
