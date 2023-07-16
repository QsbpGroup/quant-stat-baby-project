import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
data_short = pd.read_csv('/Users/kai/Desktop/qs/data_csv_distinct_0606/_horizontal_area_pics/results_short.csv')
data_short.columns = ['min_len_of_window', 'n', 'mean']
data_short.dropna(inplace=True)
data_long = pd.read_csv('/Users/kai/Desktop/qs/data_csv_distinct_0606/_horizontal_area_pics/results_long.csv')
data_long.columns = ['code', 'min_len_of_window', 'max_len_of_window', 'n', 'mean']
data_long = data_long[['min_len_of_window', 'n', 'mean']]
data_long.dropna(inplace=True)
# merge两个dataframe
data = pd.concat([data_short, data_long], axis=0)
# 对mean这一列画出min_len_of_window=6,18,65三种情况下的boxplot
plt.rcParams['figure.figsize'] = [5, 8]
sns.boxplot(x='min_len_of_window', y='mean', data=data)
plt.show()