from high_low_xuejie_zuhui import *
from horizontal_area import *
import os
from test import *

counter = 0
results = []
os.chdir('/Users/kai/Desktop/qs/data_csv_distinct_0606')
for filename in os.listdir():
    if counter > 99:
        break
    # filename = '000001.SZ.csv'
    os.chdir('/Users/kai/Desktop/qs/data_csv_distinct_0606/_horizontal_area_pics')
    df = df_init(filename)
    peaks, valleys, high_points, low_points = find_high_low_xuejieZuhui(
        df, filename, save_data=False, draw_n_days=200, draw=False)

    for max_len_of_window in (100,):
        for min_len_of_window in (65,):
            gamma = 0.3  # 最大变化率
            view_coe = 0.8  # 视野·系数

            result = find_horizontal_area(df, high_points, low_points, max_len_of_window,
                                        min_len_of_window, gamma, view_coe, ignore_hl=True)
            temp_n, temp_mean = len(result), result['interval'].mean()
            if pd.isna(temp_mean):
                print('temp_mean is nan.')
            else:
                print(filename)
                temp_mean = round(temp_mean, 2)
                results.append([filename, min_len_of_window, max_len_of_window, temp_n, temp_mean])
                # print(results)
                pd.DataFrame(results).to_csv('results_long.csv', index=False)

    # n_days = 300
    # draw_horizontal_area(df, result, peaks, valleys, high_points,
    #                     low_points, filename, n_days, print_result=False, show_plot=False)
    os.chdir('/Users/kai/Desktop/qs/data_csv_distinct_0606')
    counter += 1
