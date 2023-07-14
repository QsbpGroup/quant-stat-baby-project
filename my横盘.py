from high_low_xuejieZuhui import *
from horizontal_area import *
import os


os.chdir('/Users/kai/Desktop/qs/data_csv_distinct_0606')
for filename in os.listdir():
    # filename = '000001.SZ.csv'
    os.chdir('/Users/kai/Desktop/qs/data_csv_distinct_0606/_horizontal_area_pics')
    df = df_init(filename)
    peaks, valleys, high_points, low_points = find_high_low_xuejieZuhui(
        df, filename, save_data=False, draw_n_days=200, draw=False)

    max_len_of_window = 40
    min_len_of_window = 12
    gamma = 0.3  # 最大变化率
    view_coe = 0.8  # 视野·系数

    result = find_horizontal_area(df, high_points, low_points, max_len_of_window,
                                min_len_of_window, gamma, view_coe, ignore_hl=True)

    n_days = 300
    draw_horizontal_area(df, result, peaks, valleys, high_points,
                        low_points, filename, n_days, print_result=False, show_plot=False)
    os.chdir('/Users/kai/Desktop/qs/data_csv_distinct_0606')
