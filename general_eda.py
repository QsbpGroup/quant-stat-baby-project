import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def hist_plot(df, data_col, hue='', bins=40, quantile=1, absolute=False):
    """
    This function plots a histogram of the data_col in the df dataframe.
    The hue parameter is used to color the histogram based on the hue column.
    The quantile parameter is used to filter the data_col column based on the quantile value.
    The absolute parameter is used to take the absolute value of the data_col column.
    Bonus: if the hue column has only {'rise', 'fall'}, the histogram is colored red and green.
    
    Example:
    --------
    >>> hist_plot(df, 'price', 'wave_type', quantile=0.95, absolute=True)
    """
    sns.set_style("darkgrid")

    quantiled_df = df[(df[data_col] < df[data_col].quantile(quantile)) & (df[data_col] > df[data_col].quantile(1-quantile))].copy()
    if absolute:
        quantiled_df[data_col] = quantiled_df[data_col].abs()

    plt.figure(figsize=(7, 5))
    if hue == '':
        sns.histplot(data=quantiled_df, x=data_col, bins=bins, kde=True)
        plt.show()
        return
    if set(df[hue]) == {'rise', 'fall'}:
        sns.histplot(data=quantiled_df, x=data_col, hue=hue, bins=bins, kde=True, palette=['red', 'green'])
    else:
        sns.histplot(data=quantiled_df, x=data_col, hue=hue, bins=bins, kde=True)
    plt.show()
    return


def box_plot(df, y, x='', quantile=1, absolute=False):
    """
    This function plots a boxplot of the data_col in the df dataframe.
    The hue parameter is used to color the boxplot based on the hue column.
    The quantile parameter is used to filter the data_col column based on the quantile value.
    The absolute parameter is used to take the absolute value of the data_col column.
    Bonus: if the hue column has only {'rise', 'fall'}, the boxplot is colored red and green.
    
    Example:
    --------
    >>> box_plot(df, 'price', 'wave_type', quantile=0.95, absolute=True)
    """
    sns.set_style("darkgrid")

    quantiled_df = df[(df[y] < df[y].quantile(quantile)) & (df[y] > df[y].quantile(1-quantile))].copy()
    if absolute:
        quantiled_df[y] = quantiled_df[y].abs()

    if x == '':
        ax = sns.boxplot(data=quantiled_df, x=y, width=0.6)
        ax.set_title(y, fontsize=14)
        ax.set_xlabel('')
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.show()
        return

    plt.figure(figsize=(4, 7))
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    if set(df[x]) == {'rise', 'fall'}:
        ax = sns.boxplot(data=quantiled_df, x=x, y=y, palette=["#fd5a45", "#3bc66d"])
        ax.set_title(y, fontsize=16, loc='left')
        ax.set_xlabel('')
        ax.set_ylabel('')
    else:
        sns.boxplot(data=quantiled_df, x=x, y=y)
    for i in range(len(quantiled_df.groupby(x))):
        plt.text(1-i, quantiled_df.groupby(x)[y].median()[i], 
                round(quantiled_df.groupby(x)[y].median()[i], 2), 
                ha='center', va='bottom', fontsize=14)
    plt.show()
    return