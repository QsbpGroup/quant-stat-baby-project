import os
import sys
import cx_Oracle 
import pandas as pd

# 添加 oracle 客户端路径
sys.path.append('/Users/kai/instantclient_19_8')

intern = cx_Oracle.makedsn('119.147.213.115', '1522', service_name='wktzorcl')

def queryOracle(sql):
    # 创建连接
    conn = cx_Oracle.connect(user='wktz_intern01', password='wktz_intern01', dsn=intern)
    # 创建游标
    curs = conn.cursor()
    # 执行查询
    curs.execute(sql)
    # 返回查询结果
    data = curs.fetchall()
    # 获取列名
    columns = [i[0] for i in curs.description]
    # 将结果转换为DataFrame格式
    df = pd.DataFrame(data, columns=columns)
    # 关闭游标和连接
    curs.close()
    conn.close()
    
    return df

# make a new folder in pwd and move into it
os.mkdir('data')
os.chdir('data')

# 1. 获取所有股票代码
sql_get_all_code = '''
SELECT distinct(s_info_windcode) FROM WKWD_SYNC.ashareeodprices
'''
windcodes = queryOracle(sql_get_all_code)

# 2. 获取所有股票的数据 并保存为excel

for i in windcodes['S_INFO_WINDCODE']:
    sql_temp = '''
    select * from WKWD_SYNC.ashareeodprices where s_info_windcode = '{}' 
    order by trade_dt 
    '''.format(i)
    df_temp = queryOracle(sql_temp)
    # write a csv in the name of i
    df_temp.to_csv('{}.csv'.format(i), index=False)
