import pandas as pd
from math import sqrt
# scikit-learn 朴素贝叶斯\高斯混合模型\
# sknetwork中有pagerank函数
# mean
# p - value
# confidence interval
data = pd.read_csv('./dataset/AB_Test_Dataset.csv')
data_user = {}
data_variant = {}
user_list = []
for line, row in data.iterrows(): 
    user_id = row['USER_ID']
    variant_name = row['VARIANT_NAME']
    revenue = row['REVENUE']
    if not data_user.get(user_id):
        data_user[user_id] = {}
        data_variant[user_id] = {}
        user_list.append(user_id) 
    if not data_user[user_id].get(variant_name):
        data_user[user_id][variant_name] = 0.0
    data_variant[user_id][variant_name] = 1
    data_user[user_id][variant_name] += float(revenue)
# find invalid user
invalid_user = []
for user_id in user_list:
    if 'control' in data_variant[user_id] and 'variant' in data_variant[user_id] and data_variant[user_id]['control'] == 1 and data_variant[user_id]['variant'] == 1:
        invalid_user.append(user_id)
data_list = []
c_cnt, t_cnt = 0, 0
c_sum, t_sum = 0, 0
for k, item in data_user.items():
    user_id = k
    if user_id in invalid_user:
        continue
    if 'control' in data_variant[user_id] and data_variant[user_id]['control'] == 1:
        variant_name = 'control'
        c_cnt += 1
    if 'variant' in data_variant[user_id] and data_variant[user_id]['variant'] == 1:
        variant_name = 'variant'
        t_cnt += 1
    revenue = item[variant_name]
    if variant_name == 'control':
        c_sum += revenue
    else:
        t_sum += revenue
    data_list.append([user_id, variant_name, revenue])
df = pd.DataFrame(data_list, columns = ['USER_ID', 'VARIANT_NAME', 'REVENUE'])
# 长度
n = len(df)
print("n", n)
print(df.shape)
print(df.dtypes)
# P-value描述的是随机性
# 虚无假设（Null hypothesis）为真的情况下，得到该结果的概率，即在treatment和control没有区别的情况下（A/A实验）得到该数据的概率。
t_mean = t_sum / t_cnt
c_mean = c_sum / c_cnt
t_var = 0.0
for data in data_list:
    if data[1] == 'variant':
        t_var += (float(data[2]) - t_mean) * (float(data[2]) - t_mean)
t_var /= t_cnt
print("t_var", t_var)
t_sx = sqrt(t_var) / sqrt(t_cnt)
print(t_sx)
alpha = 0.05
z2 = 1.96
l = t_mean - z2 * t_sx
r = t_mean + z2 * t_sx
print("l: ", l, ",r: ", r)

import seaborn as sns
