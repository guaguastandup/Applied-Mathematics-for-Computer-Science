import pandas as pd
import numpy as np
import math
import csv 
def get_p(x, mean, var):
    return np.exp(-(x - mean) ** 2 / (2 * var ** 2)) / (math.sqrt(2 * math.pi) * var) 

dataset_train_reader = pd.read_csv('./dataset/kddcup99_test.csv', header = None, error_bad_lines=False, nrows = None, chunksize = 10000)
dataset_test = pd.read_csv('./dataset/kddcup.data_10_percent', header = None, error_bad_lines=False, nrows = None)

data_yes_cnt = {}  # cnt = data_yes_cnt[col_index][col_value]
data_no_cnt = {}   # cnt = data_no_cnt[col_index][col_value]
data_yes_value = {}  # cnt = data_yes_cnt[col_index][col_value]
data_no_value = {}   # cnt = data_no_cnt[col_index][col_value]
for i in range(42):
    data_yes_cnt[i] = {}
    data_no_cnt[i] = {}
    data_yes_value[i] = []
    data_no_value[i] = []

yes_n, no_n = 0, 0
print("ok~~")

mean_yes, var_yes, mean_no, var_no = {}, {}, {}, {}

for i in range(42):
    if i!=1 and i!=2 and i!=3:
        mean_yes[i] = 0
        var_yes[i]  = 0
        mean_no[i]  = 0
        var_no[i]   = 0


# 离散值统计
# get avg
n = 0
cnt = 0
for dataset_train in dataset_train_reader:
    for line, row in dataset_train.iterrows(): 
        n += 1
        label = row[41]
        if label == "normal.": # yes
            yes_n += 1
            for i in range(41):
                val = row[i]
                if not data_yes_cnt[i].get(val):
                    data_yes_cnt[i][val] = 0
                data_yes_cnt[i][val] += 1
                if i!=1 and i!=2 and i!=3:
                    mean_yes[i] += float(val)
        if label != "normal.": # no
            no_n += 1
            for i in range(41):
                val = row[i]
                if not data_no_cnt[i].get(val):
                    data_no_cnt[i][val] = 0
                data_no_cnt[i][val] += 1
                if i!=1 and i!=2 and i!=3:
                    mean_no[i] += float(val)
print("avg ok")
# get cov
dataset_train_reader = pd.read_csv('./dataset/kddcup99_train.csv', header = None, error_bad_lines=False, nrows = None, chunksize = 10000)
for dataset_train in dataset_train_reader:
    for line, row in dataset_train.iterrows(): 
        label = row[41]
        if label == "normal.": # yes
            for i in range(41):
                val = row[i]
                if i!=1 and i!=2 and i!=3:
                    var_yes[i] += (float(val) - mean_yes[i])*(float(val) - mean_yes[i])
        if label != "normal.": # no
            for i in range(41):
                val = row[i]
                if i!=1 and i!=2 and i!=3:
                    var_no[i] += (float(val) - mean_no[i])*(float(val) - mean_no[i])
# yes or no
print("yes_n: ", yes_n)
print("no_n: ", no_n)
print("n: ", n)

for i in range(42):
    if i!=1 and i!=2 and i!=3:
        var_yes[i]  = math.sqrt(var_yes[i] / (n - 1))
        var_no[i]   = math.sqrt(var_no[i] / (n - 1))

ok, no = 0, 0
predict = []
TP, FP, FN, TN = 0, 0, 0, 0

for line, row in dataset_test.iterrows(): 
    label = row[41]
    p_yes, p_no = yes_n/(yes_n + no_n), no_n/(yes_n + no_n)
    flag = 0
    for i in range(41):
        if i!=1 and i!=2 and i!=3:
            if var_yes[i]!=0:
                p_yes_temp = get_p(row[i], mean_yes[i], var_yes[i])
            else:
                p_yes_temp = 1
            if var_no[i]!=0:
                p_no_temp = get_p(row[i], mean_no[i], var_no[i])
            else:
                p_no_temp = 1
        else:
            if not data_yes_cnt[i].get(row[i]):
                p_yes_temp = 1.0*1 / (yes_n + 1)
            else: 
                p_yes_temp = (data_yes_cnt[i][row[i]] + 1) / (yes_n + 1)
            if not data_no_cnt[i].get(row[i]): 
                flag = 1
                p_no_temp = 1 / (no_n + 1)
            else: 
                p_no_temp  = (data_no_cnt[i][row[i]] + 1) / (no_n + 1)
        p_yes *= p_yes_temp
        p_no *= p_no_temp
    if p_yes > p_no:
        if label == "normal.": 
            ok += 1
            TP += 1
        else: 
            no += 1
            FP += 1
    else:
        if label != "normal.": 
            no += 1
            FN += 1
        else: 
            ok += 1
            TN += 1
    
print(ok, no)
precision = TP / (TP + FP)
accuracy = (TP+TN) / (TP + TN + FP + FN)
# f1 = (2 * precision * recall) / (precision + recall)
print("accuracy: ", str(100 * accuracy)+"%")
print("precision:", str(100 * precision)+"%")