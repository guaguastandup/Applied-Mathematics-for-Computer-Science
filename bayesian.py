import pandas as pd
import numpy as np
import math

def get_p(x, mean, var):
    return np.exp(-(x - mean) ** 2 / (2 * var ** 2)) / (math.sqrt(2 * math.pi) * var) 

dataset_train = pd.read_csv('./dataset/Bayesian_Dataset_train.csv', header = None)
dataset_test = pd.read_csv('./dataset/Bayesian_Dataset_test.csv', header = None)

data_yes_cnt = {}  # cnt = data_yes_cnt[col_index][col_value]
data_no_cnt = {}   # cnt = data_no_cnt[col_index][col_value]
data_yes_value = {}  # cnt = data_yes_cnt[col_index][col_value]
data_no_value = {}   # cnt = data_no_cnt[col_index][col_value]

for i in range(11):
    data_yes_cnt[i] = {}
    data_no_cnt[i] = {}
    data_yes_value[i] = []
    data_no_value[i] = []

yes_n, no_n = 0, 0
train_n = dataset_train.shape[0]
test_n = dataset_test.shape[0]
print("train_n: ", train_n)
print("test_n: ", test_n)

# 离散值统计
for line, row in dataset_train.iterrows(): 
    label = row[10]
    if label == " >50K": # yes
        yes_n += 1
        for i in range(10):
            val = row[i]
            if not data_yes_cnt[i].get(val):
                data_yes_cnt[i][val] = 0
            data_yes_cnt[i][val] += 1
            if i == 0 or i == 2:
                data_yes_value[i].append(int(val))
    if label == " <=50K": # no
        no_n += 1
        for i in range(10):
            val = row[i]
            if not data_no_cnt[i].get(val):
                data_no_cnt[i][val] = 0
            data_no_cnt[i][val] += 1
            if i == 0 or i == 2:
                data_no_value[i].append(int(val))
# yes or no
print("yes_n: ", yes_n)
print("no_n: ", no_n)
mean_yes, var_yes, mean_no, var_no = {}, {}, {}, {}

for i in range(3):
    if i == 0 or i == 2:
        mean_yes[i] = np.mean(data_yes_value[i])
        var_yes[i]  = math.sqrt(np.cov(data_yes_value[i]))
        mean_no[i]  = np.mean(data_no_value[i])
        var_no[i]   = math.sqrt(np.cov(data_no_value[i]))

ok, no = 0, 0
predict = []
TP, FP, FN, TN = 0, 0, 0, 0
dataset_test = pd.concat([dataset_test, pd.DataFrame(columns = ['predict'])])

for line, row in dataset_test.iterrows(): 
    label = row[10]
    p_yes, p_no = yes_n/(yes_n + no_n), no_n/(yes_n + no_n)
    flag = 0
    for i in range(10):
        if i == 0 or i == 2:
            p_yes_temp = get_p(row[i], mean_yes[i], var_yes[i])
            p_no_temp = get_p(row[i], mean_no[i], var_no[i])
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
        dataset_test['predict'][line] = " >50K"
        if label == " >50K": 
            ok += 1
            TP += 1
        else: 
            no += 1
            FP += 1
    else:
        dataset_test['predict'][line] = " <=50K"
        if label == " >50K": 
            no += 1
            FN += 1
        else: 
            ok += 1
            TN += 1
    
print(ok, no)
recall = TP / (TP + FN)
precision = TP / (TP + FP)
accuracy = (TP+TN) / (TP + TN + FP + FN)
f1 = (2 * precision * recall) / (precision + recall)
print("accuracy: ", str(100 * accuracy)+"%")
print("precision:", str(100 * precision)+"%")
print("recall:   ", str(100 * recall)+"%")
print("F1 score: ", str(100 * f1)+"%")
dataset_test.to_csv("my.csv", index=False, header=False)