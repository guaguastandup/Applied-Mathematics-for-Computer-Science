import numpy as np
import pandas as pd


dataset_test = pd.read_csv('./dataset/kddcup.data_10_percent', header = None, error_bad_lines=False, nrows = None)

dataset_test = dataset_test.drop(columns = 41)
print(dataset_test.head())

d1 = {}
d2 = {}
d3 = {}

cnt1 = 0
cnt2 = 0
cnt3 = 0

for line, row in dataset_test.iterrows(): 
    val1 = row[1]
    val2 = row[2]
    val3 = row[3]
    if not d1.get(val1):
        d1[val1] = cnt1 + 1
        cnt1 += 1
    if not d2.get(val2):
        d2[val2] = cnt2 + 1
        cnt2 += 1
    if not d3.get(val3):
        d3[val3] = cnt3 + 1
        cnt3 += 1    

print(d1)
print(d2)
print(d3)

def pca(dataMat, topNfeat):
    meanVals = np.mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals  # 标准化（去均值）
    covMat = np.cov(meanRemoved, rowvar=False)
    eigVals, eigVets = np.linalg.eig(np.mat(covMat))  # 计算矩阵的特征值和特征向量
    eigValInd = np.argsort(eigVals)  # 将特征值从小到大排序，返回的是特征值对应的数组里的下标
    eigValInd = eigValInd[:-(topNfeat + 1):-1]  # 保留最大的前K个特征值
    redEigVects = eigVets[:, eigValInd]  # 对应的特征向量
    lowDDatMat = meanRemoved * redEigVects  # 将数据转换到低维新空间
    # reconMat = (lowDDatMat * redEigVects.T) + meanVals  # 还原原始数据
    return lowDDatMat

X = np.array([[1, 5, 3, 1],[4 ,2, 6, 3],[1, 4, 3, 2],[4, 4, 1, 1],[5, 5, 2, 3]])

data = pca(X, 2)
print(data)

X = np.array([[-2,1,0,-1],[1,-2,3,1],[-2,0,0,0],[1,0,-2,-1],[2,1,-1,1]])

data = pca(X, 2)
print(data)