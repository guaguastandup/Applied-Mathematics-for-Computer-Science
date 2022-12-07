import numpy as np

X = np.array([[-2, 1, -2, 1, 2],[1, -2, 0, 0, 1],[0, 3, 0, -2, -1],[-1, 1, 0, -1, 1]])
XT = X.T

C = X.dot(XT)
C = C/5
print("C:")
print(C)

Value, P = np.linalg.eig(C)
print("Value:")
print(Value)

print("P:")
print(P)

P = P[:2]
print("P:")
print(P)

Y = P.dot(X)
print("Y:")
print(Y)

print("------------------------------------------------------------")



def pca(X,k):
    m_samples , n_features = X.shape
    print("here:", m_samples, n_features)
    #中心化  去均值  均值为0
    mean = np.mean(X,axis=0)
    normX = X - mean  #去均值，中心化
    cov_mat = np.dot(np.transpose(normX),normX) #协方差矩阵
    #对二维数组的transpose操作就是对原数组的转置操作  矩阵相乘
    vals , vecs = np.linalg.eig(cov_mat) #得到特征向量和特征值
    print('特征值',vals)
    print('特征向量',vecs)
    eig_pairs = [(np.abs(vals[i]),vecs[:,i]) for i in range(n_features)]
    print(eig_pairs)
    print('-------------')
    #将特征值由大到小排列
    eig_pairs.sort(reverse=True)
    print("eig_pairs: ")
    print(eig_pairs)
    print('-------------')
    feature = np.array(eig_pairs[0][k])
    print("feature: ")
    print(feature)
    #将数据进行还原操作 normX 中心化后的数据 和 特征向量相乘
    data = np.dot(normX,np.transpose(feature))
    return data


# X = np.array([[-1,1],[-2,-1],[-3,-2],[1,1],[2,1],[3,2]])
# X = np.array([[-2, 1, -2, 1, 2],[1, -2, 0, 0, 1],[0, 3, 0, -2, -1],[-1, 1, 0, -1, 1]])
X = np.array([[-2,1,0,-1],[1,-2,3,1],[-2,0,0,0],[1,0,-2,-1],[2,1,-1,1]])
# X = np.array([[1, 5, 3, 1],[4 ,2, 6, 3],[1, 4, 3, 2],[4, 4, 1, 1],[5, 5, 2, 3]])

# # data = pca(X,1)
# data = pca(X, 1)
# print("data:")
# print(data)


# from sklearn.decomposition import PCA
# p = PCA(n_components = 2)
# a = p.fit_transform(X)
# print(a)





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

# X = np.array([[-2, 1, -2, 1, 2],[1, -2, 0, 0, 1],[0, 3, 0, -2, -1],[-1, 1, 0, -1, 1]])
# X = np.array([[-2,1,0,-1],[1,-2,3,1],[-2,0,0,0],[1,0,-2,-1],[2,1,-1,1]])
X = np.array([[1, 5, 3, 1],[4 ,2, 6, 3],[1, 4, 3, 2],[4, 4, 1, 1],[5, 5, 2, 3]])

data = pca(X, 2)
print(data)

X = np.array([[-2,1,0,-1],[1,-2,3,1],[-2,0,0,0],[1,0,-2,-1],[2,1,-1,1]])
# X = np.array([[1, 5, 3, 1],[4 ,2, 6, 3],[1, 4, 3, 2],[4, 4, 1, 1],[5, 5, 2, 3]])

data = pca(X, 2)
print(data)