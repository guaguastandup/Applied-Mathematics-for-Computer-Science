import pandas as pd
import numpy as np
import random

def GtoM(G, N):
    M = np.zeros((N, N))
    for i in range(N):
        D_i = sum(G[i])
        if D_i == 0:
            continue
        for j in range(N):
            M[j][i] = G[i][j] / D_i # watch out! M_j_i instead of M_i_j
    return M

def PageRank(M, N, T=300, eps=1e-6):
    R = np.ones(N) / N
    for time in range(T):
        R_new = np.dot(M, R)
        if np.linalg.norm(R_new - R) < eps:
            break
        R = R_new.copy()
    return R_new

data = pd.read_csv('./dataset/PageRank_Dataset.csv')

N = 8000
G = np.zeros((N, N))
for line, row in data.iterrows(): 
    x = int(row['node_1'])
    y = int(row['node_2'])
    G[x][y] = 1
M = GtoM(G, N)

# 输出top20
values = PageRank(M, N, T=2000)
print(values)
# 输出pagerank
np.save("pageRank.npy", values)
data = np.load("pageRank.npy")

result = np.argsort(data)
print(result[-20:][::-1])

index = []
data_value = []
for i in range(N):
    index.append(i)
    data_value.append(data[i])

output = pd.DataFrame({'node': index, 'pagerank': data_value})
print(output.head())

output.to_csv("pagerank.csv", index=False)