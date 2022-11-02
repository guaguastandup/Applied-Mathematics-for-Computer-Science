import numpy as np 
import math
import pandas as pd

N = 8000

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