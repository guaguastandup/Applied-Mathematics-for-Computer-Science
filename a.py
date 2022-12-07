import pandas as pd
import numpy as np
import math

n = 3
a = np.array([1, 2, 3])

s = np.sum(a)
avg = s/n

cov = 0
for i in range(3):
    cov += (a[i] - avg)*(a[i] - avg)

print("cov", cov)
print(np.cov(a))

b = math.sqrt(np.cov(a))

print(b)


