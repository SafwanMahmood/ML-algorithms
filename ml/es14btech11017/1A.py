import numpy as np

n = int(raw_input("Enter size:"))
a = np.zeros((n,n))
a[:n:2,:n:2] = 1
a[1:n:2,1:n:2] = 1
print(a.astype(int))    