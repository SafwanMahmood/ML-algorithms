
import numpy as np
a=np.genfromtxt("data.csv",delimiter=',')
print(a.shape)
keep = []
for i in range(0, a.shape[1]) :
		if (a[i, j] == "?") :
		      keep.append(i)		
         
print keep
a2 = a[keep]
print(a2)
#np.isfinite(A).all(1)
#A =A[np.isfinite(A).all(1)]
print(a2.shape)