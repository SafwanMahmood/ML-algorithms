import numpy as np


print("Enter the size of matrix")
n = int(input())


myarray = np.zeros((n,n))

myarray = np.fromfunction(lambda i,j : i+j , (n,n) ,dtype=int)

print(myarray)
