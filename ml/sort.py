import numpy as np


print("Enter the size of matrix")
n = int(input())

print("Enter the elements of matrix")
my_array = np.random.normal(size=(n, n))


print("Enter the cloumn to sort the matrix")
col = int(input())
col = col - 1
l = my_array[np.argsort(my_array[:,col])]


print(l) 