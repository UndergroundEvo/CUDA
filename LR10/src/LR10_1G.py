import numpy as np
from time import time

start = time()
num = 1 << 11
rows = 2 * num

A = np.empty((rows, rows))
B = np.empty((rows, rows))

for i in range(rows):
    for j in range(rows):
        A[i][j] = i+j
        B[i][j] = i+j+1

#print(A)
#print(B)
start = time()

С = np.matmul(A, B)

end = time()
print('Функция выполнилась за ', end - start)
