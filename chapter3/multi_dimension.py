import numpy as np

A  = np.array([1, 2, 3, 4])
np.ndim(A)  # 1 배열의 차원 수
A.shape  # (4,) 배열의 모양
A.shape[0]  # 4 배열의 첫 번째 차원의 크기

# 행렬의 내적
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print(np.dot(A, B))  # 행렬 내적 수행

# 신경망의 내적
X = np.array([1, 2])
W = np.array([[1, 3, 5], [2, 4, 6]])
Y = np.dot(X, W) # [1*1 + 2*2, 1*3 + 2*4, 1*5 + 2*6] -> Y 를 단숨에 계산 가능