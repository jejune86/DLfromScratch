import numpy as np
import matplotlib.pylab as plt

# 시그모이드 함수
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 계단 함수 (실수만 받을 수 있음)
def step_function1(x):
    return np.array(x > 0, dtype=np.int)

# 계단 함수 (넘파이 배열도 받음)
def step_function2(x):
    # y = x > 0: [True, False, True, True, False]
    # y.astype(int): [1, 0, 1, 1, 0]
    y = x > 0
    return y.astype(int)

# ReLU 함수
def relu(x):
    return np.maximum(0, x)

if __name__ == "__main__" :
    # 계단함수 출력
    x = np.arange(-5.0, 5.0, 0.1)
    y = step_function2(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)  # y축의 범위 지정
    plt.show()

    # 시그모이드 함수 출력
    x = np.arange(-5.0, 5.0, 0.1)
    y = sigmoid(x)
    plt.plot(x, y)
    plt.ylim(-0.1, 1.1)  # y축의 범위 지정
    plt.show()

