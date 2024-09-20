import numpy as np
from differentiation import numerical_gradient


# 경사 하강법
# f : 최적화하려는 함수, init_x : 초깃값, lr : 학습률, step_num : 반복횟수
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x

def function_2(x):
    return x[0]**2 + x[1]**2

init_x = np.array([-3.0, 4.0])
print(gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100)) 
# [ -6.11110793e-10  8.14814391 ] -> 거의 (0,0)에 가까운 결과

