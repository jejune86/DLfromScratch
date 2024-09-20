import numpy as np
import matplotlib.pylab as plt

def numerical_diff(f, x):
    # 수치 미분의 안좋은 예
    # h = 10e-50 # rounding error 문제, 너무 작은 값은 반올림 오차 문제를 일으킬 수 있다.
    # return (f(x+h) - f(x)) / h # 엄밀하지 않음

    h = 1e-4  # 0.0001
    return (f(x + h) - f(x - h)) / (2 * h) # 중심 차분 사용

def function_1(x):
    return 0.01*x**2 + 0.1*x

def function_2(x): # 변수가 2개 
    return x[0]**2 + x[1]**2


# 편미분
def function_tmp1(x0): 
    return x0*x0 + 4.0**2.0 # x1에 어떤값을 넣고 미분

def function_tmp2(x1):
    return 3.0**2.0 + x1*x1 # x0에 어떤값을 넣고 미분

# 기울기
def numerical_gradient(f, x): # 모든 변수의 편미분을 벡터로 정리한 것
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)  # x와 형상이 같은 배열을 생성
    
    for idx in range(x.size):
        tmp_val = x[idx]
        
        # f(x+h) 계산
        x[idx] = tmp_val + h
        fxh1 = f(x)
        
        # f(x-h) 계산
        x[idx] = tmp_val - h
        fxh2 = f(x)
        
        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val  # 값 복원
        
    return grad 


x = np.arange(0.0, 20.0, 0.1) # 0에서 20까지 0.1 간격의 배열 x를 만든다.
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x, y)
plt.show()
