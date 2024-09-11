import numpy as np

# AND 게이트
def AND(x1, x2):
    x = np.array([x1, x2])  # 입력
    w = np.array([0.5, 0.5])  # 가중치
    b = -0.7  # 편향
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1
    
# NAND 게이트
def NAND(x1, x2):
    x = np.array([x1, x2])  # 입력
    w = np.array([-0.5, -0.5])  # 가중치
    b = 0.7  # 편향
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1
    
# OR 게이트
def OR(x1, x2):
    x = np.array([x1, x2])  # 입력
    w = np.array([0.5, 0.5])  # 가중치
    b = -0.2  # 편향
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1
    
# XOR 게이트
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    return AND(s1, s2)
    
# x = np.array([0, 1])  # 입력
# w = np.array([0.5, 0.5])  # 가중치
# b = -0.7  # 편향
# print(w*x) # w1*x1, w2*x2
# print(np.sum(w*x)) # w1*x1 + w2*x2
# print(np.sum(w*x) + b) # w1*x1 + w2*x2 + b