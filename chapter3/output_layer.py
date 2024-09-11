import numpy as np


# softmax 함수
def softmax(a):
    
    # exp_a = np.exp(a) # e^x
    # sum_exp_a = np.sum(exp_a) # 합
    # -> overflow 문제 발생 가능
    
    #개선 버전
    c = np.max(a)
    exp_a = np.exp(a - c) # 오버플로 대책
    sum_exp_a = np.sum(exp_a)
    
    y = exp_a / sum_exp_a # softmax e^a/sum(e^a)
    return y