import numpy as np

def mean_squared_error(y, t) :
    return 0.5 * np.sum((y-t)**2)

t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0] # 정답은 2 -> one-hot encoding
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0] #2일 확률이 가장 높다고 추정

mean_squared_error(np.array(y), np.array(t)) # 0.0975

y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0] #7일 확률이 가장 높다고 추정
mean_squared_error(np.array(y), np.array(t)) # 0.5975

#첫번째가 정확하므로 MSE가 작다.

# def cross_entropy_error(y, t) :
#     delta = 1e-7 # log0 방지
#     return -np.sum(t * np.log(y + delta))

def cross_entropy_error(y, t) : # 배치 데이터 처리 가능한 교차 엔트로피 오차
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    batch_size = y.shape[0]
    # 원-핫 인코딩일 때
    # return -np.sum(t * np.log(y + 1e-7)) / batch_size 
    
    # 정답 레이블이 숫자 레이블일 때
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size  