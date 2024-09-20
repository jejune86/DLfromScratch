import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3) # 정규분포로 초기화
        
    def predict(self, x): # 예측
        return np.dot(x, self.W)
    
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        
        return loss

if __name__=='__main__' :
    net = simpleNet()
    print(net.W) # 가중치 매개변수
    
    x = np.array([0.6, 0.9])
    p = net.predict(x) 
    print(p)
    
    print(np.argmax(p) )# 최댓값의 인덱스
    
    t = np.array([0, 0, 1])
    print(net.loss(x, t)) # 손실함수 값
    
    # 기울기 
    
    # def f(W):
    #     return net.loss(x, t)
    f = lambda w: net.loss(x, t) # 람다식으로 표현
    
    dW = numerical_gradient(f, net.W)
    print(dW)
    
    