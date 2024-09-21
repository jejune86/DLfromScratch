import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np
from common.functions import softmax, cross_entropy_error

class Relu :
    def __init__(self) :
        self.mask = None # mask는 True/False로 구성된 넘파이 배열
        
    def forward(self, x) :
        self.mask = (x <= 0) # x가 0 이하인 원소는 True, 그 외는 False
        out = x.copy()
        out[self.mask] = 0 # x가 0 이하인 원소는 0으로 변환
        
        return out
    
    def backward(self, dout) :
        dout[self.mask] = 0 # mask의 원소가 True인 곳은 상류에서 전파된 dout을 0으로 설정
        dx = dout
        
        return dx
    
class Sigmoid :
    def __init__(self) :
        self.out = None
    
    def forward(self, x) :
        out = 1 / (1 + np.exp(-x))
        self.out = out
        
        return out
    
    def backward(self, dout) :
        dx = dout * (1.0 - self.out) * self.out
        
        return dx