import numpy as np

class SGD : 
    def __init__(self, lr=0.01) : #lr = learning rate
        self.lr = lr
    
    def update(self, params, grads) :
        for key in params.keys() :
            params[key] -= self.lr*grads[key] # -= 기울기 * 학습률,  

class Momentum :
    def __init__(self, lr=0.01, momentum=0.9) :
        self.lr = lr
        self.momentum = momentum  #모멘텀 상수 alpha
        self.v = None #속도, 초기화 시에는 None
        
    def update(self, params, grads) :
        if self.v is None :
            self.v = {}
            for key, val in params.items() :
                self.v[key] = np.zeros_like(val) # 0으로 초기화
        
        for key in params.keys() :
            self.v[key] = self.momentum*self.v[key] - self.lr*grads[key] # v = alpha*v - lr*grad
            params[key] += self.v[key] # W = W + v
            

class AdaGrad :
    def __init__(self, lr=0.01) :
        self.lr = lr
        self.h = None
        
    def update(self, params, grads) :
        if self.h is None :
            self.h = {}
            for key, val in params.items() :
                self.h[key] = np.zeros_like(val)
        
        for key in params.keys() :
            self.h[key] += grads[key]*grads[key] # h = h + grad^2
            params[key] -= self.lr*grads[key]/(np.sqrt(self.h[key])+1e-7) # 1e-7은 0으로 나누는 것을 방지하기 위한 작은 값
            
            
class Adam :
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999) :
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        
    def update(self, params, grads) :
        if self.m is None :
            self.m, self.v = {}, {}
            for key, val in params.items() :
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
                
        self.iter += 1
        lr_t = self.lr*np.sqrt(1.0 - self.beta2**self.iter)/(1.0 - self.beta1**self.iter)
        
        for key in params.keys() :
            self.m[key] += (1 - self.beta1)*(grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2)*(grads[key]**2 - self.v[key])
            
            params[key] -= lr_t*self.m[key]/(np.sqrt(self.v[key]) + 1e-7)