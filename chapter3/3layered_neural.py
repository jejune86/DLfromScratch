import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

X = np.array([1.0, 0.5]) # 입력값

W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]]) #1층의 가중치
B1 = np.array([0.1, 0.2, 0.3]) # 1층의 편향

A1 = np.dot(X, W1) + B1 # 1층의 a 값 구함
Z1 = sigmoid(A1) # 1층의 z 값 구함, 활성함수를 거친 값, 활성함수로 sigmoid 사용

W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])

A2 = np.dot(Z1, W2) + B2 # 2층의 a 값 구함
Z2 = sigmoid(A2) # 2층의 z 값 구함

# 출력층의 활성함수로 항등함수를 사용
def identity_function(x):
    return x  

W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])

A3 = np.dot(Z2, W3) + B3 # 출력층의 a 값 구함
Y = identity_function(A3) # 항등함수를 활성함수로 사용하여 추력


# 3층 신경망 구현 정리

# 네트워크 구성
def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network

# 순전파
def forward(network, x) :
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)

    return y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y) # [0.31682708 0.69627909]