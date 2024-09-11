import matplotlib.pyplot as plt  # 그림으로 보기 위한 matplotlib 라이브러리 import
import numpy as np
import pickle
from tensorflow.keras.datasets import mnist  # 라이브러리가 기본으로 제공하는 mnist 데이터셋
from activation_fuction import sigmoid
from output_layer import softmax
from PIL import Image


(x_train, t_train), (x_test, t_test) = mnist.load_data()

def _change_one_hot_label(X): # 원-핫 인코딩 형태로 변환
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1
        
    return T

def init_data(normalize=True, flatten=True, one_hot_label=False):
    global x_train, t_train, x_test, t_test
    if normalize: # 정규화 전처리 
        x_train = x_train.astype(np.float32)
        x_train /= 255.0
        x_test = x_test.astype(np.float32)
        x_test /= 255.0

    if one_hot_label:
        t_train = _change_one_hot_label(t_train)
        t_test = _change_one_hot_label(t_test)

    if flatten:
        x_train = x_train.reshape(x_train.shape[0], 784)
        x_test = x_test.reshape(x_test.shape[0], 784)
        

def init_network() :
    with open("chapter3/sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)  
    return network

def predict(network, x) :
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    
    return y

def img_show(img) :
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()
         
if __name__ == '__main__' :
    
    init_data()
    network = init_network()
    accuracy_cnt = 0
    
    # 하나씩 이미지를 꺼내서 예측
    # for i in range(len(x_test)):
    #     y = predict(network, x_test[i])
    #     p = np.argmax(y) #가장 확률이 높은 인덱스 get
    #     if p == t_test[i]:
    #         accuracy_cnt += 1
    
    # 배치 처리로 이미지를 꺼내서 예측 (효율적)
    batch_size = 100
    for i in range(0, len(x_test), batch_size):
        x_batch = x_test[i:i+batch_size] # 100개씩 묶음
        y_batch = predict(network, x_batch) # 100개씩 예측
        p = np.argmax(y_batch, axis=1) #가장 확률이 높은 인덱스 get -> 예측한 답
        accuracy_cnt += np.sum(p == t_test[i:i+batch_size]) # 정답과 비교하여 맞은 개수 세기
            
    print ("Accuracy:" + str(float(accuracy_cnt) / len(x_test)))


