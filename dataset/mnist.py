from tensorflow.keras.datasets import mnist
import numpy as np

def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    (x_train, t_train), (x_test, t_test) = mnist.load_data()
    if normalize:
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

    return (x_train, t_train), (x_test, t_test)

def _change_one_hot_label(X): # 원-핫 인코딩 형태로 변환
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1

    return T
