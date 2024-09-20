import sys, os
import numpy as np
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# print (x_train.shape) # (60000, 784)
# print (t_train.shape) # (60000, 10)

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size) # train_size 미만 중 무작위로 batch개 선택
x_batch = x_train[batch_mask] # 랜덤으로 뽑힌 애만 선택
t_batch = t_train[batch_mask]
