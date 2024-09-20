import numpy as np
import sys, os
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet




(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

#하이퍼 파라미터
iters_num = 10000 # 반복 횟수
train_size = x_train.shape[0] # 훈련 데이터 개수
batch_size = 100   # 미니배치 크기
learning_rate = 0.1 # 학습률


train_loss_list = []
train_acc_list = []
test_acc_list = []
# 1 에폭당 반복 수
iter_per_epoch = max(train_size / batch_size, 1)




network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

for i in range(iters_num):
    print("i : ", i)
    #미니배치 획득
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    #기울기 계산
    grad = network.numerical_gradient(x_batch, t_batch) # 매개변수에 따른 손실함수 기울기
    #grad = network.gradient(x_batch, t_batch) # 성능 개선판!
    
    #매개변수 갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]
        
    #학습 경과 기록
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    # print("loss : ", loss)
    
    # 1 에폭당 정확도 계산
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))
        # training data 에 포함되지 않은 데이터를 사용해 정기적으로 평가
        ## overfiiting을 방지하기 위함
        
# 그래프 그리기
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
