import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random

# 读取csv文件
content = pd.read_csv('/Users/ruixiao/jupyter/NLP-titanic/dataset/train.csv')

# 把空值去掉
content = content.dropna()

#简化问题：年龄>22岁，票价>130 <400
ages_with_fares = content[
    (content['Age']>22)&(content['Fare']>130)&(content['Fare']<400)
]
sub_ages = ages_with_fares['Age']
sub_fares = ages_with_fares['Fare']

def func(age,k,b): return  k * age + b

def loss(y,yhat):
    '''
    param y: the real fares
    param yhat: the estimated fares
    return: how good is the estimated fares

    '''
    # loss function为 |y-yhat|
    return np.mean(np.abs(y-yhat))

min_error_rate = float('inf')

loop_times = 10000

losses = []

change_directions = [
    # (k,b)
    (+1, -1),  # k increase, b decrease
    (+1, +1),
    (-1, +1),
    (-1, -1)   # k decrease, b decrease
]

# k,b在实数域
k_hat = random.random() * 20 - 10
b_hat = random.random() * 20 - 10

best_k, best_b = k_hat, b_hat

best_direction = None

def step(): 
    return random.random() * 1

direction = random.choice(change_directions)

while loop_times > 0:
    # k,b在实数域
    # k_hat = random.random() * 20 - 10
    # b_hat = random.random() * 20 - 10
    # k,b在整数域
    # k_hat = random.randint(-10,10)
    # b_hat = random.randint(-10,10)

    k_delta_direction, b_delta_direction = direction

    k_delta = k_delta_direction * step()
    b_delta = b_delta_direction * step()

    new_k = best_k + k_delta
    new_b = best_b + b_delta

    estimated_fares = func(sub_ages,new_k,new_b)
    error_rate = loss(y=sub_fares,yhat=estimated_fares)

    if error_rate < min_error_rate:
        min_error_rate = error_rate
        best_k,best_b = new_k,new_b

        direction = (k_delta_direction, b_delta_direction)

        print(min_error_rate)
        print('loop == {}'.format(loop_times))
        losses.append(min_error_rate)
        print('f(age)={} * age + {}, with error rate:{}'.format(best_k,best_b,min_error_rate))
    else:
        direction = random.choice(change_directions)

    loop_times -= 1

# 绘制散点图
plt.scatter(sub_ages,sub_fares)
plt.plot(sub_ages,func(sub_ages,best_k,best_b),c = 'r')
plt.show()

# 观察loss变化
# plt.plot(range(len(losses)),losses)
# plt.show()

# 结果：
# loop == 9268
# f(age)=3.94787034383992 * age + 9.244789740288, with error rate:61.10090419875046

# loop == 108
# f(age)=0.6348203900065892 * age + 135.60033322984336, with error rate:46.72990251738206
