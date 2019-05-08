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
    
    return np.mean(np.abs(y-yhat))

min_error_rate = float('inf')

loop_times = 10000

losses = []


# k,b在实数域
k_hat = random.random() * 20 - 10
b_hat = random.random() * 20 - 10


def derivate_k(y,yhat,x):
    abs_values = [1 if (y_i - yhat_i) >0 else -1 for y_i,yhat_i in zip(y,yhat)]
    return np.mean([a * -x_i for a, x_i in zip(abs_values,x)])

def derivate_b(y,yhat):
    abs_values = [1 if (y_i - yhat_i) >0 else -1 for y_i,yhat_i in zip(y,yhat)]
    return np.mean([a * -1 for a in abs_values])

learing_rate = 1e-3

while loop_times > 0:
    #变化方向应该是导数的反方向（乘-1）
    k_delta = -1 * learing_rate * derivate_k(sub_fares,func(sub_ages,k_hat,b_hat),sub_ages)
    b_delta = -1 * learing_rate * derivate_b(sub_ages,func(sub_ages,k_hat,b_hat))

    # k_delta_direction, b_delta_direction = direction

    # k_delta = k_delta_direction * step()
    # b_delta = b_delta_direction * step()

    # new_k = best_k + k_delta
    # new_b = best_b + b_delta

    k_hat += k_delta
    b_hat += b_delta

    estimated_fares = func(sub_ages,k_hat,b_hat)
    error_rate = loss(y=sub_fares,yhat=estimated_fares)

    # if error_rate < min_error_rate:
       #  min_error_rate = error_rate
        # best_k,best_b = new_k,new_b


        # direction = (k_delta_direction, b_delta_direction)

    # print(min_error_rate)
    print('loop == {}'.format(loop_times))
    # losses.append(min_error_rate)
    print('f(age)={} * age + {}, with error rate:{}'.format(k_hat,b_hat,error_rate))
    # else:
        # direction = random.choice(change_directions)
    losses.append(error_rate)
    loop_times -= 1

# 绘制散点图
# plt.scatter(sub_ages,sub_fares)
# plt.plot(sub_ages,func(sub_ages,k_hat,b_hat),c = 'r')
# plt.show()

# 观察loss变化
plt.plot(range(len(losses)),losses)
plt.show()

# 结果：
# loop == 1
# f(age)=4.283485462681436 * age + -11.11018667137825, with error rate:63.94256154040861