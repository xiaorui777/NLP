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
best_k, best_b = None,None
loop_times = 10000

losses = []

while loop_times > 0:
    # k,b在实数域
    k_hat = random.random() * 20 - 10
    b_hat = random.random() * 20 - 10
    # k,b在整数域
    # k_hat = random.randint(-10,10)
    # b_hat = random.randint(-10,10)

    estimated_fares = func(sub_ages,k_hat,b_hat)
    error_rate = loss(y=sub_fares,yhat=estimated_fares)

    if error_rate < min_error_rate:
        min_error_rate = error_rate
        
        print(min_error_rate)
        print('loop == {}'.format(loop_times))
        losses.append(min_error_rate)
        print('f(age)={} * age + {}, with error rate:{}'.format(k_hat,b_hat,min_error_rate))
    
    loop_times -= 1

# 绘制散点图
plt.scatter(sub_ages,sub_fares)
plt.plot(sub_ages,estimated_fares,c = 'r')
plt.show()

# 观察loss变化
# plt.plot(range(len(losses)),losses)
# plt.show()

# 结果：
# loop == 3983  
# f(age)=3.979430122520034 * age + 9.256969473321604, with error rate:61.14482748887377