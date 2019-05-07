# NLP-titanic

## Introduction
    - dataset：titanic train.csv
    - 读取 Age 和 Fare 两列，使用一条直线去拟合 Age 和 Fare 的关系
    - 函数： Fare = k * Age + b
    - 损失函数： |y-yhat| （L1 loss）
    ![image](https://github.com/xiaorui777/NLP-titanic/raw/master/picture/version1_fig.png)

## Version1
    - 随机生成 k，b ，将最小值保留下来
      缺点：速度较慢
    - 结果：loop = 3983（共10000次）
           loss = 61.144827
    ![image](https://github.com/xiaorui777/NLP-titanic/raw/master/picture/version1_loss.png)

## Version2
    - 给k，b一个变化方向，假设下一步模型效果更好了，那么就让k，b继续沿着该方向变化，否则则沿相反方向变化
    - 结果：loop = 9368（共10000次）—— 很快达到 Version1 的收敛
           loss = 61.144827

           loop = 108
           loss = 46.729902
    ![image](https://github.com/xiaorui777/NLP-titanic/raw/master/picture/version2_loss.png)

## Version3
    - 梯度下降，使k，b沿着导数的反方向下降，直到loss为最小值
    - 结果： loop = 1
            loss = 63.9425615 (参数更多的时候这个版本更好用)
    ![image](https://github.com/xiaorui777/NLP-titanic/raw/master/picture/version3_loss.png)