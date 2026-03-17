"""
线性回归:
    用线性方程描述多个自变量和一个因变量之间的关系
    最简单的就是一元一次方程y=kx+b，其实就是一种线性回归

    有监督学习，有特征，有标签，标签连续，预测的是一个数

    在线性回归中：
        一元线性
            y=wx+b
                w：权重
                b：偏置（Bias）
        多元线性
            y=w1x1+w2x2+w3x3+...+wnxn+b
            =w的转置*x+b
            矩阵乘法

    误差=预测值-真实值
    损失函数（Loss Function）
    l'(x)
    描述每个样本点和其预测值之间的关系，损失函数越小，误差和越小，效率越高

    正规方程法
    梯度下降法
    损失函数分类:
        最小二乘：每个样本误差的平方和
        MSE(Mean Square Error,均方误差):每个样本点误差的平方和 / 样本个数
        RMSE(Root Mean Square Error，均方根误差):均方误差开平方根
        MAE(Mean Absolute Error，均绝对误差):每个样本误差的绝对值和 / 样本个数
"""

from sklearn.linear_model import LinearRegression

#1.准备数据
#2.数据与处理
#3.特征工程（特征提取，特征预处理）
#4.模型训练
#5.模型预测
#6.模型评估

x_train=[[160],[166],[172],[174],[180]]
y_train=[56.3,60.6,65.1,68.5,75]
x_test=[[176]]
estimator=LinearRegression()
estimator.fit(x_train,y_train)

print(f'权重:{estimator.coef_}')
print(f'偏置:{estimator.intercept_}')

y_pre=estimator.predict(x_test)
print(y_pre)