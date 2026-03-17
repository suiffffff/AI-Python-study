"""
    线性回归：有监督学习，有特征，有标签，标签连续
    用线性公式描述特征和标签之间的关系

    如何衡量模型好坏：
        思路：
            预测值和真实值之间的误差越小越好
        具体方案：
            1.最小二乘
            2.均方误差MSE
            3.均方根误差RMSE
            4.平均绝对误差MAE
        如何让损失函数最小：
            1.正规方程  =>
            2.梯度下降  => 全梯度下降FGD 随机梯度下降SGD，小批量梯度Min-Batch 随机平均梯度梯度SAG
"""

from sklearn.preprocessing import StandardScaler #特征处理
from sklearn.model_selection import train_test_split #数据集划分
from sklearn.linear_model import LinearRegression,SGDRegressor #正规方程与梯度下降
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error  # 均方误差

import pandas as pd
import numpy as np

data_url="http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
# print(f'特征:{data.shape}')
# print(f'标签:{target.shape}')
# print(f'特征数据:{data[:5]}')
# print(f'标签数据:{target[:5]}')
x_train,x_test,y_train,y_test=train_test_split(data,target,test_size=0.2,random_state=888)

transfer=StandardScaler()

x_train=transfer.fit_transform(x_train)
x_test=transfer.transform(x_test)

estimator=LinearRegression(fit_intercept=True)
estimator.fit(x_train,y_train)

print(f'权重:{estimator.coef_}')
print(f'偏置:{estimator.intercept_}')

y_pre=estimator.predict(x_test)
print(f'预测结果为:{y_pre}')
print(f'均方误差:{mean_squared_error(y_test,y_pre)}')
print(f'均方根误差:{root_mean_squared_error(y_test,y_pre)}')
print(f'平均绝对误差:{mean_absolute_error(y_test,y_pre)}')

