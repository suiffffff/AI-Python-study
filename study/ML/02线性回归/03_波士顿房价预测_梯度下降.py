"""
使用随机梯度下降法
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

x_train,x_test,y_train,y_test=train_test_split(data,target,test_size=0.2,random_state=666)

transfer=StandardScaler()
x_train=transfer.fit_transform(x_train)
x_test=transfer.transform(x_test)

#参一：计算截距
estimator=SGDRegressor(fit_intercept=True,learning_rate='constant',eta0=0.01)
estimator.fit(x_train,y_train)

print(f'权重:{estimator.coef_}')
print(f'偏置:{estimator.intercept_}')

y_pre=estimator.predict(x_test)

print(f'均方误差:{mean_squared_error(y_test,y_pre)}')
print(f'均方根误差:{root_mean_squared_error(y_test,y_pre)}')
print(f'平均绝对误差:{mean_absolute_error(y_test,y_pre)}')

