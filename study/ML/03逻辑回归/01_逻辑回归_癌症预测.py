"""
逻辑回归：
    有监督学习，有特征，有标签，标签离散
    适用于二分类
原理：
    线性回归处理后的预测值 -> sigmoid激活函数映射到[0,1]概率 -> 通过自定义阈值分类
极大似然估计：
    通过结果反推可能性最大的概率，也就是找到的概率，使结果发生的概率最大
损失函数：
    需要注意的是1-L其实才是很准确的极小似然估计
    但-logL是一种趋势，L悦达，则-logL越小，L越小，-logL越大
    求-logL的极小值，等价于求L的极大值
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data_url="./data/breast-cancer-wisconsin.csv"
data=pd.read_csv(data_url)

data.replace("?",np.nan,inplace=True)
data.dropna(axis=0,inplace=True)

x=data.iloc[:,1:-1]
y=data.iloc[:,-1]

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=666)

transfer=StandardScaler()
x_train=transfer.fit_transform(x_train)
x_test=transfer.transform(x_test)

estimator=LogisticRegression()
estimator.fit(x_train,y_train)

y_pre=estimator.predict(x_test)

print(f'训练前评估：{estimator.score(x_test,y_test)}')
print(f'训练后评估：{accuracy_score(y_pre,y_test)}')

