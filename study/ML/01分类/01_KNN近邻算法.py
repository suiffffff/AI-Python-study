"""
基于欧氏距离：
    本质类似于求两点间的距离，找到两点间距离最近的N个样本
    根据N个样本的票数决定平均值/分类

分类：
    有特征，有标签，标签不连续（无关系）
回归：
    有特征，有标签，标签连续（有关系）
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor

# x_train=[[0],[1],[2],[3]]
# y_train=[0,0,1,1]
# x_test=[[5]]
#
# estimator=KNeighborsClassifier(n_neighbors=2)
#
# estimator.fit(x_train,y_train)
#
# y_pre=estimator.predict(x_test)
#
# print(f"预测值为:{y_pre}")

x_train=[[0,0,1],[1,1,0],[3,10,10],[4,11,12]]
y_train=[0.1,0.2,0.3,0.4]

x_test=[[3,11,10]]

estimator=KNeighborsRegressor(n_neighbors=3)

estimator.fit(x_train,y_train)
y_pre=estimator.predict(x_test)
print(f"预测值为:{y_pre}")