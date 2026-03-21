"""
    线性回归和回归决策树有一定区别，需要对比

    回归决策树容易过拟合（非线性，将相似的数据分类  b）

    CART分类回归决策树，既可以做分类，也可以做回归，但一般用于分类
    做分类是用基尼值，做回归用平方损失

"""
from cProfile import label

import numpy as np
import pandas as pf
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

x_train=np.array(list(range(1,11))).reshape(-1,1)
y_train=np.array([5.56,5.7,5.91,6.4,6.8,7.05,8.9,8.7,9,9.05])

estimator=LinearRegression()
estimator2=DecisionTreeRegressor(max_depth=1)
estimator3=DecisionTreeRegressor(max_depth=3)

estimator.fit(x_train,y_train)
estimator2.fit(x_train,y_train)
estimator3.fit(x_train,y_train)

x_test=np.arange(0,10,0.1).reshape(-1,1)
print(x_test)

y_pre1=estimator.predict(x_test)
y_pre2=estimator2.predict(x_test)
y_pre3=estimator3.predict(x_test)

plt.scatter(x_train,y_train)
plt.plot(x_test,y_pre1,label='Linear regression')
plt.plot(x_test,y_pre2,label='Max depth=1')
plt.plot(x_test,y_pre3,label='Max depth=3')
plt.legend()
plt.xlabel('data')
plt.ylabel('target')
plt.title('Decision Tree Regressor')
plt.show()
