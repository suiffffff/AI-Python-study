"""
欠拟合:     模型过于简单，在训练集和测试集上表现都不好
正好拟合:   模型在训练集和测试集都表现好
过拟合:     模型过于复杂，在训练集上表现好，但在测试集上表现差

解决方法:
    欠拟合： 增加特征列，增加模型复杂度
    过拟合： 减少模型复杂度，手动减少特征列，L1和L2正则化

正则化:
    都是基于惩罚系数来降低模型特征权重的，惩罚系数越大，修改力度越大，对应系数越小
区别:
    L1正则化：可以使权重变为0，达成特征选择的目的
    L2正则化：只能让权重无限趋于0，但达不到0
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import  LinearRegression
from  sklearn.metrics import  mean_squared_error,root_mean_squared_error,mean_absolute_error
from sklearn.linear_model import Lasso,Ridge

#欠拟合
def dm01_under_fitting():
    np.random.seed(666)
    x=np.random.uniform(-3,3,100)
    y=0.5*x**2+x+2+np.random.normal(0,1,100)

    print(f'特征:{x}')
    print(f'标签:{y}')

    X=x.reshape(-1,1)

    estimator=LinearRegression()
    estimator.fit(X,y)
    y_pre=estimator.predict(X)

    print(f'均方误差:{mean_squared_error(y, y_pre)}')
    print(f'均方根误差:{root_mean_squared_error(y, y_pre)}')
    print(f'平均绝对误差:{mean_absolute_error(y, y_pre)}')

    plt.scatter(x,y)
    plt.plot(x,y_pre)
    plt.show()

#正好拟合
def dm02_just_fitting():
    np.random.seed(666)
    x=np.random.uniform(-3,3,100)
    y=0.5*x**2+x+2+np.random.normal(0,1,100)

    #print(f'特征:{x}')
    #print(f'标签:{y}')

    X=x.reshape(-1,1)
    X2=np.hstack([X,X**2])
    print(X2[:5])

    estimator=LinearRegression()
    estimator.fit(X2,y)
    y_pre=estimator.predict(X2)

    print(f'均方误差:{mean_squared_error(y, y_pre)}')
    print(f'均方根误差:{root_mean_squared_error(y, y_pre)}')
    print(f'平均绝对误差:{mean_absolute_error(y, y_pre)}')

    plt.scatter(x,y)
    plt.plot(np.sort(x),y_pre[np.argsort(x)])
    plt.show()

def dm03_over_fitting():
    np.random.seed(666)
    x=np.random.uniform(-3,3,100)
    y=0.5*x**2+x+2+np.random.normal(0,1,100)

    #print(f'特征:{x}')
    #print(f'标签:{y}')

    X=x.reshape(-1,1)
    X3=np.hstack([X,X**2,X**3,X**4,X**5,X**6,X**7,X**8,X**9,X**10],)
    print(X3[:5])

    estimator=LinearRegression()
    estimator.fit(X3,y)
    y_pre=estimator.predict(X3)

    print(f'均方误差:{mean_squared_error(y, y_pre)}')
    print(f'均方根误差:{root_mean_squared_error(y, y_pre)}')
    print(f'平均绝对误差:{mean_absolute_error(y, y_pre)}')

    plt.scatter(x,y)
    plt.plot(np.sort(x),y_pre[np.argsort(x)])
    plt.show()

def dm04_l1_regularization():
    np.random.seed(666)
    x=np.random.uniform(-3,3,100)
    y=0.5*x**2+x+2+np.random.normal(0,1,100)

    #print(f'特征:{x}')
    #print(f'标签:{y}')

    X=x.reshape(-1,1)
    X3=np.hstack([X,X**2,X**3,X**4,X**5,X**6,X**7,X**8,X**9,X**10],)
    print(X3[:5])

    estimator=Lasso(alpha=0.1)
    estimator.fit(X3,y)
    y_pre=estimator.predict(X3)

    print(f'均方误差:{mean_squared_error(y, y_pre)}')
    print(f'均方根误差:{root_mean_squared_error(y, y_pre)}')
    print(f'平均绝对误差:{mean_absolute_error(y, y_pre)}')

    plt.scatter(x,y)
    plt.plot(np.sort(x),y_pre[np.argsort(x)])
    plt.show()

def dm05_l2_regularization():
    np.random.seed(666)
    x=np.random.uniform(-3,3,100)
    y=0.5*x**2+x+2+np.random.normal(0,1,100)

    #print(f'特征:{x}')
    #print(f'标签:{y}')

    X=x.reshape(-1,1)
    X3=np.hstack([X,X**2,X**3,X**4,X**5,X**6,X**7,X**8,X**9,X**10],)
    print(X3[:5])

    estimator=Ridge(alpha=10)
    estimator.fit(X3,y)
    y_pre=estimator.predict(X3)

    print(f'均方误差:{mean_squared_error(y, y_pre)}')
    print(f'均方根误差:{root_mean_squared_error(y, y_pre)}')
    print(f'平均绝对误差:{mean_absolute_error(y, y_pre)}')

    plt.scatter(x,y)
    plt.plot(np.sort(x),y_pre[np.argsort(x)])
    plt.show()


if __name__=='__main__':
    #dm01_under_fitting()
    #dm02_just_fitting()
    #dm03_over_fitting()
    #dm04_l1_regularization()
    dm05_l2_regularization()