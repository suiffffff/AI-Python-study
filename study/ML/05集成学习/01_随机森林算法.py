"""
    集成学习 Bagging思想 随机森林算法

    把多个弱学习器组成一个强学习器的过程 -> 集成学习
    思想：
        Bagging思想：
            1.有放回的随机抽样。
            2.平权投票
            3.可以并行执行
        Boosting思想：
            1.每次训练都会使用全部样本
            2.加权投票 -> 预测正确：权重降低 预测错误：权重增加
            3.只能串行执行

    随机森林算法：
        1.每个弱学习器都是CART树（二叉树）
        2.有放回的随机抽样，平权投票，并行执行

"""

import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

df=pd.read_csv('./data/train.csv')
# df.info()

x=df[['Pclass','Sex','Age']].copy()
y=df['Survived']

x['Age']=x['Age'].fillna(x['Age'].mean())
x=pd.get_dummies(x)

# 单一决策树
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=666)

estimator1=DecisionTreeClassifier()
estimator1.fit(x_train,y_train)

y_pre=estimator1.predict(x)

print(f'准确率:{estimator1.score(x_test,y_test)}')

# 随机森林算法 默认参数

estimator2=RandomForestClassifier()

estimator2.fit(x_train,y_train)

print(f'随机森林准确率:{estimator2.score(x_test,y_test)}')

# 随机森林算法 网格搜索

estimator3=RandomForestClassifier()
estimator3.fit(x_train,y_train)
params={'n_estimators':[60,90,100,130,150],'max_depth':[3,5,7,9]}
gs_estimator=GridSearchCV(estimator3,param_grid=params,cv=3)
gs_estimator.fit(x_train,y_train)

print(f'网格搜索准确率:{gs_estimator.score(x_test,y_test)}')
print(f'最佳参数:{gs_estimator.best_params_}')


 