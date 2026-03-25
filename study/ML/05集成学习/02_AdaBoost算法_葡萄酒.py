"""
    AdaBoost算法：
        串行执行，每次使用全部样本，最后加权投票
        1.使用全部样本，通过决策树模型（第一个弱分类器）进行训练，获得结果
            思路：
                预测正确 -> 权重下降
                预测错误 -> 权重上升
        2.把第一个弱分类的处理结果，交给第二个弱分类器训练，以此类推
            思路：
                预测正确 -> 权重下降
                预测错误 -> 权重上升
        3.依次类推，串行执行，直到获得最终结果
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score

df_wine=pd.read_csv('./data/wine0501.csv')
df_wine.info()
# 决策树是二叉树,不能实现三个品类
# print(df_wine['Class label'].unique())

df_wine=df_wine[df_wine['Class label']!= 1]
# print(df_wine['Class label'].unique())
x=df_wine[['Alcohol','Hue']]
y=df_wine['Class label']

le=LabelEncoder()
y=le.fit_transform(y)
print(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=43,stratify=y)

#单一决策树充当弱分类器
estimator1=DecisionTreeClassifier(max_depth=3)
estimator1.fit(x_train,y_train)
y_pre=estimator1.predict(x_test)
print(f'单一决策树预测正确率:{accuracy_score(y_test,y_pre)}')
#集成学习
estimator2=AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1),n_estimators=200,learning_rate=0.1)

estimator2.fit(x_train,y_train)
y_pre2=estimator2.predict(x_test)
print(f'集成预测正确率:{accuracy_score(y_test,y_pre2)}')