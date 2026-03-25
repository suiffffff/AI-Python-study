"""
    GBDT(Gradient Boosting Decision Tree，梯度提升树)

        概述：
            通过拟合 负梯度 来获取一个强学习器
        流程：
            1.采用所有目标均值作为第一个弱学习器的预测值
            2.目标值-预测值=负梯度（残差），该列值作为第二个机器学习的目标值
            3.针对于第一个弱学习器，需要计算每个分割点的最小平方和（也就是损失函数），找到最佳分割点
            4.把上述分割点带入第二个弱学习器，计算预测值（以分割点为界限，分别求平均）
            5.以此类推，计算第二个弱学习器的负梯度，最佳分割点...
"""

import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score

df=pd.read_csv('./data/train.csv')

x=df[['Pclass','Sex','Age']]
y=df['Survived']

x['Age']=x['Age'].fillna(x['Age'].mean())
x=pd.get_dummies(x)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=765)

# 单决策树
estimator=DecisionTreeClassifier()
estimator.fit(x_train,y_train)

y_pre=estimator.predict(x_test)

print(f'单个决策树:{classification_report(y_test,y_pre)}')
print(f'单个准确率:{accuracy_score(y_test,y_pre)}')

#梯度提升树
estimator2=GradientBoostingClassifier()
estimator2.fit(x_train,y_train)
y_pre2=estimator2.predict(x_test)
print(f'梯度决策树:{classification_report(y_test,y_pre2)}')

print(f'梯度准确率:{accuracy_score(y_test,y_pre2)}')


#针对于GBDT进行参数调优

param_dict={
    'n_estimators':[60,70,80,90,100,110,120,130,140,150],
    'learning_rate':[0.1,0.2],
    'max_depth':[1,2,3]
}
estimator3=GridSearchCV(GradientBoostingClassifier(),param_dict,cv=3)

estimator3.fit(x_train,y_train)

print(f'网格搜索后:{estimator3.best_score_}')
print(f'网格搜索后:{estimator3.best_estimator_}')


test_data=pd.read_csv('./titanic/test.csv')
x_test=test_data[['Pclass','Sex','Age']]
x_test['Age']=x_test['Age'].fillna(x_test['Age'].mean())

x_test=pd.get_dummies(x_test,columns=['Sex'])

y_test_pre = estimator3.predict(x_test)

submission_df = pd.DataFrame({
    'PassengerId': test_data['PassengerId'], # ID
    'Survived': y_test_pre           # 预测结果
})
submission_path = "./titanic/my_kaggle_submission.csv"
submission_df.to_csv(submission_path, index=False)