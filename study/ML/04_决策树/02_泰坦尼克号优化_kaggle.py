import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

data=pd.read_csv('./titanic/train.csv')

data.info()

x=data[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
y=data['Survived']
x.info()
x['Age']=x['Age'].fillna(x['Age'].median())
x['Embarked'] = x['Embarked'].fillna(x['Embarked'].mode()[0])

x = pd.get_dummies(x, columns=['Sex', 'Embarked'])

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=666)

estimator=DecisionTreeClassifier(max_depth=4,random_state=666)
estimator.fit(x_train,y_train)
y_pre = estimator.predict(x_test)
print(f'分类评估报告:\n{classification_report(y_test, y_pre)}')

"""
以下为测试内容
"""

test_data=pd.read_csv('./titanic/test.csv')
test_data.info()
x_test=test_data[['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked']]
x_test['Age']=x_test['Age'].fillna(x_test['Age'].median())
x_test['Fare'] = x_test['Fare'].fillna(x_test['Fare'].median())
x_test.info()

x_test=pd.get_dummies(x_test,columns=['Sex','Embarked'])

y_test_pre = estimator.predict(x_test)

submission_df = pd.DataFrame({
    'PassengerId': test_data['PassengerId'], # ID
    'Survived': y_test_pre           # 预测结果
})
submission_path = "./titanic/my_kaggle_submission.csv"
submission_df.to_csv(submission_path, index=False)