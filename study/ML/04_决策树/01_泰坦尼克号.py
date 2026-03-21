
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

data=pd.read_csv("./titanic/train.csv")
# data.info()

x=data[['Pclass','Sex','Age']]
y=data['Survived']

# x.fillna({'Age':x['Age'].mean()},inplace=True)
x['Age'] = x['Age'].fillna(x['Age'].mean())
x.info()
# 这里的修改并不会修改原数据
# data.info()

x=pd.get_dummies(x,columns=['Sex'])

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=666)

# max_depth=10 决策树结构最大为10
estimator=DecisionTreeClassifier(max_depth=10)
estimator.fit(x_train,y_train)

y_pre=estimator.predict(x_test)
print(f'预测值为:{y_pre}')

print(f'分类评估报告:\n{classification_report(y_test,y_pre)}')

plt.figure(figsize=(194,96))
plot_tree(estimator,filled=True,max_depth=10)
plt.savefig("./titanic/my_titanic.png")
plt.show()