"""
交叉验证：
    将数据分为n份，每次选一份作为测试集，剩下的作为训练集
    最后计算n次准确率的平均值
    对于准确率最高的一项，用全部数据训练模型，再次用测试集对模型测试

网格搜索：
    接受超参可能出现的值，然后针对每个超参进行交叉验证，获得最优组合
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris_data=load_iris()

x_train,x_test,y_train,y_test=train_test_split(iris_data.data,iris_data.target,test_size=0.2,random_state=999)

transfer=StandardScaler()
x_train=transfer.fit_transform(x_train)
x_test=transfer.transform(x_test)

estimator=KNeighborsClassifier(n_neighbors=3)
param_dict={'n_neighbors':[i for i in range(1,11)]}
estimator=GridSearchCV(estimator,param_dict,cv=4)

estimator.fit(x_train,y_train)

print(f'最优评分:{estimator.best_score_}')
print(f'最优超参组合:{estimator.best_params_}')
print(f'最优的估计器对象:{estimator.best_estimator_}')
print(f'具体的交叉验证结果:{estimator.cv_results_}')

#获取最优超参
# estimator=estimator.best_estimator_
estimator=KNeighborsClassifier(n_neighbors=7)
estimator.fit(x_train,y_train)
y_pre=estimator.predict(x_test)

print(f'准确率:{accuracy_score(y_test,y_pre)}')