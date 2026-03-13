from os import name

from sklearn.datasets import load_iris
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#定义并加载鸢尾花数据集
def dm01_load_iris():
    iris_data=load_iris()
    # print(f'数据集:{iris_data}')
    # print(f'数据集:{iris_data.keys()}')
    # print(f'类型:{type(iris_data)}')
    # print(f'具体数据:{iris_data.data[:5]}')
    # print(f'具体标签:{iris_data.target[:5]}')
    # print(f'特征名:{iris_data.feature_names}')

#函数可视化
def dm02_show_iris():
    iris_data=load_iris()
    iris_df=pd.DataFrame(iris_data.data,columns=iris_data.feature_names)
    iris_df['label']=iris_data.target
    print(iris_df)
    sns.lmplot(data=iris_df,x='sepal length (cm)',y='sepal width (cm)',hue='label',fit_reg=False)
    plt.title('iris data')
    plt.tight_layout()
    plt.show()

#定义函数，切分训练集和测试集
def dm03_split_train_test():
    iris_data=load_iris()
    x_train,x_test,y_train,y_test=train_test_split(iris_data.data,iris_data.target,test_size=0.2,random_state=666)
    print(f'训练集特征:{x_train},个数:{len(x_train)}')
    print(f'训练集标签:{y_train},个数:{len(y_train)}')
    print(f'测试集特征:{x_test},个数:{len(x_test)}')
    print(f'测试集标签:{y_test},个数:{len(y_test)}')

#定义函数实现鸢尾花完整案例
def dm04_iris_evaluate_test():
    iris_data=load_iris()

    x_train, x_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.2,random_state=666)

    transfer=StandardScaler()

    #fit_transform 兼具fit和transform的功能，训练，转化，该函数适用于：第一次标准化的时候使用，一般用于处理训练集
    x_train=transfer.fit_transform(x_train)
    #transform 只有转化，适用于重复标准化动作时使用，一般用于测试集
    x_test=transfer.transform(x_test)

    estimator=KNeighborsClassifier(n_neighbors=3)

    estimator.fit(x_train,y_train)

    y_pre=estimator.predict(x_test )

    print(f'预测值为:{y_pre}')

    my_data=[[7.8,2.1,3.9,1.6]]
    my_data=transfer.transform(my_data)
    y_pre_new=estimator.predict(my_data)
    print(f'预测值为:{y_pre_new}')
    y_pre_proba=estimator.predict_proba(my_data)
    print(y_pre_proba)

    print(f'正确率:{estimator.score(x_test,y_test)}')
    print(f'正确率:{accuracy_score(y_test,y_pre)}')

if __name__=='__main__':
    # dm01_load_iris()
    # dm02_show_iris()
    # dm03_split_train_test()
    dm04_iris_evaluate_test()