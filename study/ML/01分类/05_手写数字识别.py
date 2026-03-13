"""
个人感觉有点倾向于暴力破解了吧？

每张图片28*28的下昂苏，csv中每一行都有784个列，表示每个像素点的颜色
从而构成图像

需要注意的是图片是灰度图，灰度图就是0黑1白，数表示亮度，越亮数字越大
能够有效的减少彩色图带来的计算影响，因为彩色的一般表达是（red，yellow，blue）

如果图片大小不一样，还能正确预测吗？估计应该不行
"""

import matplotlib.pyplot as plt
import pandas
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib
from  collections import Counter

#1.定义函数，接受用户传入的索引，展示索引图片

def show_digit(idx):
    df=pd.read_csv('../data/手写数字识别.csv')
    print(df)
    if idx<0 or idx>len(df)-1:
        print('索引越界')
        return
    x=df.iloc[:,1:]
    y=df.iloc[:,0]

    print(f'该图片的数字是:{y.iloc[idx]}')
    print(f'标签分布情况:{Counter(y)}')
    # print(x.iloc[idx].shape)
    x=x.iloc[idx].values.reshape(28,28)

    plt.imshow(x,cmap='gray')
    plt.axis('off')
    plt.show()

#2. 定义模型，训练模型并保存
def train_model():
    df=pd.read_csv('./data/手写数字识别.csv')

    x=df.iloc[:,1:]
    y=df.iloc[:,0]

    x =x/255

    #stratify:按比例抽取,防止出现某些数字没抽到的极端情况
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=888,stratify=y)

    estimator=KNeighborsClassifier(n_neighbors=3)
    estimator.fit(x_train,y_train)

    print(f'准确率:{estimator.score(x_test,y_test)}')
    print(f'准确率:{accuracy_score(y_test,estimator.predict(x_test))}')

    joblib.dump(estimator,'./model/手写数字识别.pkl')
    print("模型保存成功")

#3.测试模型
def use_model():
    x=plt.imread('./data/demo.png')
    # plt.imshow(x,cmap='gray')
    # plt.axis('off')
    # plt.show()

    #训练模型归一化,预测也要归一化
    #但需要注意的是plt.imread会自动将读取的数归一化，所以这里就不用/255了
    print(x)
    x=x.reshape(1,-1)

    estimator=joblib.load('./model/手写数字识别.pkl')
    y_pre=estimator.predict(x)
    print(f'你要预测的数字是:{y_pre}')

if __name__=='__main__':
    # show_digit(9)
    # train_model()
    use_model()