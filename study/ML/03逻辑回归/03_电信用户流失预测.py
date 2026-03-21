from os import rename

import pandas as pd
import numpy as np
from matplotlib.pyplot import flag
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report

def dm01_data_processing():
    churn_df=pd.read_csv('./data/churn.csv')
    churn_df.info()
    churn_df=pd.get_dummies(churn_df,columns=['Churn','gender'])
    churn_df.info()
    print(churn_df.head())
    churn_df.drop(['Churn_No','gender_Male'],axis=1,inplace=True)
    churn_df.rename(columns={"Churn_Yes":"flag"},inplace=True)
    print(churn_df.flag.value_counts())

def dm02_data_visualization():
    churn_df=pd.read_csv("./data/churn.csv")
    churn_df=pd.get_dummies(churn_df,columns=['Churn','gender'])
    churn_df.drop(['Churn_No','gender_Male'],axis=1,inplace=True)
    churn_df.rename(columns={'Churn_Yes':'flag'},inplace=True)
    sns.countplot(data=churn_df,x='Contract_Month',hue='flag')
    plt.show()

def dm03_logistic_regression():
    churn_df=pd.read_csv("./data/churn.csv")
    churn_df=pd.get_dummies(churn_df,columns=['Churn','gender'])
    churn_df.drop(['Churn_No','gender_Male'],axis=1,inplace=True)
    churn_df.rename(columns={'Churn_Yes':'flag'},inplace=True)
    x=churn_df[['Contract_Month','internet_other','PaymentElectronic']]
    y=churn_df['flag']
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=666)


    estimator=LogisticRegression()
    estimator.fit(x_train,y_train)

    y_pre=estimator.predict(x_test)

    print(f'准确率:{estimator.score(x_test,y_test)}')
    print(f'准确率:{accuracy_score(y_test,y_pre)}')

    print(f'精确率:{precision_score(y_test,y_pre)}')
    print(f'召回率:{recall_score(y_test,y_pre)}')
    print(f'F1值:{f1_score(y_test,y_pre)}')

    #macro avg:宏平均，不考虑样本权重，直接求平均，适用于数据均衡的状况
    #weight avg:样本权重平均，考虑样本权重求平均，适用于数据不均衡的情况
    print(f'分类评估报告:\n{classification_report(y_test,y_pre)}')

if __name__=="__main__":
    # dm01_data_processing()
    # dm02_data_visualization()
    dm03_logistic_regression()