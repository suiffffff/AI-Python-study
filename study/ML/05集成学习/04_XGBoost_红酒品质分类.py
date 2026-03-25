
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from collections import Counter
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.utils import class_weight


def dm01_data_split():
    df=pd.read_csv('./data/红酒品质分类.csv')
    df.info()

    x=df.iloc[:,:-1]
    y=df.iloc[:,-1]-3

    print(x[:5])
    print(y[:5])
    print(Counter(y))

    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=666,stratify=y)
    pd.concat([x_train,y_train],axis=1).to_csv('./data/红酒品质分类_train.csv',index=False)
    pd.concat([x_test,y_test],axis=1).to_csv('./data/红酒品质分类_test.csv',index=False)

def dm02_train_model():
    train_data=pd.read_csv('./data/红酒品质分类_train.csv')
    test_data=pd.read_csv('./data/红酒品质分类_test.csv')
    x_train=train_data.iloc[:,:-1]
    y_train=train_data.iloc[:,-1]
    x_test=test_data.iloc[:,:-1]
    y_test=test_data.iloc[:,-1]

    estimator=xgb.XGBClassifier(
        max_depth=5,
        n_estimators=100,
        learning_rate=0.1,
        random_state=666,
        objective='multi:softmax'
    )
    class_weight.compute_sample_weight('balanced',y_train)

    estimator.fit(x_train,y_train)
    y_pre=estimator.predict(x_test)
    print(f'准确率:{estimator.score(x_test,y_test)}')
    print(f'准确率:{accuracy_score(y_test,y_pre)}')
    joblib.dump(estimator,'./model/红酒品质分类.pkl')
    print("模型保存成功")

def dm03_use_model():
    train_data=pd.read_csv('./data/红酒品质分类_train.csv')
    test_data=pd.read_csv('./data/红酒品质分类_test.csv')
    x_train=train_data.iloc[:,:-1]
    y_train=train_data.iloc[:,-1]
    x_test=test_data.iloc[:,:-1]
    y_test=test_data.iloc[:,-1]
    estimator=joblib.load('./model/红酒品质分类.pkl')

    param_dict={
        'max_depth':[2,3,5,6,7],
        'n_estimators':[30,50,100,200],
        'learning_rate':[0.1,0.2,1,1.3]
    }
    skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=666)
    gs_estimator=GridSearchCV(estimator,param_dict,cv=skf)
    gs_estimator.fit(x_train,y_train)
    y_pre=gs_estimator.predict(x_test)
    print(f'最优组合:{gs_estimator.best_estimator_}')
    print(f'最优评分:{gs_estimator.best_score_}')
    print(f'准确率:{accuracy_score(y_test,y_pre)}')



if __name__=='__main__':
    # dm01_data_split()
    # dm02_train_model()
    dm03_use_model()
