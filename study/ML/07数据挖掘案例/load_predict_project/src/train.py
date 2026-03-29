import os
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from utils.log import Logger
from utils.common import data_preprocessing
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error, \
    mean_absolute_percentage_error
import joblib

plt.rcParams['font.family']='SimHei'
plt.rcParams['font.size']=15

class PowerLoadModel:

    def __init__(self):
        logfile_name='train_'+datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        self.logfile=Logger('../',logfile_name,).get_logger()
        self.logfile.info('开始获得数据源')
        self.data_source=data_preprocessing('../data/train.csv')

#分析数据
def ana_data(data):
    ana_data=data.copy()
    ana_data.info()

    fig=plt.figure(figsize=(20,40))
    ax1=fig.add_subplot(411)
    ax1.hist(ana_data['power_load'],bins=100)
    ax1.set_title('负荷整体分布情况')
    ax1.set_xlabel('负荷')

    ana_data['hour']=ana_data['time'].str[11:13]
    hour_load_mean=ana_data.groupby('hour',as_index=False)['power_load'].mean()
    ax2=fig.add_subplot(412)
    ax2.plot(hour_load_mean['hour'],hour_load_mean['power_load'])
    ax2.set_title('各个小时平均负荷')
    ax2.set_xlabel('小时')

    ana_data['month']=ana_data['time'].str[5:7]
    month_load_mean=ana_data.groupby('month',as_index=False)['power_load'].mean()
    ax3=fig.add_subplot(413)
    ax3.plot(month_load_mean['month'],month_load_mean['power_load'])
    ax3.set_title('各月份平均负荷')
    ax3.set_xlabel('月')

    ana_data['weekday']=ana_data['time'].apply(lambda x:pd.to_datetime(x).weekday())
    ana_data['is_holiday']=ana_data['weekday'].apply(lambda x: 1 if x in [5,6] else 0)
    work_load_mean=ana_data[ana_data['is_holiday']==0].power_load.mean()
    holiday_load_mean=ana_data[ana_data['is_holiday']==1].power_load.mean()
    ax4=fig.add_subplot(414)
    ax4.bar(['工作日','周沫'],[work_load_mean,holiday_load_mean])
    ax4.set_title('工作日和周末')

    plt.savefig('../data/fig/负荷分布整体情况.png')
    plt.show()


def feature_engineering(data,logger):
    feature_data=data.copy()
    feature_data['hour']=feature_data['time'].str[11:13]
    feature_data['month']=feature_data['time'].str[5:7]
    hour_month_data=pd.get_dummies(feature_data[['hour','month']])
    feature_data=pd.concat([feature_data,hour_month_data],axis=1)

    #上n个小时
    load_1h_data=feature_data['power_load'].shift(1)
    load_2h_data=feature_data['power_load'].shift(2)
    load_3h_data=feature_data['power_load'].shift(3)
    load_shift_df=pd.concat([load_1h_data,load_2h_data,load_3h_data],axis=1)
    load_shift_df.columns=['前1小时','前2小时','前3小时']
    feature_data=pd.concat([feature_data,load_shift_df],axis=1)
    feature_data.info()

    feature_data['yesterday_time']=feature_data['time'].apply(lambda x:(pd.to_datetime(x)-datetime.timedelta(days=1)).strftime('%Y-%m-%d %H:%M:%S'))
    time_load_dict=feature_data.set_index('time')['power_load'].to_dict()
    # print(feature_data.head(10))
    feature_data['yesterday_load']=feature_data['yesterday_time'].apply(lambda x: time_load_dict.get(x))

    feature_data=feature_data.dropna()
    feature_data.info()
    feature_columns=list(hour_month_data.columns)+list(load_shift_df.columns)+['yesterday_load']
    # print(feature_columns)
    #['hour_00', 'hour_01', 'hour_02', 'hour_03', 'hour_04', 'hour_05', 'hour_06', 'hour_07', 'hour_08', 'hour_09', 'hour_10', 'hour_11', 'hour_12', 'hour_13', 'hour_14', 'hour_15', 'hour_16', 'hour_17', 'hour_18', 'hour_19', 'hour_20', 'hour_21', 'hour_22', 'hour_23', 'month_01', 'month_02', 'month_03', 'month_04', 'month_05', 'month_06', 'month_07', 'month_08', 'month_09', 'month_10', 'month_11', 'month_12', '前1小时', '前2小时', '前3小时', 'yesterday_time']
    return feature_data,feature_columns

def model_train(data,features,logger):
    x=data[features]
    y=data['power_load']
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=666)

    # logger.info('---网格搜索+交叉验证 寻找最优超参----')
    # logger.info(f'开始时间:{datetime.datetime.now()}')
    # param_dict={
    #     'n_estimators':[50,100,150,200],
    #     'max_depth':[3,5,7],
    #     'learning_rate':[0.01,0.1]
    # }
    # estimator=XGBRegressor()
    # gs=GridSearchCV(estimator=estimator,param_grid=param_dict,cv=5)
    # gs.fit(x_train,y_train)
    # logger.info(f'最优超参:{gs.best_estimator_}')
    # logger.info(f'结束时间:{datetime.datetime.now()}')

    estimator=XGBRegressor(n_estimators=100,max_depth=3,learning_rate=0.1)
    estimator.fit(x_train,y_train)
    y_pre=estimator.predict(x_test)
    joblib.dump(estimator,'../model/xgb_20260328.pkl')

    print(f'均方误差:{mean_squared_error(y_test,y_pre)}')
    print(f'均方根误差:{root_mean_squared_error(y_test,y_pre)}')
    print(f'平均绝对误差:{mean_absolute_error(y_test,y_pre)}')
    print(f'平均绝对百分比误差:{mean_absolute_percentage_error(y_test,y_pre)}')




if __name__ == '__main__':
    pm=PowerLoadModel()
    # print(pm.data_source)
    # ana_data(pm.data_source)
    feature_data,feature_columns=feature_engineering(pm.data_source,pm.logfile)
    model_train(feature_data,feature_columns,pm.logfile)




