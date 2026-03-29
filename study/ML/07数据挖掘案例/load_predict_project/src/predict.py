import datetime
import os

import sys  # 需要导入 sys 模块
from cProfile import label

from scipy.stats import alpha

# 1. 获取当前文件(predict.py)的绝对路径的所在目录 (即 src 的路径)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 2. 获取 src 的上一级目录 (即 load_predict_project 的路径)
parent_dir = os.path.dirname(current_dir)
# 3. 将上一级目录添加到 Python 的模块搜索路径中
sys.path.append(parent_dir)

import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mick
import pandas as pd
from sklearn.metrics import mean_absolute_error
from utils.common import data_preprocessing
from utils.log import Logger

plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['font.size'] = 15

class   PowerLoadPredict(object):

    def __init__(self,file_path):
        logfile_name='predict'+datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        self.logger=Logger('../',logfile_name).get_logger()
        self.data_source=data_preprocessing(file_path)
        self.time_load_dict=self.data_source.set_index('time').power_load.to_dict()


def pred_feature_extract(data_dict, time, logger):
    """
    预测数据解析特征，保持与模型训练时的特征列名一致
    1.解析时间特征
    2.解析时间窗口特征
    3.解析昨日同时刻特征
    :param data_dict:历史数据，字典格式，key：时间，value:负荷
    :param time:预测时间，字符串类型，格式为2024-12-20 09:00:00
    :param logger:日志对象
    :return:
    """
    logger.info(f'=========解析预测时间为：{time}所对应的特征==============')
    # 特征列清单
    feature_names = ['hour_00', 'hour_01', 'hour_02', 'hour_03', 'hour_04', 'hour_05',
                     'hour_06', 'hour_07', 'hour_08', 'hour_09', 'hour_10', 'hour_11',
                     'hour_12', 'hour_13', 'hour_14', 'hour_15', 'hour_16', 'hour_17',
                     'hour_18', 'hour_19', 'hour_20', 'hour_21', 'hour_22', 'hour_23',
                     'month_01', 'month_02', 'month_03', 'month_04', 'month_05', 'month_06',
                     'month_07', 'month_08', 'month_09', 'month_10', 'month_11', 'month_12',
                     '前1小时', '前2小时', '前3小时', 'yesterday_load']

    pre_hour=time[11:13]
    hour_list=[]
    for i in range(24):
        if pre_hour==feature_names[i][5:7]:
            hour_list.append(1)
        else:
            hour_list.append(0)
    # print(hour_list)

    pre_month=time[5:7]
    month_list=[]
    for i in range(24,36):
        if pre_month==feature_names[i][6:8]:
            month_list.append(1)
        else:
            month_list.append(0)
    # print(month_list)

    last_1h_time=(pd.to_datetime(time)-pd.to_timedelta('1h')).strftime('%Y-%m-%d %H:%M:%S')
    last_1h_load=data_dict.get(last_1h_time,500)

    last_2h_time = (pd.to_datetime(time) - pd.to_timedelta('2h')).strftime('%Y-%m-%d %H:%M:%S')
    last_2h_load = data_dict.get(last_2h_time, 500)

    last_3h_time = (pd.to_datetime(time) - pd.to_timedelta('3h')).strftime('%Y-%m-%d %H:%M:%S')
    last_3h_load = data_dict.get(last_3h_time, 500)

    yesterday_time = (pd.to_datetime(time) - pd.to_timedelta('1D')).strftime('%Y-%m-%d %H:%M:%S')
    yesterday_load = data_dict.get(yesterday_time, 500)

    feature_data=hour_list+month_list+[last_1h_load,last_2h_load,last_3h_load,yesterday_load]
    # print(feature_data)
    feature_df=pd.DataFrame([feature_data],columns=feature_names)
    return feature_df

def prediction_plot(data):
    fig=plt.figure(figsize=(20,40))
    ax=fig.add_subplot()
    ax.plot(data['预测时间'],data['真实负荷'],label='真实负荷')
    ax.plot(data['预测时间'],data['预测负荷'],label='预测负荷')

    ax.set_title('真实负荷与预测负荷关系秃',fontsize=30)
    ax.set_xlabel('时间')
    ax.set_ylabel('负荷')
    ax.grid(True,linestyle='--',alpha=0.5)
    ax.legend(loc='best')

    ax.xaxis.set_major_locator(mick.MultipleLocator(base=50))
    plt.xticks(rotation=45)

    plt.savefig('../data/fig/真实负荷和预测负荷关系图.png')
    plt.show()

if __name__ == '__main__':
    pp=PowerLoadPredict('../data/test.csv')
    print(pp.data_source)
    print(pp.time_load_dict)
    print('*'*30)
    estimator=joblib.load('../model/xgb_20260328.pkl')

    pre_times=pp.data_source['time'][pp.data_source['time'] >= '2015-08-01 00:00:00']
    # print(pre_times)

    evaluate_list=[]

    #滚动预测，每次预测新值时，会获得前一小时的预测数据，并根据此进行预测
    for pre_time in pre_times:
        # print(f'正在预测{pre_time}:')
        time_load_dict_masked={k:v for k,v in pp.time_load_dict.items() if k < pre_time}
        # print(time_load_dict_masked)
        feature_df=pred_feature_extract(time_load_dict_masked,pre_time,pp.logger)

        y_pre=estimator.predict(feature_df)
        # print(y_pre)

        true_value=pp.time_load_dict.get(pre_time,500)

        evaluate_list.append([pre_time,true_value,y_pre[0]])

    evaluate_df=pd.DataFrame(evaluate_list,columns=['预测时间','真实负荷','预测负荷'])
    print(evaluate_df)

    print(f'平均绝对误差:{mean_absolute_error(evaluate_df['真实负荷'],evaluate_df['预测负荷'] ) } ')
    prediction_plot(evaluate_df)