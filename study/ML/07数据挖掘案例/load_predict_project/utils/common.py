import pandas as pd

#对数据预处理 -> 时间格式化，按照时间升序排列，对数据去重
#train.csv  -> 拆分训练集和测试集
#test.csv   -> 不拆分

def data_preprocessing(file_path):
    data=pd.read_csv(file_path)
    data.info()

    # '%Y-%m-%d %H:%M:%S'
    data['time']=pd.to_datetime(data['time']).dt.strftime('%Y-%m-%d %H:%M:%S')
    # print(data.head())

    data.sort_values('time',ascending=True,inplace=True)

    data.drop_duplicates(inplace=True)
    return data


if __name__ == '__main__':
    data_preprocessing()