"""
归一化
高考赋分的模式
防止因为单位问题，导致特征列的方差相差较大
x' = （当前值-该列最小值）/（该列最大值-该列最小值）
x'' = x'*（mx-mi）+mi
"""

# from  sklearn.preprocessing import MinMaxScaler
#
# x_train=[
#     [90,2,10,40],
#     [60,4,15,45],
#     [75,3,13,46],
# ]
#
# scaler=MinMaxScaler(feature_range=(0,1))
#
# x_train_new=scaler.fit_transform(x_train)
#
# print('归一化后数据集：\n')
# print(x_train_new)

"""
标准化
同样的防止因为单位问题，导致特征列的方差相对较大，在数据量大时某些异常值基本无影响
x' =(当前值-该列平均值）/该列的标准差
标准差为方差求根
"""

from sklearn.preprocessing import StandardScaler

x_train=[
    [90,2,10,40],
    [60,4,15,45],
    [75,3,13,46],
]

transfer=StandardScaler()

x_train_new=transfer.fit_transform(x_train)

print('归一化后数据集：\n')
print(x_train_new)

print(f'均值:{transfer.mean_}')
print(f'方差:{transfer.var_}')
print(f'标准差:{transfer.scale_}')
