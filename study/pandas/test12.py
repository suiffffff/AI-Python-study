import pandas as pd
import  numpy as np
from matplotlib import pyplot as plt

file_path="./PM2.5/BeijingPM20100101_20151231.csv"
df=pd.read_csv(file_path)

print(df.info())


#period = pd.to_datetime(df[["year", "month", "day", "hour"]]).dt.to_period("h")

datetime_series = pd.to_datetime(df[["year", "month", "day", "hour"]])

period = pd.PeriodIndex(datetime_series, freq="h")

df["datetime"]=period
df.set_index("datetime",inplace=True)
df=df.resample("7D").mean(numeric_only=True)

data = df["PM_US Post"].dropna()
data_china=df["PM_Dongsi"].dropna()
print(data)
_x=data.index
_y=data.values

_x_china=data_china.index
_y_china=data_china.values


plt.figure(figsize=(20,8),dpi=80)

#plt.plot(range(len(_x)),_y,label="US")
#plt.plot(range(len(_x_china)),_y_china,label="CN")
plt.plot(data.index.to_timestamp(), data.values, label="US")
plt.plot(data_china.index.to_timestamp(), data_china.values, label="CN")

tick_positions = _x.to_timestamp()[::10]
tick_labels = _x.astype(str)[::10]
plt.xticks(tick_positions, tick_labels, rotation=45)

#plt.xticks(range(0,len(_x),10),list(_x)[::10],rotation=45)
#plt.xticks(range(0,len(_x_china),10),list(_x_china)[::10],rotation=45)

plt.legend()

plt.show()