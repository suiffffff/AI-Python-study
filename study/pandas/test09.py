import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

file_path="./testdata/starbucks_store_worldwide.csv"

df=pd.read_csv(file_path)

print(df.info())

data1=df.groupby(by="Country")["Brand"].count().sort_values(ascending=False)[:10]

print(data1)
_x=data1.index
_y=data1.values

plt.figure(figsize=(20,8),dpi=80)

plt.bar(range(len(_x)),_y)
plt.xticks(range(len(_x)),_x)
plt.show()

