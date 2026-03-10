import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

file_path="./testdata/books.csv"
df=pd.read_csv(file_path)

print(df.info())

# data1=df[pd.notnull(df["original_publication_year"])]
data1 = df.dropna(subset=["original_publication_year"])
# print(data1)
# group=data1.groupby(by="original_publication_year")["title"].count()
# print(group)
group_mean=data1["average_rating"].groupby(by=data1["original_publication_year"]).mean()
print(group_mean)

_x=group_mean.index
_y=group_mean.values

plt.figure(figsize=(20,8),dpi=80)
plt.plot(range(len(_x)),_y)
# plt.xticks(list(range(len(_x)))[::10],_x[::10].astype(int),rotation=90)
plt.xticks(range(0, len(_x), 10), _x[::10].astype(int), rotation=90)

plt.show()