import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

file_path="./testdata/IMDB-Movie-Data.csv"
df=pd.read_csv(file_path)

genre_list=df["Genre"].str.split(",").explode().str.strip().unique()
print(genre_list)
# zeros_df=pd.DataFrame(np.zeros((df.shape[0],len(genre_list))),columns=genre_list)

genres_df = df["Genre"].str.replace(", ", ",").str.get_dummies(sep=",")
print(genres_df)

genres_count=genres_df.sum(axis=0)
genres_count=genres_count.sort_values()

_x=genres_count.index
print(_x)
_y=genres_count.values
print(_y)


plt.figure(figsize=(20,8),dpi=80)
plt.bar(range(len(_x)),_y)
plt.xticks(range(len(_x)),_x)
plt.show()