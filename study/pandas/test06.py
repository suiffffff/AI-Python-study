import numpy as np
import pandas as pd


file_path="./testdata/IMDB-Movie-Data.csv"
df=pd.read_csv(file_path)

# print(df.info())

# print(df.head(1))

print(df["Rating"].mean())
print(len(set(df["Director"].tolist())))
print(df["Director"].unique())
print(df["Director"].nunique())

temp_action_list=df["Actors"].str.split(",").tolist()
actors_list=[i.strip() for j in temp_action_list for i in j]
# np.array(temp_action_list).flatten()
actors_num=len(set(actors_list))

# actors_num = df["Actors"].str.split(",").explode().str.strip().nunique()
print(actors_num)