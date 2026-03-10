import pandas as pd
from matplotlib import pyplot as plt
file_path="./testdata/IMDB-Movie-Data.csv"

df=pd.read_csv(file_path)
print(df.head(1))
print(df.info())

runtime_data=df["Runtime (Minutes)"].values

max_runtime=runtime_data.max()
min_runtime=runtime_data.min()

num_bin=(max_runtime-min_runtime)//5

plt.figure(figsize=(20,8),dpi=80)
plt.hist(runtime_data,num_bin)

plt.xticks(range(min_runtime,max_runtime+5,5))

plt.show()
