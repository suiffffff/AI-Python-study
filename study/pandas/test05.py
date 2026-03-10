import pandas as pd
import numpy as np
from  matplotlib import pyplot as plt

file_path="./testdata/IMDB-Movie-Data.csv"

df=pd.read_csv(file_path)

rating_data=df["Rating"].values

max_rating=rating_data.max()
min_rating=rating_data.min()

num_bin_list=[0.5]*13+[0.6]


plt.figure(figsize=(20,8),dpi=80)

_x = np.arange(min_rating, max_rating + 0.5, 0.5)

plt.hist(rating_data,_x)
plt.xticks(_x)
plt.show()