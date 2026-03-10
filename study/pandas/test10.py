import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import font_manager


my_font=font_manager.FontProperties(fname="/Windows/Fonts/SIMHEI.TTF")
file_path="./testdata/starbucks_store_worldwide.csv"

df=pd.read_csv(file_path)
df=df[df["Country"]=="CN"]
print(df.head(1))



data1=df.groupby(by="City")["Brand"].count().sort_values(ascending=False)[:25]
print(data1)

_x=data1.index
_y=data1.values

plt.figure(figsize=(20,12),dpi=80)

# plt.bar(range(len(_x)),_y,width=0.3)
plt.barh(range(len(_x)),_y,height=0.3)
# plt.xticks(range(len(_x)),_x,fontproperties=my_font)
plt.yticks(range(len(_x)),_x,fontproperties=my_font)

plt.show()


