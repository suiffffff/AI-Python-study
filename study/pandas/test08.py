import pandas as pd
import numpy as np

file_path="./testdata/starbucks_store_worldwide.csv"
df=pd.read_csv(file_path)

# print(df.head(1))
# print(df.info())

# group=df.groupby(by="Country")
# for i,j in group:
#     print(i)
#     print("-"*100)
#     print(j)
#     print("*"*100)

# print(group.count())
# country_count=group["Brand"].count()
# print(country_count)
# print(country_count["US"])
# print(country_count["CN"])

# china_data=df[df["Country"]=="CN"]
# group=china_data.groupby(by="State/Province")["Brand"].count()
# print(group)

# group=df["Brand"].groupby(by=[df["Country","State/Province"]]).count()
# print(group)
# print(type(group))

group=df.groupby(by=["Country","State/Province"])[["Brand"]].count()
print(group)
print(type(group))
print(group.index)