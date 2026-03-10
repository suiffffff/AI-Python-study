import pandas  as pd

df=pd.read_csv("./testdata/test.csv")

df=df.sort_values(by="使用次数",ascending=False)
print(df)