"""
    贝叶斯：仅仅依靠概率就可以分类的一种机器学习的算法
    朴素：不考虑特征之间的关联性，特征之间相互独立
        原始：P(AB)=P(A)*P(B|A)=P(B)*P(A|B)
        加入朴素后:P(AB)=P(A)*P(B)

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB

df=pd.read_csv('./data/书籍评价.csv',encoding='gbk')
df.info()
df['labels']=np.where(df['评价']=='好评',1,0)
y=df['labels']

commit_list=[",".join(jieba.lcut(line))  for line in df["内容"]]

with  open('./data/stopwords.txt',encoding='utf-8') as src_f:
    stopwords_list=src_f.readlines()
    stopwords_list=[ line.strip() for line in stopwords_list]
    stopwords_list=list(set(stopwords_list))

transfer=CountVectorizer(stop_words=stopwords_list)
transfer.fit(commit_list)
x=transfer.transform(commit_list).toarray()

x_train=x[:10]
y_train=y[:10]
x_test=x[10:]
y_test=y[10:]
print(transfer.get_feature_names_out())
print(len(transfer.get_feature_names_out()))

estimator=MultinomialNB()
estimator.fit(x_train,y_train)

y_pre=estimator.predict(x_test)
print(f'准确率{accuracy_score(y_test,y_pre)}')