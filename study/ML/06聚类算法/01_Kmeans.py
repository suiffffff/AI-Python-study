"""
    Kmeans简介：
        无监督学习，有特征无标签，根据样本相似性进行划分
        相似性可以理解为距离，如欧氏距离，曼哈顿距离，切比雪夫距离等
        在没有先备知识的情况下可能会使用
"""
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs #正态分布生成数据
from sklearn.metrics import calinski_harabasz_score

x,y=make_blobs(n_samples=1000,n_features=2,centers=[[-1,-1],[0,0],[1,1],[2,2]],cluster_std=[0.2,0.3,0.4,0.5],random_state=666)
print(x,y)
# plt.scatter(x[:,0],x[:,1])
# plt.show()

estimator=KMeans(n_clusters=4,random_state=666)
y_pre=estimator.fit_predict(x)
print(y_pre)
plt.scatter(x[:,0],x[:,1],c=y_pre)
plt.show()
print(calinski_harabasz_score(x,y_pre))