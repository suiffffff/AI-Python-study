"""
    聚类算法评估指标：
    思路1：SSE+肘部法
        SSE：
            概述：
                所有簇的所有样本导该簇质心的 误差平方和
            特点：
                随着K值增加，SSE值会逐渐减少
            目标：
                SSE值越小，代表簇内样本越聚集，内聚程度越高
        肘部法：
            K值增大，SSE值会随之减小，下降梯度徒然变缓的时候，K值就是我们要的最佳值

    思路2：SC轮廓系数
        考虑簇内 -> 聚集程度，越小越好
        考虑簇外 -> 分离程度，越大越好

    思路3：CH轮廓系数
        考虑簇内 -> 聚集程度，越小越好
        考虑簇外 -> 分离程度，越大越好
        考虑K值 ->  K值越小，代表簇内样本越聚集，内聚程度越高
"""
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import calinski_harabasz_score, silhouette_score


def dm01_sse():
    sse_list=[]
    x,y=make_blobs(
        n_samples=1000,
        n_features=2,
        centers=[[-1,-1],[0,0],[1,1],[2,2]],
        cluster_std=[0.2,0.3,0.4,0.5],
        random_state=666,
    )
    for k in range(1,100):
        estimator=KMeans(n_clusters=k,max_iter=100,random_state=666)
        estimator.fit(x)
        sse_value=estimator.inertia_
        sse_list.append((sse_value))
    print(sse_list)
    plt.figure(figsize=(20,10))
    plt.title('SSE value')


    plt.xticks(range(0,100,3))

    plt.xlabel('k')
    plt.ylabel('sse')
    plt.grid()

    plt.plot(range(1, 100), sse_list)
    plt.show()

def dm02_sc():
    sc_list=[]
    x,y=make_blobs(
        n_samples=1000,
        n_features=2,
        centers=[[-1,-1],[0,0],[1,1],[2,2]],
        cluster_std=[0.2,0.3,0.4,0.5],
        random_state=666,
    )
    for k in range(2,100):
        estimator=KMeans(n_clusters=k,max_iter=100,random_state=666)
        estimator.fit(x)
        y_pre=estimator.predict(x)
        sc_value=silhouette_score(x,y_pre)
        sc_list.append((sc_value))
    print(sc_list)
    plt.figure(figsize=(20,10))
    plt.title('sc value')


    plt.xticks(range(0,100,3))

    plt.xlabel('k')
    plt.ylabel('sc')
    plt.grid()

    plt.plot(range(2, 100), sc_list)
    plt.show()

def dm03_ch():
    ch_list=[]
    x,y=make_blobs(
        n_samples=1000,
        n_features=2,
        centers=[[-1,-1],[0,0],[1,1],[2,2]],
        cluster_std=[0.2,0.3,0.4,0.5],
        random_state=666,
    )
    for k in range(2,100):
        estimator=KMeans(n_clusters=k,max_iter=100,random_state=666)
        estimator.fit(x)
        y_pre=estimator.predict(x)
        ch_value=calinski_harabasz_score(x,y_pre)
        ch_list.append((ch_value))
    print(ch_list)
    plt.figure(figsize=(20,10))
    plt.title('ch value')


    plt.xticks(range(0,100,3))

    plt.xlabel('k')
    plt.ylabel('ch')
    plt.grid()

    plt.plot(range(2, 100), ch_list)
    plt.show()

if __name__ == '__main__':
    # dm01_sse()
    # dm02_sc()
    dm03_ch()