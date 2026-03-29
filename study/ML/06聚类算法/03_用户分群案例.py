from cProfile import label

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import calinski_harabasz_score,silhouette_score
import pandas as pd
import numpy as np



def dm01_find_k():
    df=pd.read_csv('./data/customers.csv')
    # df.info()
    # print(df.head())
    sse_list=[]
    sc_list=[]
    x=df.iloc[:,3:5]
    for k in range(2,20):
        estimator=KMeans(n_clusters=k,max_iter=100,random_state=666)
        estimator.fit(x)
        y_pre=estimator.predict(x)
        sse_list.append(estimator.inertia_)
        sc_list.append(silhouette_score(x,y_pre))

    plt.figure(figsize=(20,10))
    plt.plot(range(2,20),sse_list,label='SSE')
    plt.show()

    plt.figure(figsize=(20, 10))
    plt.plot(range(2, 20), sc_list, label='SC')
    plt.show()

def dm02_train_predict_evaluate():
    df=pd.read_csv('./data/customers.csv')
    x = df.iloc[:, 3:5]
    print(df.head())
    estimator=KMeans(n_clusters=5,max_iter=100,random_state=666)
    estimator.fit(x)
    y_pre=estimator.predict(x)

    plt.figure(figsize=(20,10))

    plt.scatter(x.values[y_pre==0,0],x.values[y_pre==0,1])
    plt.scatter(x.values[y_pre==1,0],x.values[y_pre==1,1])
    plt.scatter(x.values[y_pre==2,0],x.values[y_pre==2,1])
    plt.scatter(x.values[y_pre==3,0],x.values[y_pre==3,1])
    plt.scatter(x.values[y_pre==4,0],x.values[y_pre==4,1])

    plt.scatter(estimator.cluster_centers_[:,0],estimator.cluster_centers_[:,1])
    plt.title('Cluster of Customers')
    plt.xlabel('Annual Income')
    plt.ylabel('Spending Score(1-100)')
    plt.show()

if __name__ == '__main__':
    # dm01_find_k()
    # dm02_train_predict_evaluate()

    #x.values[y_pre==0,0] 解释
    #这里用到了np数组的根据bool索引取值，true为取，false不取，但数组个数要和bool个数一致
    x=[[1,1],[2,2],[3,3],[4,4],[5,5],[6,6]]
    x2=np.array(x)
    result=x2[[True,True,False,False,True,False]]
    print(result)