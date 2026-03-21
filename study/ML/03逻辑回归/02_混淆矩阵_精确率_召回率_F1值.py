"""
混淆矩阵：
                        预测标签（正例） 预测标签（反例）
        真实标签（正例）    真正例（TP）    伪反例（FN）
        真实标签（反例）    伪正例（FP）    真反例（TN）
        T：true F：False P：Positive N：Negative

结论：
        1.默认使用分类较少的充当正例
        2.精确率：真正例在预测正例中的占比 TP/TP+FP
        3.召回率：真正例在真正例中的占比 TP/TP+FP
        4.F1值：2*精确率*召回率/精确率+召回率

        预测对的不一定全（精确率），预测全的不一定对（召回率） 
"""
from operator import index

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score

y_train = ['恶性','恶性','恶性','恶性','恶性','恶性',   '良性','良性','良性','良性']


y_pre_a = ['恶性','恶性','恶性','良性','良性','良性',   '良性','良性','良性','良性']
y_pre_b = ['恶性','恶性','恶性','恶性','恶性','恶性',   '良性','恶性','恶性','恶性']

label =['恶性','良性']
df_label=['恶性(正例)','良性(反例)']

cm_a=confusion_matrix(y_train,y_pre_a,labels=label)
print(cm_a)

df_a=pd.DataFrame(cm_a,index=df_label,columns=df_label)
print(df_a)

print(f'a 精确率：{precision_score(y_train,y_pre_a,pos_label='恶性')}')
print(f'a 召回率：{recall_score(y_train,y_pre_a,pos_label='恶性')}')
print(f'a F1值：{f1_score(y_train,y_pre_a,pos_label='恶性')}')

cm_b=confusion_matrix(y_train,y_pre_b,labels=label)
print(cm_b)
df_b=pd.DataFrame(cm_b,index=df_label,columns=df_label)
print(df_b)
print(f'b 精确率：{precision_score(y_train,y_pre_b,pos_label='恶性')}')
print(f'b 召回率：{recall_score(y_train,y_pre_b,pos_label='恶性')}')
print(f'b F1值：{f1_score(y_train,y_pre_b,pos_label='恶性')}')
