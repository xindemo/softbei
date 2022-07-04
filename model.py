# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 17:18:09 2022

@author: Administrator
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def judge_type(x, fee, count):
    if((x[0]>=fee) & (x[1]>=count)):
        type = "高价值型客户"
    elif((x[0]>=fee) & (x[1]<count)):
        type = "潜力型客户"
    elif((x[0]<fee) & (x[1]>=count)):
        type = "大众型客户"
    else:
        type = "低价值型客户"
    return type
def judge_type_int(x, fee, count):
    if((x[0]>=fee) & (x[1]>=count)):
        type = 0
    elif((x[0]>=fee) & (x[1]<count)):
        type = 1
    elif((x[0]<fee) & (x[1]>=count)):
        type = 2
    else:
        type = 3
    return type
def value_sum(x, y):
    return x+y
data = pd.read_excel("C:\\Users\\Administrator\\Desktop\\1647848272130494.xlsx")

df = pd.DataFrame(data)


#print(df.groupby("用户编号").groups)

group = df.groupby("用户编号").groups
count = df.groupby("用户编号").size()
count.name = '缴费次数'
count = pd.DataFrame(count)

sum_fee = df.groupby("用户编号").sum()
sum_count = pd.concat([sum_fee, count], axis=1)

avg_fee = df.groupby("用户编号").agg(np.mean)
avg_count = pd.concat([avg_fee, count], axis=1)

df2 = avg_count

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df2)
df3 = scaler.transform(df2)
df2_transform = pd.DataFrame(df3, index=df2.index, columns=df2.columns)
std_scale = pd.DataFrame(df2_transform)

fee = df.loc[:, "缴费金额（元）"].sum()/df.loc[:, "缴费日期"].size
count_n = df.loc[:, '缴费日期'].size/count.size
avg_fee_count = np.array([count.size, fee, count_n])

index = ["居民户数", "平均缴费金额", "平均缴费次数"]
column = ["金额/次数/户数"]
avg_fee_count = pd.DataFrame(avg_fee_count, index = index, columns=column)
#avg_fee_count.to_csv("C:\\Users\\Administrator\\Desktop\\居民客户的用电缴费习惯分析1.csv")
#df2 = df.sum()

avg_count['客户类型'] = avg_count[['缴费金额（元）', '缴费次数']].apply(lambda x : judge_type(list(x), fee, count_n),axis=1)
#avg_count.to_csv("C:\\Users\\Administrator\\Desktop\\居民客户的用电缴费习惯分析2.csv")

label = avg_count.loc[:, "客户类型"]

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

data2 = pd.concat([std_scale, label], axis=1) #标准化后用于聚类分析的数据
data3 = data2[data2["客户类型"] != "高价值型客户"]
data3["价值分数"] = data3["缴费金额（元）"]+data3["缴费次数"]
sort_data3 = data3.sort_values(by="价值分数", ascending=False)
most_value_top5 = sort_data3.iloc[0:5, :]
most_value_top5.to_csv("C:\\Users\\Administrator\\Desktop\\居民客户的用电缴费习惯分析3.csv")
X = data2.iloc[:, :2]
y = data2.iloc[:, 2]
y2 = pd.DataFrame(label)
y2 = y2.reset_index()
y2 = y2.iloc[:, 1:]
y3 = y2.values


#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

plt.scatter(X.iloc[:,0], X.iloc[:,1])
plt.title("row data")
inertia_scores = []
sil_scores = []
for n in range(2, 10):
    km = KMeans(n_clusters=n).fit(X, y)
    inertia_scores.append(km.inertia_)
    
    sc = silhouette_score(X, km.labels_)
    sil_scores.append(sc)
    print("n_clusters:{}\tinertia:{}\tsilhoutte_socres:{}".format(n, km.inertia_, sc))

#由上述silhoutte_score分数得：聚类簇数为2时最好，4次之，这里选择4为簇数
model = KMeans(n_clusters=4, random_state=0)
model.fit(X)
y_hat = model.predict(X)
c = model.labels_

#Fs = pd.Series(c, index=data2.index)
#Fs = Fs.sort_values(ascending=True)

#画图，把原始数据和最终预测数据在图上表现出来
'''
cm = mpl.colors.ListedColormap(list('rgby'))
plt.figure(figsize=(15,9), facecolor='w')
plt.subplot(121)  
plt.title('原始数据')  
plt.grid(True)  
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y3, s=30, cmap=cm, edgecolors='none',)  
plt.subplot(122)  
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=y_hat, s=30, cmap=cm, edgecolors='none')  
plt.title('K-Means算法聚类结果')  
plt.grid(True)  
plt.show()
'''
