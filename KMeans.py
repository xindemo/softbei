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
group = data.groupby("用户编号").groups
count = data.groupby("用户编号").size()
count.name = '缴费次数'
count = pd.DataFrame(count)

avg_fee = df.groupby("用户编号").agg(np.mean)
avg_count = pd.concat([avg_fee, count], axis=1)

df2 = avg_count

fee = data.loc[:, "缴费金额（元）"].sum()/data.loc[:, "缴费日期"].size
count_n = data.loc[:, '缴费日期'].size/count.size

avg_count['0'] = avg_count[['缴费金额（元）', '缴费次数']].apply(lambda x : judge_type_int(list(x), fee, count_n),axis=1)

label = avg_count.loc[:, "0"]

# 对原始数据进行均值方差标准化，以进行下一步的聚类分析
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df2)
df3 = scaler.transform(df2)
df2_transform = pd.DataFrame(df3, index=df2.index, columns=df2.columns)
std_scale = pd.DataFrame(df2_transform)

from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

X = data2.iloc[:, :2]
y = data2.iloc[:, 2]
y2 = pd.DataFrame(label)
y2 = y2.reset_index()
y2 = y2.iloc[:, 1:]
y3 = y2.values


#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# 绘制原始数据点图
plt.scatter(X.iloc[:,0], X.iloc[:,1])
plt.title("row data")
# 通过inertia指标和轮廓系数silhouette_score确定聚类的簇数
inertia_scores = []
sil_scores = []
for n in range(2, 10):
    km = KMeans(n_clusters=n).fit(X, y)
    inertia_scores.append(km.inertia_)
    
    sc = silhouette_score(X, km.labels_)
    sil_scores.append(sc)
    print("n_clusters:{}\tinertia:{}\tsilhoutte_socres:{}".format(n, km.inertia_, sc))

#由上述silhoutte_score分数得：聚类簇数为2时最好，4次之，这里选择4为簇数以和我们自定义的四类标签进行对比
model = KMeans(n_clusters=4, random_state=0)
model.fit(X)
y_hat = model.predict(X)
c = model.labels_

#画图，把原始数据和最终预测数据在图上表现出来
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
