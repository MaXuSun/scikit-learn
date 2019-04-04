# 这一块讲的是数据的降维

# 1. PCA
# 参考博客 https://www.cnblogs.com/eczhou/p/5433856.html  https://blog.csdn.net/qq_36523839/article/details/82558636
import numpy as np

x1 = np.random.normal(size=100)
x2 = np.random.normal(size=100)
x3 = x1+x2
X = np.c_[x1,x2,x3]
print(X.shape,x1.shape,x2.shape)

from sklearn import decomposition
pca = decomposition.PCA()
print(pca.fit(X))
print(pca.explained_variance_,pca.explained_variance_ratio_)# 输出pca每个特征的方差和方差占所有特征的百分比

# 降维
pca.n_components = 2
X_reduced = pca.fit_transform(X)
print(X_reduced.shape)

# 2. ICA 独立成分分析(没有学，具体也不是多清楚)
# 参考博客：https://blog.csdn.net/sinat_37965706/article/details/71330979

from scipy import signal
import numpy as np
time = np.linspace(0,10,2000)  # 构建等差数列
s1 = np.sin(2*time)
s2 = np.sign(np.sin(3*time))
s3 = signal.sawtooth(2*np.pi*time)
S = np.c_[s1,s2,s3]
S +=0.2*np.random.normal(size=S.shape) # 添加噪音
S /= S.std(axis=0)  # 标准化数据

A = np.array([[1,1,1],[0.5,2,1],[1.5,1,2]])
X = np.dot(S,A.T)

ica = decomposition.FastICA()
S_ = ica.fit_transform(X)
A = ica.mixing_.T
np.allclose(X,np.dot(S_,A_)+ica.mean_)