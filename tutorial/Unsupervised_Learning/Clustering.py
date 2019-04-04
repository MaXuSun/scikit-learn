# 1. K-means 聚类
from sklearn import cluster,datasets
iris = datasets.load_iris()
X_iris = iris.data
y_iris = iris.target

k_means = cluster.KMeans(n_clusters=3)
print(k_means.fit(X_iris))
print(k_means.labels_[::10])
print(y_iris[::10])             #注意这里得到标签和原有标签编码不同，但是每个编码都代表一个类

# 2. 使用K-means进行矢量量化，比如对图片进行色调分离：将图片中主要的集中颜色提取出来，然后只用这几种颜色表示；
# 或者将255*255*255种颜色压缩到16中颜色，这里可以使用分布直方图
# 具体可以看 https://blog.csdn.net/jasonzhoujx/article/details/81942106 https://www.cnblogs.com/shixisheng/p/7116045.html

# 3. 分层聚类：凝聚(自下而上)；分裂(自上而下)。具体理论在机器学习中已经学习

# 4. 连通性约束的聚类
import matplotlib.pyplot as plt

from skimage.data import coins
from skimage.transform import rescale

from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import AgglomerativeClustering

orig_coins = coins()
# 使用高斯滤波对其进行平滑，然后缩小便于处理
# smoothened_coins = gaussian_filter(orig_coins, sigma=2)   # 这个高斯滤波有问题
# rescaled_coins = rescale(smoothened_coins, 0.2, mode="reflect")
# X = np.reshape(rescaled_coins, (-1, 1))
# connectivity = grid_to_graph(*rescaled_coins.shape)

# 5. 特征聚集(将类似的特征合并在一起)  看的有点迷糊，先把代码粘下来吧
import numpy as np
digits = datasets.load_digits()
images = digits.images
X = np.reshape(images, (len(images), -1))
connectivity = grid_to_graph(*images[0].shape)

agglo = cluster.FeatureAgglomeration(connectivity=connectivity,
                                     n_clusters=32)
agglo.fit(X)

X_reduced = agglo.transform(X)

X_approx = agglo.inverse_transform(X_reduced)
images_approx = np.reshape(X_approx, images.shape)