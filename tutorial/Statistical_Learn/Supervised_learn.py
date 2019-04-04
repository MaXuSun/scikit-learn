# 用sklearn 进行分类和回归任务

# 最近邻和维数诅咒
import numpy as np
from sklearn import datasets
iris = datasets.load_iris()
iris_X = iris.data
iris_Y = iris.target
print(iris.feature_names)   # 查看4个特征名称
print(np.unique(iris_Y))
# unique返回数组里面有的值,去除重复的，如下：
# A = [1, 2, 2, 3, 4, 3]
# a = np.unique(A)
# print a            # 输出为 [1 2 3 4]
# a, b, c = np.unique(A, return_index=True, return_inverse=True)
# print a, b, c      # 输出为 [1 2 3 4], [0 1 3 4], [0 1 1 2 3 2]

## kNN
np.random.seed(0)
indices = np.random.permutation(len(iris_X))
# permutation函数对原来数组进行洗牌并且返回新数组，shuffle直接洗牌原来的数组
iris_X_train = iris_X[indices[:-10]]
iris_Y_train = iris_Y[indices[:-10]]
iris_X_test = iris_X[indices[-10:]]
iris_Y_test = iris_Y[indices[-10:]]

from sklearn.neighbors import KNeighborsClassifier  # 创建一个临近分类器
knn = KNeighborsClassifier()
print(knn.fit(iris_X_train,iris_Y_train))
print(knn.predict(iris_X_test))
print(iris_Y_test)

# 线性模型，将回归变为稀疏
# 使用糖尿病数据集,(422,10),预测疾病的生理变量进展
diabetes = datasets.load_diabetes()
diabetes_X_train = diabetes.data[:-20]
diabetes_X_test = diabetes.data[-20:]
diabetes_Y_train = diabetes.target[:-20]
diabetes_Y_test = diabetes.target[-20:]

## 线性回归,直线模拟
from sklearn import linear_model
regr = linear_model.LinearRegression()
regr.fit(diabetes_X_train,diabetes_Y_train)
print(regr.coef_)
pre_mean = np.mean((regr.predict(diabetes_X_test) - diabetes_Y_test)**2)
print(pre_mean)
print(regr.score(diabetes_X_test, diabetes_Y_test))

## 收缩(shrinkage)，因为没学，不懂，暂时没看   ridge regression

## 稀疏(sparsity),没学，也没看

# 逻辑回归
log = linear_model.LogisticRegression(solver='lbfgs',C=1e5,multi_class='multinomial')
print(log.fit(iris_X_train,iris_Y_train))
print(log.predict(iris_X_test))

# SVM支持向量机
# 其中参数C，小值意味着使用决策面附件较多或者所有观察计算边界;大值意味着使用较少的值计算
# 具体代码看 plot_iris_exercise.py 和 plot.iris.py 代码
