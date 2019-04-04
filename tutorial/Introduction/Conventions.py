# scikit-learn中的预测器遵循某些规则，使其行为更具预测性
import numpy as np
from sklearn import random_projection

# 无特殊说明，使用float32位，这里通过fit_transform 将其转为float64
rng = np.random.RandomState(0)
X = rng.rand(10,2000)
X = np.array(X,dtype='float32')
print(X.dtype)

transformer = random_projection.GaussianRandomProjection()
X_new = transformer.fit_transform(X)
print(X_new.dtype)


# fit传入不同的Y,得到测预测结果不同,第一个传入target,第二个传入target_names
from sklearn import datasets
from sklearn.svm import SVC
iris = datasets.load_iris()
clf = SVC(gamma='scale')
print(clf.fit(iris.data,iris.target))
print(clf.predict(iris.data[:3]))
print(clf.fit(iris.data,iris.target_names[iris.target]))
print(clf.predict(iris.data[:3]))

# 调用构造器后同样可以使用set_params()函数重新设置超参数，每次调用fit函数都会覆盖上面得到的结果
import numpy as np
from sklearn.svm import  SVC
rng = np.random.RandomState(0)
X = rng.rand(100,10)
Y = rng.binomial(1,0.5,100)
X_test = rng.rand(5,10)

clf = SVC()
print(clf.set_params(kernel='linear').fit(X,Y))
print(clf.predict(X_test))
print(clf.set_params(kernel='rbf',gamma='scale').fit(X,Y))
print(clf.predict(X_test))

# 进行多类别预测
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelBinarizer

X = [[1,2],[2,4],[4,5],[3,2],[3,1]]
y = [0,0,1,1,2]
classif = OneVsRestClassifier(estimator=SVC(gamma='scale',random_state=0))
print(classif.fit(X,y).predict(X))    # 1维多类标签数组
y = LabelBinarizer().fit_transform(y)
print('y',y)
print(classif.fit(X,y).predict(X))    # 在y的二进制标签表示上拟合，predict返回表示相应多标记预测的二维数组
# 第4和5个实例返回全零表示它们不匹配三个适合的标签

# 下面的是适合一个样本对应多个标签的表示结果，其结果使用二维表示，同上
from sklearn.preprocessing import MultiLabelBinarizer
y = [[0,1],[0,2],[1,3],[0,2,3],[2,4]]
y = MultiLabelBinarizer().fit_transform(y)
print(classif.fit(X,y).predict(X))