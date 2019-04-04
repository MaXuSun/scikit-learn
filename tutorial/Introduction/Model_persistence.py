# 通过使用python 中自带的pinkle将训练的模型保存起来，然后再加载

from sklearn import svm
from sklearn import datasets
clf = svm.SVC(gamma='scale')
iris = datasets.load_iris()
X, Y = iris.data,iris.target
print(clf.fit(X,Y))

import pickle # 用来序列化对象或者变量到磁盘中
# pickle.dumps()方法把任意对象序列化成一个bytes，然后，就可以把这个bytes写入文件。
# 或者用另一个方法pickle.dump()直接把对象序列化后写入一个file-like Object：
# 先把内容读到一个bytes，然后用pickle.loads()方法反序列化出对象，
# 也可以直接用pickle.load()方法从一个file-like Object中直接反序列化出对象。
s = pickle.dumps(clf)
clf2 = pickle.loads(s)
clf2.predict(X[0:1])
print(clf2.predict(X[0:1]))

# 或者使用joblib来保存
from joblib import dump, load
dump(clf,'filename.joblib')
clf.load('filename.joblib')
