from sklearn import svm
from sklearn import datasets
# sklearn 中每个预测期都是一个包含了fir(X,Y)和predict(T)的对象
# 这里只是用支持向量机作为例子
# sklearn 中每个实例的构造函数传入的参数都是对应模型中不能训练得到的参数，比如正则项
# 可使用 grid search 和cross validation等方法进行最优参数的尝试

digits = datasets.load_digits()
clf = svm.SVC(gamma=0.001,C = 100.)
clf.fit(digits.data[:-1],digits.target[:-1]) # 将除最后一个数据传入fit中进行训练
print(clf) # 打印训练结果
predict = clf.predict(digits.data[-1:])    # 得到预测结果
print(predict)
