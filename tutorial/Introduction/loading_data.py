# 加载样例数据
from sklearn import datasets
iris = datasets.load_iris()  # only data
digits = datasets.load_iris() # data and target,each's shape is (8*8)
# print(iris)
# print(digits.data)
print(digits.images[0])
print(dir(digits))

# 加载其他数据需要使用numpy加载后再传到sklearn里面