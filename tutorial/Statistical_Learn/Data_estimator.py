from sklearn import datasets
# sklearn 中的数据需要是 (n_samples,n_features)这种格式，其他格式需要转为这种格式
iris = datasets.load_iris()
data = iris.data
print(data.shape)

digits = datasets.load_digits()
print(digits.images.shape)# (1797,8,8)
print(digits.data.shape)  # (1797,64)
import matplotlib.pyplot as plt
# plt.imshow(digits.images[-1],cmap=plt.cm.gray_r)
# plt.show()
# 将image转为 (1797,64),其实可以直接用里面的 digits.data数据
data = digits.images.reshape((digits.images.shape[0],-1))

# estimator = Estimator(param1=1,param2=2)
# estimator.fit(data)
# estimator.estimated_param_ # 预测器中所有参数都是带下划线的