# 构造一个流水线的预测器，就是使用Pipeline将多个预测器放到一起
# 然后使用网格搜索找到这些预测器超参数的最佳组合
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
# 1. 先将要使用的estimator创建出来
logistic = SGDClassifier(loss='log',penalty='l2',early_stopping=True,max_iter=10000,tol=1e-5,random_state=0)
pca = PCA()
pipe = Pipeline(steps=[('pca',pca),('logistic',logistic)])

# 2. 得到需要的数据
digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target

# 3.使用网格搜索找到最佳参数，可使用'__'连接estimator和其对应的参数
param_grid = {
    'pca__n_components':[5,20,30,40,50,64],
    'logistic__alpha':np.logspace(-4,4,5),
}
search = GridSearchCV(pipe,param_grid,iid=False,cv=5,return_train_score=False)
search.fit(X_digits,y_digits)
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)

# 绘制PCA谱图
pca.fit(X_digits)

fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, figsize=(6, 6))
ax0.plot(pca.explained_variance_ratio_, linewidth=2)
ax0.set_ylabel('PCA explained variance')
