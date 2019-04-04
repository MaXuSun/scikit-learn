# 1. 直接预测得分和交叉预测得分
from sklearn import datasets,svm
digits = datasets.load_digits()
X_digits = digits.data
y_digits = digits.target
svc = svm.SVC(C=1,kernel='linear')
svc.fit(X_digits[:100],y_digits[:100])
print(svc.score(X_digits[-100:],y_digits[-100:])) # 分值越高说明预测效果越好

# 2. 为了得到好的预测效果，我们可以通过将数据集进行划分，进行交叉验证：不动测试集情况下进行评估
# 这里实现的是 K-fold 交叉验证方法，k个子集，每个子集均做一次测试集，其余作为训练集，交叉验证重复k次，选区平均值作为结果
import numpy as np
X_folds = np.array_split(X_digits,3)
y_folds = np.array_split(y_digits,3)
# array_split可以实现不均等划分,np.split只能实现均等划分
# 也可这样使用[0 - 8]  -- np.split([3,5,6,10])-->[0,1,2],[3,4],[5],[6,7]
scores = list()
for k in range(3):
    X_train = list(X_folds)
    X_test = X_train.pop(k)
    X_train = np.concatenate(X_train) # np.concatenate是数组连接函数
    y_train = list(y_folds)
    y_test = y_train.pop(k)
    y_train = np.concatenate(y_train)
    scores.append(svc.fit(X_train,y_train).score(X_test,y_test))
print(scores)


# 3. 交叉验证生成器
# sklearn 自带一类交叉验证器(Cross-validation generators):KFold,StratifiedKFold,GroupKFold,ShuffleSplit,StratifiedShuffleSplit,
# GroupShuffleSplit,LeaveOneGrouupOut,LeavePGroupOut,LeaveOneOut,LeavePOut,PredefinedSplit
from sklearn.model_selection import KFold,cross_val_score
X = ['a','a','a','b','b','c','c','c','c','c']
k_fold = KFold(n_splits=3)
for train_indices,test_indices in k_fold.split(X):
    print('Train: %s | test: %s' % (train_indices, test_indices))
# 下面一行相当于上面 2中 自己实现的K-fold交叉验证
print([svc.fit(X_digits[train],y_digits[train]).score(X_digits[test],y_digits[test]) for train,test in k_fold.split(X_digits)])

# 更简单的是连for循环都不用写，一行代码即可
print(cross_val_score(svc,X_digits,y_digits,cv=k_fold,n_jobs=-1)) #n_jobs表示将任务分配计算机上所有CPU进行计算
print(cross_val_score(svc,X_digits,y_digits,cv=k_fold,scoring='precision_macro')) #scoring=参数可以指定评估的方法

# 练习，见plot_cv_digits

# 4. 网格搜索和交叉验证生成器
# sklearn 提供GridSearchCV模块，可以在不同超参数组合中找到最优组合,结果以字典形式保存在clf.cv_results中
# 可查看 https://www.cnblogs.com/nwpuxuezha/p/6618205.html
from sklearn.model_selection import GridSearchCV,cross_val_score
Cs = np.logspace(-6,-1,10)
# parameters = {'kernel':('linear', 'rbf'), 'C':[1, 2, 4], 'gamma':[0.125, 0.25, 0.5 ,1, 2, 4]}
# 创造等比数列[10^-6,…,10^-1],可通过np.logspace(-6,-1,10,base=2)来更改底数

# clf = GridSearchCV(svc,parameters,n_jobs=-1)
# clf = GridSearchCV(svc,dict(Cs),n_jobs=-1,cv=3)
clf = GridSearchCV(estimator=svc,param_grid=dict(C=Cs),n_jobs=-1)

# 构造参数cv,默认是3,传入整数 k 代表 k 折交叉验证，分类任务使用StratifiedFold,其他使用KFold
print(clf.fit(X_digits[:1000],y_digits[:1000]))
print(clf.best_score_,clf.best_estimator_.C,clf.best_params_)

# 5.交叉验证预测器
# sklearn 中一些预测器将cv暴露出来(可以看出是后面带CV的estimator)，通过设置cv可以自己进行交叉评估验证
from sklearn import linear_model,datasets
lasso = linear_model.LassoCV(cv=3)
diabetes = datasets.load_diabetes()
X_diabetes = diabetes.data
Y_diabetes = diabetes.target
lasso.fit(X_diabetes,Y_diabetes)
print(lasso.alpha_)

# 练习参见plot_cv_diabetes.py