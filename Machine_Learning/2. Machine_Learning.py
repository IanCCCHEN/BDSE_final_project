
# 3.1 訓練模型

# 3.1.1 將filter與wrapper法 篩選的結果匯入
import numpy as np
import pandas as pd


# 3.2.1 (GBM) Gradient Tree Boosting https://scikit-learn.org/stable/modules/ensemble.html?highlight=gbm
# API https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier

# from sklearn.ensemble import GradientBoostingClassifier
# clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
# clf.fit(df_x, df_y)
# clf.score(df_x, df_y)
# print(clf.predict_proba(df_x))
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.5, random_state=100)


clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
clf.fit(X_train,  y_train)
clf.predict(X_test)

clf.score(X_test, y_test)
clf.predict_proba(X_train)

#3.2.2 sklearn.ensemble.RandomForestClassifier API  https://scikit-learn.org/0.24/modules/generated/sklearn.ensemble.RandomForestClassifier.html?highlight=randomforest#sklearn.ensemble.RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X_train,  y_train)
clf.predict(X_test)
clf.predict_proba(X_test)
clf.score(X_train, y_train)
clf.score(X_test, y_test)

# 3.3.1 sklearn.model_selection.GridSearchCV 網格搜尋 
# RandomizedSearchCV的使用方法其實是和GridSearchCV一致的 https://tw511.com/a/01/8581.html
# https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
# from sklearn.model_selection import GridSearchCV



# 4 AUC-ROC sklearn.metrics.plot_roc_curve API https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_roc_curve.html

import matplotlib.pyplot as plt
from sklearn import metrics
metrics.plot_roc_curve(clf, X_test, y_test) 

plt.show()


# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 10:18:44 2021

@author: Kevin Dai
"""
from xgboost import XGBClassifier #scikit learn API https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#XGB 調整參數順序 https://blog.csdn.net/han_xiaoyang/article/details/52665396
#GBM 餐討調餐程序 https://blog.csdn.net/han_xiaoyang/article/details/52663170

# 導入數據集轉化成90個特徵
df = pd.read_csv('all_features_train_OneHot.csv')

Features_bf_filter = pd.read_csv('ROC_ANOVA_intersection_100in90.csv')
Features_bf_filter = Features_bf_filter.iloc[:,1].tolist()
Features_bf_filter
data_90 = df[Features_bf_filter]
target = df['label']
test = pd.read_csv('all_features_test_OneHot.csv')
test_90 = test[Features_bf_filter]
#Xgboost 的三種參數
# 通用参数：宏观函数控制。
# Booster参数：控制每一步的booster(tree/regression)。
# 学习目标参数：控制训练目标的表现。

# 1. 通用參數調整

# 1.1 booster[默認gbtree] - gbtree 基於樹的模型、gbliner 線性模型 OK
# ☆ 選擇 gbtree 

# 1.2 silent[默认0] OK
# 当这个参数值为1时，静默模式开启，不会输出任何信息。
# 一般这个参数就保持默认的0，因为这样能帮我们更好地理解模型。
# ☆ 選擇 0 
# 1.3 nthread[默认值为最大可能的线程数] OK
# 这个参数用来进行多线程控制，应当输入系统的核数。
# 如果你希望使用CPU全部的核，那就不要输入这个参数，算法会自动检测它。
# ☆ 不用設定任由默認達到最快

# 2. booster参数調整

# 2.1 eta[默认0.3] 與GBM Learning_rate相似 OK - 0.3
#和GBM中的 learning rate 参数类似。
# 通过减少每一步的权重，可以提高模型的鲁棒性(Robustness) as 穩健程度。
# 典型值为0.01-0.2。
# ☆ 看來可以在典型值範圍內找0.01為區間去搜尋
# 2.2 min_child_weight[默认1]  0.9999999999999999
# 决定最小叶子节点样本权重和。
# 和GBM的 min_child_leaf 参数类似，但不完全一样。XGBoost的这个参数是最小样本权重的和，而GBM参数是最小样本总数。
# 这个参数用于避免过拟合。当它的值较大时，可以避免模型学习到局部的特殊样本。
# 但是如果这个值过高，会导致欠拟合。这个参数需要使用CV来调整。
# ☆ 不能太大不能太小 還是要再找參考值
# 2.3 max_depth[默认6]  2 
# 和GBM中的参数相同，这个值为树的最大深度。
# 这个值也是用来避免过拟合的。max_depth越大，模型会学到更具体更局部的样本。
# 需要使用CV函数来进行调优。
# 典型值：3-10
# ☆ 典型值區間調整1
# 2.4 max_leaf_nodes
# 树上最大的节点或叶子的数量。
# 可以替代max_depth的作用。因为如果生成的是二叉树，一个深度为n的树最多生成n^2的葉子 
# 如果定义了这个参数，GBM会忽略max_depth参数。
# ☆ 看來是可以不調整
# 2.5 gamma[默认0] 0
# 在点分裂时，只有分裂后损失函数的值下降了，才会分裂这个节点。Gamma指定了节点分裂所需的最小损失函数下降值。
# 这个参数的值越大，算法越保守。这个参数的值和损失函数息息相关，所以是需要调整的。
# ☆ 沒有具體講要怎麼調，還要再查資料
# 2.6 max_delta_step[默认0] 0
# 这参数限制每棵树权重改变的最大步长。如果这个参数的值为0，那就意味着没有约束。
# 如果它被赋予了某个正值，那么它会让这个算法更加保守。
# 通常，这个参数不需要设置。但是当各类别的样本十分不平衡时，它对逻辑回归是很有帮助的。
# 这个参数一般用不到，但是你可以挖掘出来它更多的用处。
# ☆ 不限制步長，所以不調整
# 2.7 subsample[默认1] 0.5700000000000001 - 90 features  0.9500000000000004 -87 features
# 和GBM中的subsample参数一模一样。这个参数控制对于每棵树，随机采样的比例。
# 减小这个参数的值，算法会更加保守，避免过拟合。但是，如果这个值设置得过小，它可能会导致欠拟合。
# 典型值：0.5-1
# ☆ 中途進行了 XGBimportance特徵篩選 所以 用兩個資料集表示
# 2.8 colsample_bytree[默认1]
# 和GBM里面的max_features参数类似。用来控制每棵随机采样的列数的占比(每一列是一个特征)。
# 典型值：0.5-1
# ☆ 
# 2.9 colsample_bylevel[默认1]
# 用来控制树的每一级的每一次分裂，对列数的采样的占比。
# 我个人一般不太用这个参数，因为subsample参数和colsample_bytree参数可以起到相同的作用。但是如果感兴趣，可以挖掘这个参数更多的用处。
# ☆ 
# 2.10、lambda[默认1]
# 权重的L2正则化项。(和Ridge regression类似)。
# 这个参数是用来控制XGBoost的正则化部分的。虽然大部分数据科学家很少用到这个参数，但是这个参数在减少过拟合上还是可以挖掘出更多用处的。
# ☆ 
# 2.11、alpha[默认1]
# 权重的L1正则化项。(和Lasso regression类似)。
# 可以应用在很高维度的情况下，使得算法的速度更快。
# ☆ 
# 2.12 scale_pos_weight[默认1]
# 在各类别样本十分不平衡时，把这个参数设定为一个正值，可以使算法更快收敛。
# ☆ 
# 3 学习目标参数
# 3.1 objective[默认reg:linear]
# 这个参数定义需要被最小化的损失函数。最常用的值有：
# binary:logistic 二分类的逻辑回归，返回预测的概率(不是类别)。
# multi:softmax 使用softmax的多分类器，返回预测的类别(不是概率)。
# 在这种情况下，你还需要多设一个参数：num_class(类别数目)。
# multi:softprob 和multi:softmax参数一样，但是返回的是每个数据属于各个类别的概率。
# ☆ 
# 3.2 eval_metric[默认值取决于objective参数的取值]
# 对于有效数据的度量方法。
# 对于回归问题，默认值是rmse，对于分类问题，默认值是error。
# 典型值有：
# rmse 均方根误差
# mae 平均绝对误差
# logloss 负对数似然函数值
# error 二分类错误率(阈值为0.5)
# merror 多分类错误率
# mlogloss 多分类logloss损失函数
# auc 曲线下面积
# ☆ 
# 3.3 seed(默认0)
# 随机数的种子
# 设置它可以复现随机数据的结果，也可以用于调整参数
# ☆ 90

#1 用网格搜索调整eta
XGB = XGBClassifier(booster='gbtree',random_state=90)
param_grid = {'eta':np.arange(0.01,0.2,0.01)}
GS = GridSearchCV(XGB, param_grid, cv=10)
GS.fit(data, target)

best_param = GS.best_params_
best_score = GS.best_score_
print(best_param, best_score)
#{'eta': 0.01} 0.4399907977338863



#2 用网格搜索调整 min_child_weight
XGB = XGBClassifier(booster='gbtree',eta=0.01, random_state=90)
param_grid = {'min_child_weight':np.arange(0.5,1.5,0.1)}
GS = GridSearchCV(XGB, param_grid, cv=10)
GS.fit(data, target)

best_param = GS.best_params_
best_score = GS.best_score_
print(best_param, best_score)
# {'min_child_weight':  0.9999999999999999} 0.4399907977338863


#3 用网格搜索调整 max_depth - eta維持0.3 
XGB = XGBClassifier(booster='gbtree',eta=0.3, min_child_weight= 0.9999999999999999,random_state=90)
param_grid = {'max_depth':np.arange(2,30,1)}
GS = GridSearchCV(XGB, param_grid, cv=10)
GS.fit(data, target)

best_param = GS.best_params_
best_score = GS.best_score_
print(best_param, best_score)
# 調整前eta前 {'max_depth': 2} 0.4347172399384907
GS.predict_proba(test)  #實際上傳值 0.59148

#4 試驗看看eta = 0.01 失敗
XGB = XGBClassifier(max_depth=2,booster='gbtree',eta=0.01, min_child_weight= 0.9999999999999999,random_state=90)
XGB.fit(data, target)
XGB.predict_proba(test)
# 實際上傳值 0.55181 



#5 用网格搜索调整eta 加大範圍
XGB = XGBClassifier(max_depth=2, booster='gbtree' , min_child_weight= 0.9999999999999999,random_state=90)
param_grid = {'eta':np.arange(0.01,0.6,0.01)}
GS = GridSearchCV(XGB, param_grid, cv=10)
GS.fit(data, target)

best_param = GS.best_params_
best_score = GS.best_score_
print(best_param, best_score)

#{'eta': 0.01} 0.4798447149797739 - 0.0.1 卻失敗 eta 應使用 0.3

#6. 用网格搜索调整gamma 加大範圍 
XGB = XGBClassifier(max_depth=2, booster='gbtree' , min_child_weight= 0.9999999999999999,etc =0.3, random_state=90)
param_grid = {'gamma':np.arange(0.00,0.3,0.01)}
GS = GridSearchCV(XGB, param_grid, cv=10)
GS.fit(data, target)
 

best_param = GS.best_params_
best_score = GS.best_score_
print(best_param, best_score)

# {'gamma': 0.0} 0.4347172399384907


# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 14:10:57 2021

@author: Student
"""

from xgboost import XGBClassifier 
from sklearn.model_selection import cross_val_score
import pandas as pd
import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
# # 載入樣本資料集 - 先不轉換feature 
df = pd.read_csv('all_features_train_OneHot.csv')
test = pd.read_csv('all_features_test_OneHot.csv')
# Features_bf_filter = pd.read_csv('ROC_ANOVA_intersection_100in90.csv')
# Features_bf_filter = Features_bf_filter.iloc[:,1].tolist()

# data = df[Features_bf_filter]
# test1 = test[Features_bf_filter]
target = df['label']
train_data = df.iloc[:,4:]
X_train,X_test,y_train,y_test = train_test_split(train_data,target,test_size=0.2,random_state=12343)

# 訓練模型
model = xgb.XGBClassifier(max_depth=2, booster='gbtree' , min_child_weight= 0.9999999999999999,etc =0.3, random_state=90)
model.fit(X_train,y_train)


# 對測試集進行預測
y_pred = model.predict(X_test)
 
#計算準確率
accuracy = accuracy_score(y_test,y_pred)

print('accuracy:%2.f%%'%(accuracy*100))
# accuracy:75%
# 顯示重要特徵
plot_importance(model,max_num_features =87,title = "Feature importance Top 20")

model.feature_importances_
model.get_num_boosting_rounds()
model.get_xgb_params()
from xgboost import plot_tree 
plot_tree(model,num_trees =2)
plt.show()

#篩選出重要值大於0 使用xgboost在進行一次特徵篩選
model.feature_importances_ > 0 
train_data_importance =  train_data[train_data.columns[model.feature_importances_ > 0 ]]



# 拿篩選過的 87 features 做一次 modeling
X_train,X_test,y_train,y_test = train_test_split(train_data_importance,target,test_size=0.2,random_state=12343)

# 訓練模型
model = xgb.XGBClassifier(max_depth=2, booster='gbtree' , min_child_weight= 0.9999999999999999,etc =0.3, random_state=90)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test,y_pred)
model

print('accuracy:%2.f%%'%(accuracy*100))
# accuracy:75%
cross_val_score(model, train_data, target, cv=3).mean()
# 0.6065144130420471
test87 = test[train_data_importance.columns]
model.predict_proba(test87) 
# 上傳至Kaggle 得到 0.61069的分數 很高


# ------------------------------- 決定使用 這個feature進行條參

#6 {'gamma': 0.0} 0.4347172399384907
#7 無
#8. 用网格搜索调整subsample 加大範圍 
XGB = XGBClassifier(max_depth=2, booster='gbtree' , min_child_weight= 0.9999999999999999,etc =0.3, 
                    gamma = 0, random_state=90)
param_grid = {'subsample':np.arange(0.5,1.01,0.01)}
GS = GridSearchCV(XGB, param_grid, cv=10)
GS.fit(train_data_importance, target)
# 重作一下 10/28 5:13
best_param = GS.best_params_
best_score = GS.best_score_
print(best_param, best_score)
GS.predict_proba(test87)
# {'subsample': 0.5700000000000001} 0.44582628046816086 用原始的90_feature2跑的
# 實測上傳test 0.61069 用原始的90feature跑的

# {'subsample': 0.9500000000000004} 0.435205118136931 改成87_importance 跑的
# 測上傳test 0.60759 用xgb importance的87 feature跑的


#9 用网格搜索调整colsample_bytree加大範圍 
XGB = XGBClassifier(max_depth=2, booster='gbtree' , min_child_weight= 0.9999999999999999,etc =0.3, 
                    subsample=0.5700000000000001 ,gamma = 0, random_state=90)
param_grid = {'colsample_bytree':np.arange(0.5,1.01,0.01)}
GS = GridSearchCV(XGB, param_grid, cv=10)
# GS.fit(data, target)
GS.fit(train_data_importance, target)
best_param = GS.best_params_
best_score = GS.best_score_
print(best_param, best_score)
GS.predict_proba(test87)
#{'colsample_bytree': 0.9900000000000004} 0.44610744104356675  用原始的90_feature跑的

#{'colsample_bytree': 0.9700000000000004} 0.4338430109952129  改成87_importance 跑的

#10 用原始資料跑PCA 進行PCA 沒辦法選擇

import numpy as np
from sklearn.decomposition import PCA

param_grid = {'n_components':np.arange(80,100,1)}
PCAm = PCA()

GS = GridSearchCV(PCAm, param_grid, cv=2)
# newX = PCAm.fit_transform(X)
GS.fit(data_90, target)
best_param = GS.best_params_
best_score = GS.best_score_
print(best_param, best_score)
# 找出 {'n_components': 83} -15328.825896684333

PCA90 = PCA(n_components= 90)
PCA90.fit(data_90)
PCA90.explained_variance_ratio_  #沒法選擇

# array([7.94713829e-01, 1.13310614e-01, 3.99836489e-02, 1.68322190e-02,
#        1.34430262e-02, 1.03664789e-02, 4.29954513e-03, 3.23437968e-03,
#        1.56051317e-03, 6.29783573e-04, 4.69528834e-04, 3.56262421e-04,
#        1.72579199e-04, 1.36406016e-04, 8.46860568e-05, 6.56924049e-05,
#        5.96280481e-05, 5.30539363e-05, 4.33865968e-05, 3.65885017e-05,
#        2.96192702e-05, 2.50233802e-05, 1.47262991e-05, 1.31055303e-05,
#        1.16074667e-05, 7.64495968e-06, 7.02424623e-06, 6.35134587e-06,
#        5.73695649e-06, 4.86472962e-06, 2.79462230e-06, 2.43387414e-06,
#        2.17868460e-06, 2.09125516e-06, 1.73417541e-06, 1.33178979e-06,
#        1.24621005e-06, 1.18257209e-06, 1.00547572e-06, 6.52353206e-07,
#        6.16465460e-07, 5.89172025e-07, 5.17535649e-07, 4.60112782e-07,
#        3.86301465e-07, 3.64736717e-07, 3.52373466e-07, 2.54295495e-07,
#        2.42387153e-07, 2.16967613e-07, 1.84791731e-07, 1.77502631e-07,
#        1.67035501e-07, 1.64789076e-07, 1.40951455e-07, 1.20858937e-07,
#        1.17283472e-07, 1.16710327e-07, 1.01743444e-07, 9.27450283e-08,
#        8.76792816e-08, 7.17967690e-08, 6.08269240e-08, 4.28675190e-08,
#        3.28397851e-08, 3.05240108e-08, 2.55488659e-08, 1.76850952e-08,
#        1.73857321e-08, 1.38500686e-08, 6.18286518e-09, 1.87148412e-09,
#        6.96510093e-10, 3.89276302e-32, 1.80621216e-32, 4.62926364e-33,
#        4.62926364e-33, 4.62926364e-33, 4.62926364e-33, 4.62926364e-33,
#        4.62926364e-33, 4.62926364e-33, 4.62926364e-33, 4.62926364e-33,
#        4.62926364e-33, 4.62926364e-33, 4.62926364e-33, 4.62926364e-33,
#        4.62926364e-33, 3.23042384e-33])

# 11. 標準化方式 - 結論使用RobustScaler 資料集中存在離群點，
# 就需要利用RobustScaler針對離群點做標準化處理，該方法對資料中心化的資料的縮放更強的引數控制能力。
# 也造成要全部重新調餐的問題

from sklearn import preprocessing
# 1.Max-Min
#建立MinMaxScaler物件
minmax = preprocessing.MinMaxScaler()
# 資料標準化
data_minmax = minmax.fit_transform(train_data_importance)
#驗證
XGB = XGBClassifier(max_depth=2, booster='gbtree' , min_child_weight= 0.9999999999999999,etc =0.3, 
                    subsample=0.5700000000000001 ,gamma = 0, random_state=90)
XGB.fit(data_minmax,target)
# 對測試集進行預測並提交機率
XGB.predict_proba(test87)
# 0.57578 降低歐

# 2. Z-Score
#建立StandardScaler物件
zscore = preprocessing.StandardScaler()
# 資料標準化
data_zs = zscore.fit_transform(train_data_importance)
#驗證
XGB = XGBClassifier(random_state=90)
XGB.fit(data_zs,target)
# 對測試集進行預測並提交機率
XGB.predict_proba(test87)
# 換成原始參數後 降得更低 0.54959
# 用9步驟參數的話  0.57473

# 3.MaxAbs

#建立MinMaxScaler物件
maxabs = preprocessing.MaxAbsScaler()
# 資料標準化
data_maxabs = maxabs.fit_transform(train_data_importance)
#驗證
XGB = XGBClassifier(max_depth=2, booster='gbtree' , min_child_weight= 0.9999999999999999,etc =0.3, 
                    subsample=0.5700000000000001 ,gamma = 0, random_state=90)
XGB.fit(data_maxabs,target)
# 對測試集進行預測並提交機率
XGB.predict_proba(test87)

# 原始 0.57695
# 用9步驟參數的話0.59309


# 4.RobustScaler

#建立RobustScaler物件
robust = preprocessing.RobustScaler()
# 資料標準化
data_rob = robust.fit_transform(train_data_importance)
#驗證
XGB = XGBClassifier( random_state=90)
XGB.fit(data_rob ,target)
# 對測試集進行預測並提交機率
XGB.predict_proba(test87)

# 原始  0.59819
# 用9步驟參數的話0.59819
# 巧合都一樣

# 12 更換使用 importance的87個變數專換成RobustScaler的資料集 對過往以調整過參數
# 資料及名稱為 data_rob 
# 再次進行條餐
# 'colsample_bytree':np.arange(0.5,1.01,0.01),
#2.               'min_child_weight':np.arange(0.5,1.5,0.1),
#1.               'max_depth':np.arange(1,40,1),
#3.               'gamma':np.arange(0.00,0.3,0.01),
#4.               'subsample':np.arange(0.5,1.01,0.01),
#5.               'colsample_bytree':np.arange(0.5,1.01,0.01)
#               }

# 12-1 用网格搜索调整max_depth
XGB = XGBClassifier(booster='gbtree' ,etc =0.3, random_state=90)
param_grid = {'max_depth':np.arange(1,40,1) }
GS = GridSearchCV(XGB, param_grid, cv=10)
GS.fit(data_rob , target)
best_param = GS.best_params_
best_score = GS.best_score_

print(best_param, best_score)
GS.predict_proba(test87)

# {'max_depth': 1} 0.45940875342630616

# 12-2 用网格搜索调整min_child_weight
t1 = time.time()
XGB = XGBClassifier(max_depth=1,booster='gbtree' ,etc =0.3, random_state=90)
param_grid = {'min_child_weight':np.arange(0.5,1.5,0.1)}
GS = GridSearchCV(XGB, param_grid, cv=10)
GS.fit(data_rob , target)
best_param = GS.best_params_
best_score = GS.best_score_

print(best_param, best_score)

t2 = time.time()
print('time elapsed: ' + str(round(t2-t1, 2)) + ' seconds')
print('time elapsed: ' + str(t2-t1) + ' seconds')

# {'min_child_weight': 0.5} 0.45940875342630616
# time elapsed: 593.17 seconds
# time elapsed: 593.1707291603088 seconds

# 12-3 用网格搜索调整gamma
t1 = time.time()
XGB = XGBClassifier(min_child_weight=0.5,max_depth=1,booster='gbtree' ,etc =0.3, random_state=90)
param_grid = { 'gamma':np.arange(0.00,0.3,0.01)}
GS = GridSearchCV(XGB, param_grid, cv=10)
GS.fit(data_rob , target)
best_param = GS.best_params_
best_score = GS.best_score_

print(best_param, best_score)

t2 = time.time()
print('time elapsed: ' + str(round(t2-t1, 2)) + ' seconds')
print('time elapsed: ' + str(t2-t1) + ' seconds')
GS.predict_proba(test87)
# {'gamma': 0.0} 0.45940875342630616
# time elapsed: 1816.92 seconds
# time elapsed: 1816.9240226745605 seconds

# 12-4 用网格搜索调整subsample
t1 = time.time()
XGB = XGBClassifier(gamma=0,min_child_weight=0.5,max_depth=1,booster='gbtree' ,etc =0.3, random_state=90)
param_grid = { 'subsample':np.arange(0.5,1.01,0.01)}
GS = GridSearchCV(XGB, param_grid, cv=10)
GS.fit(data_rob , target)
best_param = GS.best_params_
best_score = GS.best_score_

print(best_param, best_score)

t2 = time.time()
print('time elapsed: ' + str(round(t2-t1, 2)) + ' seconds')
print('time elapsed: ' + str(t2-t1) + ' seconds')
GS.predict_proba(test87)
# {'subsample': 0.53} 0.4709356503807167
# time elapsed: 3781.42 seconds
# time elapsed: 3781.423168897629 seconds

# 12-5 用网格搜索调整colsample_bytree 加上 eval_metric auc
t1 = time.time()
XGB = XGBClassifier(eval_metric='auc' ,subsample=0.53,gamma=0,min_child_weight=0.5,max_depth=1,booster='gbtree' ,etc =0.3, random_state=90)
param_grid = { 'colsample_bytree':np.arange(0.5,1.01,0.01)}
GS = GridSearchCV(XGB, param_grid, cv=10)
GS.fit(data_rob , target)
best_param = GS.best_params_
best_score = GS.best_score_

print(best_param, best_score)

t2 = time.time()
print('time elapsed: ' + str(round(t2-t1, 2)) + ' seconds')
print('time elapsed: ' + str(t2-t1) + ' seconds')
GS.predict_proba(test87)

# {'colsample_bytree': 0.5} 0.47733952392032936
# time elapsed: 3714.61 seconds
# time elapsed: 3714.608857154846 seconds


# {'colsample_bytree': 0.5} 0.47733952392032936
# time elapsed: 3641.82 seconds
# time elapsed: 3641.8245639801025 seconds



# 13. 用网格搜索调整colsample_bylevel
t1 = time.time()
XGB = XGBClassifier(colsample_bytree = 0.5,eval_metric='auc' ,subsample=0.53,gamma=0,min_child_weight=0.5,max_depth=1,booster='gbtree' ,etc =0.3, random_state=90)
param_grid = { 'colsample_bylevel':np.arange(0.5,1.01,0.01)}
GS = GridSearchCV(XGB, param_grid, cv=10)
GS.fit(data_rob , target)
best_param = GS.best_params_
best_score = GS.best_score_

print(best_param, best_score)

t2 = time.time()
print('time elapsed: ' + str(round(t2-t1, 2)) + ' seconds')
print('time elapsed: ' + str(t2-t1) + ' seconds')
GS.predict_proba(test87)
# {'colsample_bylevel': 0.56} 0.4825188051356717
# time elapsed: 2869.89 seconds
# time elapsed: 2869.885647058487 seconds

# 14 用网格搜索调整lambda
t1 = time.time()
XGB = XGBClassifier(colsample_bylevel= 0.56,colsample_bytree = 0.5,eval_metric='auc' ,subsample=0.53,gamma=0,min_child_weight=0.5,max_depth=1,booster='gbtree' ,etc =0.3, random_state=90)
param_grid = { 'lambda':np.arange(0.5,1.01,0.01)}
GS = GridSearchCV(XGB, param_grid, cv=10)
GS.fit(data_rob , target)
best_param = GS.best_params_
best_score = GS.best_score_

print(best_param, best_score)

t2 = time.time()
print('time elapsed: ' + str(round(t2-t1, 2)) + ' seconds')
print('time elapsed: ' + str(t2-t1) + ' seconds')
GS.predict_proba(test87)

# {'lambda': 0.8300000000000003} 0.48252505279280034
# time elapsed: 2807.2 seconds
# time elapsed: 2807.204931974411 seconds


# 15 用网格搜索调整max_depth 增加train的大小 來降低過度擬合的機會， 改變最大深度 
t1 = time.time()
XGB = XGBClassifier(colsample_bylevel= 0.56,colsample_bytree = 0.5,eval_metric='auc' ,subsample=0.53,gamma=0,min_child_weight=0.5,max_depth=1,booster='gbtree' ,etc =0.3, random_state=90)
param_grid = { 'max_depth':np.arange(3,30,1)}
GS = GridSearchCV(XGB, param_grid, cv=15)
GS.fit(data_rob , target)
best_param = GS.best_params_
best_score = GS.best_score_

print(best_param, best_score)

t2 = time.time()
print('time elapsed: ' + str(round(t2-t1, 2)) + ' seconds')
print('time elapsed: ' + str(t2-t1) + ' seconds')
GS.predict_proba(test87)

# {'max_depth': 3} 0.48815693226313783
# time elapsed: 36254.09 seconds
# time elapsed: 36254.087679862976 seconds

# 0.57513 變低

# 15-2 更換資料集 train_data_importance
t1 = time.time()
XGB = XGBClassifier(colsample_bylevel= 0.56,co6lsample_bytree = 0.5,eval_metric='auc' ,subsample=0.53,gamma=0,min_child_weight=0.5,booster='gbtree' ,etc =0.3, random_state=90)
param_grid = { 'max_depth':np.arange(1,2,1)}
GS = GridSearchCV(XGB, param_grid, cv=15)
GS.fit(train_data_importance , target)
best_param = GS.best_params_
best_score = GS.best_score_

print(best_param, best_score)

t2 = time.time()
print('time elapsed: ' + str(round(t2-t1, 2)) + ' seconds')
print('time elapsed: ' + str(t2-t1) + ' seconds')
GS.predict_proba(test87)

# {'max_depth': 1} 0.5299367785539093
# time elapsed: 104.6 seconds
# time elapsed: 104.59893846511841 seconds
# 0.60306

# 15-3 更換資料集 90
t1 = time.time()
XGB = XGBClassifier(colsample_bylevel= 0.56,co6lsample_bytree = 0.5,eval_metric='auc' ,subsample=0.53,gamma=0,min_child_weight=0.5,booster='gbtree' ,etc =0.3, random_state=90)
param_grid = { 'max_depth':np.arange(1,2,1)}
GS = GridSearchCV(XGB, param_grid, cv=15)
GS.fit(data_90 , target)
best_param = GS.best_params_
best_score = GS.best_score_

print(best_param, best_score)

t2 = time.time()
print('time elapsed: ' + str(round(t2-t1, 2)) + ' seconds')
print('time elapsed: ' + str(t2-t1) + ' seconds')
GS.predict_proba(test_90)

# {'max_depth': 1} 0.5298619348963522
# time elapsed: 102.49 seconds
# time elapsed: 102.4905776977539 seconds
# 0.60736

#15-4 原始資料跑跑
data_orginal = df.iloc[:,3:]
test_orginal = test.iloc[:,3:]

t1 = time.time()
XGB = XGBClassifier(colsample_bylevel= 0.56,co6lsample_bytree = 0.5,eval_metric='auc' ,subsample=0.53,gamma=0,min_child_weight=0.5,booster='gbtree' ,etc =0.3, random_state=90)
param_grid = { 'max_depth':np.arange(1,2,1)}
GS = GridSearchCV(XGB, param_grid, cv=15)
GS.fit(data_orginal , target)
best_param = GS.best_params_
best_score = GS.best_score_

print(best_param, best_score)

t2 = time.time()
print('time elapsed: ' + str(round(t2-t1, 2)) + ' seconds')
print('time elapsed: ' + str(t2-t1) + ' seconds')

GS.predict_proba(test_orginal)

# {'max_depth': 1} 0.510549237099701
# time elapsed: 266.1 seconds
# time elapsed: 266.1007583141327 seconds
# 0.60844

#16 PCA 降為 嘗試在使用RobustScaler物件
from sklearn.preprocessing import RobustScaler
robust = RobustScaler()
# 1. 資料標準化
test.columns
train_robust = robust.fit_transform(df.iloc[:,3:])
test_rebust= robust.transform(test.iloc[:,3:])

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train_std = sc.fit_transform(df.iloc[:,3:])
X_test_std = sc.transform(test.iloc[:,3:])
print('訓練集資料標準化 \n%s' % X_train_std )
print('測試集標準化 \n%s' % X_test_std )


# 2. 取得特爭執
from sklearn.decomposition import PCA

pca = PCA()
X_train_pca = pca.fit_transform(X_train_std)
pca.explained_variance_ratio_


# 3. 列出並排序全部的特徵值
import matplotlib.pyplot as plt
import numpy as np
plt.bar(range(1,100), pca.explained_variance_ratio_[1:100], alpha=0.5, align='center')
plt.step(range(1,100), np.cumsum(pca.explained_variance_ratio_[1:100]), where='mid')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')

plt.show()











# 附件1 計算時間的工具
import time

def my_sort():
    time.sleep(1.1234) # 用 sleep 模擬 my_sort() 運算時間

t1 = time.time()
my_sort()
t2 = time.time()
print('time elapsed: ' + str(round(t2-t1, 2)) + ' seconds')
print('time elapsed: ' + str(t2-t1) + ' seconds')




# 附件2 .csv產生器 使Console不能打印的檔案可方便閱讀 
# Generate CSV file      

# filename =  GS.predict_proba(test87)  # 想要印出的程式碼 
filename =  GS.predict_proba(test_orginal)
Result ='15-4.csv' # 印出CSV.名稱  
def OutputCSV(filename,Result):   
      
    df_SAMPLE = pd.DataFrame.from_dict(filename)
    df_SAMPLE.to_csv( Result  , index= True )
    print( '成功產出'+Result )

OutputCSV()

# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 21:54:48 2021

@author: Kevin Dai
"""
df = pd.read_csv('all_features_train_OneHot.csv')

test = pd.read_csv('all_features_test_OneHot.csv')

X_train, X_test, y_train, y_test =  train_test_split(df, df['label'],
                     test_size=0.20,random_state = 1)

Features_bf_filter = pd.read_csv('ROC_ANOVA_intersection_100in90.csv')
type(Features_bf_filter.iloc[:,1].tolist())

Features_bf_filter = Features_bf_filter.iloc[:,1].tolist()
Features_bf_filter

X_train = X_train[Features_bf_filter]
test_90 = test[Features_bf_filter]

# RFC0 
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(random_state=0)
clf.fit(data_90,  target)
clf.predict(test_90)
clf.predict_proba(test_90)
roc_auc_score(target, clf.predict_proba(data_90)[:, 1])
# 0.982963765211813
metrics.plot_roc_curve(clf, data_90,  target , response_method="predict_proba")
plt.show()
OutputCSV(clf.predict_proba(test_90),'RFC0.csv')
# 0.59136


#RFC1 sklearn.ensemble.RandomForestClassifier API  https://scikit-learn.org/0.24/modules/generated/sklearn.ensemble.RandomForestClassifier.html?highlight=randomforest#sklearn.ensemble.RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(data_90,  target)
clf.predict(test_90)
clf.predict_proba(test_90)

roc_auc_score(target, clf.predict_proba(data_90)[:, 1])
metrics.plot_roc_curve(XGB, data_90,  target , response_method="predict_proba")
# 0.6791005553135705
plt.show()

#RCF2 - n_estimators
from sklearn.model_selection import GridSearchCV
param_grid = {'n_estimators':range(50,150,50)}
model = RandomForestClassifier(max_depth=2, random_state=0)
clf = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc')
clf.fit(data_90,  target)
best_param = clf.best_params_
best_score = clf.best_score_
print(best_param, best_score)
clf.predict_proba(test_90) 

clf = RandomForestClassifier(n_estimators= 50 ,max_depth=2, random_state=0)
clf.fit(data_90,  target)
roc_auc_score(target, clf.predict_proba(data_90)[:, 1])
# 0.6796734139948082
# {'n_estimators': 50} 0.5046264832921984

#RCF3  criterion{“gini”, “entropy”}
from sklearn.model_selection import GridSearchCV
param_grid = {'criterion':['gini', 'entropy']}
model = RandomForestClassifier(n_estimators=50,max_depth=2, random_state=0)
clf = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc')
clf.fit(data_90,  target)
best_param = clf.best_params_
best_score = clf.best_score_
print(best_param, best_score)
clf.predict_proba(test_90) 
# {'criterion': 'gini'} 0.5046264832921984
# 0.6796734139948082


#RCF4 max_depth
from sklearn.model_selection import GridSearchCV
param_grid = {'max_depth':[1,3,5,10,20,30]}
model = RandomForestClassifier(criterion='gini', n_estimators=50,max_depth=2, random_state=0)
clf = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc')
clf.fit(data_90,  target)
best_param = clf.best_params_
best_score = clf.best_score_
print(best_param, best_score)
clf.predict_proba(test_90)
# {'max_depth': 1} 0.534513155650366
clf = RandomForestClassifier(max_depth=1, n_estimators= 50, random_state=0)
clf.fit(data_90,  target)
roc_auc_score(target, clf.predict_proba(data_90)[:, 1])
#  0.6602901491610633

#RCF5 min_samples_leaf
from sklearn.model_selection import GridSearchCV
param_grid = {'min_samples_leaf':range(1,10,1)}
model = RandomForestClassifier(max_depth=1,criterion='gini', n_estimators=50,max_depth=2, random_state=0)
clf = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc')
clf.fit(data_90,  target)
best_param = clf.best_params_
best_score = clf.best_score_
print(best_param, best_score)
clf.predict_proba(test_90)

clf = RandomForestClassifier(max_depth=1, n_estimators= 50, random_state=0, min_samples_leaf = 1)
clf.fit(data_90,  target)
roc_auc_score(target, clf.predict_proba(data_90)[:, 1])
#  0.6602901491610633

from sklearn.metrics import roc_auc_score
roc_auc_score(target, clf.predict_proba(data_90)[:, 1])

# {'min_samples_leaf': 1} 0.5046264832921984


#RCF6 min_samples_leaf
param_grid = {'n_estimators':range(40,60,2)}
model = RandomForestClassifier(random_state=0)
clf = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc')
clf.fit(data_90,  target)
best_param = clf.best_params_
best_score = clf.best_score_
print(best_param, best_score)
roc_auc_score(target, clf.predict_proba(data_90)[:, 1])
# 0.9821631992279224
metrics.plot_roc_curve(clf, data_90,  target , response_method="predict_proba")

plt.show()
OutputCSV(clf.predict_proba(test_90),'RFC6.csv')

filename =  clf.predict_proba(test_90) 
Result ='RCF5_min_samples_leaf.csv' # 印出CSV.名稱  
def OutputCSV():   
      
    df_SAMPLE = pd.DataFrame.from_dict(filename)
    df_SAMPLE.to_csv( Result  , index= True )
    print( '成功產出'+Result )

OutputCSV()



