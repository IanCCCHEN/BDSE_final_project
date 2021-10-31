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
data = df[Features_bf_filter]
target = df['label']
test = pd.read_csv('all_features_test_OneHot.csv')
test = test[Features_bf_filter]
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
plot_importance(model)

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

# import numpy as np
# from sklearn.decomposition import PCA

# param_grid = {'n_components':np.arange(80,100,1)}
# PCAm = PCA()

# GS = GridSearchCV(PCAm, param_grid, cv=2)
# # newX = PCAm.fit_transform(X)
# GS.fit(train_data, target)
# best_param = GS.best_params_
# best_score = GS.best_score_
# print(best_param, best_score)
# # 找出 {'n_components': 83} -15328.825896684333
# PCA90 = PCA(n_components= 10)
# PCA90.fit(train_data_importance)
# PCA90.explained_variance_ratio_  #沒法選擇

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

# 12-5 用网格搜索调整colsample_bytree eval_metric auc
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


# 14 用网格搜索调整max_depth 
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
