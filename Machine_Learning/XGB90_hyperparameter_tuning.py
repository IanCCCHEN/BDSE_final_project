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

#Xgboost 的三種參數
# 通用参数：宏观函数控制。
# Booster参数：控制每一步的booster(tree/regression)。
# 学习目标参数：控制训练目标的表现。

# 1. 通用參數調整

# 1.1 booster[默認gbtree] - gbtree 基於樹的模型、gbliner 線性模型
# ☆ 選擇 gbtree 
# 1.2 silent[默认0]
# 当这个参数值为1时，静默模式开启，不会输出任何信息。
# 一般这个参数就保持默认的0，因为这样能帮我们更好地理解模型。
# ☆ 選擇 0 
# 1.3 nthread[默认值为最大可能的线程数]
# 这个参数用来进行多线程控制，应当输入系统的核数。
# 如果你希望使用CPU全部的核，那就不要输入这个参数，算法会自动检测它。
# ☆ 不用設定任由默認達到最快

# 2. booster参数調整

# 2.1 eta[默认0.3] 與GBM Learning_rate相似
#和GBM中的 learning rate 参数类似。
# 通过减少每一步的权重，可以提高模型的鲁棒性(Robustness) as 穩健程度。
# 典型值为0.01-0.2。
# ☆ 看來可以在典型值範圍內找0.01為區間去搜尋
# 2.2 min_child_weight[默认1]
# 决定最小叶子节点样本权重和。
# 和GBM的 min_child_leaf 参数类似，但不完全一样。XGBoost的这个参数是最小样本权重的和，而GBM参数是最小样本总数。
# 这个参数用于避免过拟合。当它的值较大时，可以避免模型学习到局部的特殊样本。
# 但是如果这个值过高，会导致欠拟合。这个参数需要使用CV来调整。
# ☆ 不能太大不能太小 還是要再找參考值
# 2.3 max_depth[默认6]
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
# 2.5 gamma[默认0]
# 在点分裂时，只有分裂后损失函数的值下降了，才会分裂这个节点。Gamma指定了节点分裂所需的最小损失函数下降值。
# 这个参数的值越大，算法越保守。这个参数的值和损失函数息息相关，所以是需要调整的。
# ☆ 沒有具體講要怎麼調，還要再查資料
# 2.6 max_delta_step[默认0]
# 这参数限制每棵树权重改变的最大步长。如果这个参数的值为0，那就意味着没有约束。
# 如果它被赋予了某个正值，那么它会让这个算法更加保守。
# 通常，这个参数不需要设置。但是当各类别的样本十分不平衡时，它对逻辑回归是很有帮助的。
# 这个参数一般用不到，但是你可以挖掘出来它更多的用处。
# ☆ 不限制步長，所以不調整








XGB = XGBClassifier(booster='gbtree',random_state=90)

# 用网格搜索调整max_depth
param_grid = {'eta':np.arange(0.01,0.2,0.01)}
GS = GridSearchCV(XGB, param_grid, cv=10)
GS.fit(data, target)

best_param = GS.best_params_
best_score = GS.best_score_
print(best_param, best_score)
print(data.shape)




