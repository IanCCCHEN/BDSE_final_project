# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 17:06:01 2021

@author: Student
"""

#依照參數影響程度進行條餐
# 模型條餐https://zhuanlan.zhihu.com/p/126288078
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 导入数据集 轉化成90個特爭
df = pd.read_csv('all_features_train_OneHot.csv')
Features_bf_filter = pd.read_csv('ROC_ANOVA_intersection_100in90.csv')
Features_bf_filter = Features_bf_filter.iloc[:,1].tolist()
Features_bf_filter
data = df[Features_bf_filter]

#再轉華城45個特徵
dflist = pd.read_csv('dfsfs_RF_forward_boolean.csv')
dflist = dflist['0']
data.columns[dflist]
data = data[data.columns[dflist]]
target = df['label']

# 建立随机森林
rfc = RandomForestClassifier(n_estimators=100, random_state=90)

# 用交叉验证计算得分
score_pre = cross_val_score(rfc, data, target, cv=10).mean()
score_pre

# 调参，绘制学习曲线来调参n_estimators（对随机森林影响最大）
score_lt = []

# 每隔10步建立一个随机森林，获得不同n_estimators的得分

for i in range(0,60,10):
    rfc = RandomForestClassifier(n_estimators=i+1
                                ,random_state=90)
    score = cross_val_score(rfc, data, target, cv=10).mean()
    score_lt.append(score)
score_max = max(score_lt)
print('最大得分：{}'.format(score_max),
      '子树数量为：{}'.format(score_lt.index(score_max)*10+1))

# 绘制学习曲线
x = np.arange(1,61,10)
plt.subplot(111)
plt.plot(x, score_lt, 'r-')
plt.show()

# 在11附近缩小n_estimators的范围为1-20
score_lt = []
for i in range(1,20):
    rfc = RandomForestClassifier(n_estimators=i
                                ,random_state=90)
    score = cross_val_score(rfc, data, target, cv=10).mean()
    score_lt.append(score)
score_max = max(score_lt)
print('最大得分：{}'.format(score_max),
      '子树数量为：{}'.format(score_lt.index(score_max)))
# 绘制学习曲线
x = np.arange(1,20)
plt.subplot(111)
plt.plot(x, score_lt,'o-')
plt.show()
# 13

# 建立n_estimators为13的随机森林
rfc = RandomForestClassifier(n_estimators=13, random_state=90)

# 用网格搜索调整max_depth
param_grid = {'max_depth':np.arange(1,21)}
GS = GridSearchCV(rfc, param_grid, cv=10)
GS.fit(data, target)

best_param = GS.best_params_
best_score = GS.best_score_
print(best_param, best_score)

print(data.shape)
# 1 

# 用网格搜索调整max_features
param_grid = {'max_features':np.arange(1,46)}

rfc = RandomForestClassifier(n_estimators=13
                            ,random_state=90
                            ,max_depth=1)
GS = GridSearchCV(rfc, param_grid, cv=10)
GS.fit(data, target)
best_param = GS.best_params_
best_score = GS.best_score_
print(best_param, best_score) 

# max_features = 1



# min_samples_leaf =1 
param_grid = {'min_samples_leaf':np.arange(1,40)}

rfc = RandomForestClassifier(n_estimators=13
                            ,random_state=90
                            ,max_depth=1
                            ,max_features=1)
GS = GridSearchCV(rfc, param_grid, cv=10)
GS.fit(data, target)
best_param = GS.best_params_
best_score = GS.best_score_
print(best_param, best_score) 

# min_samples_split =2
param_grid = {'min_samples_split':np.arange(1,40)}

rfc = RandomForestClassifier(n_estimators=13
                            ,random_state=90
                            ,max_depth=1
                            ,max_features=1
                            ,min_samples_leaf=1)
GS = GridSearchCV(rfc, param_grid, cv=10)
GS.fit(data, target)
best_param = GS.best_params_
best_score = GS.best_score_
print(best_param, best_score)
