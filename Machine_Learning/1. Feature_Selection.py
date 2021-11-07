#Created on Mon Oct 11 16:18:45 2021

#@author: Kevin Dai


import numpy as np
import pandas as pd

# df = pd.read_csv('all_features_new_train_1011.csv')
# df.shape

# df = pd.read_csv('all_features_new_train_1012_One_Hot_by_Kevin_2.csv')

#1.1 特徵轉換 進行one-hot-encoding https://ithelp.ithome.com.tw/articles/10233484
#pandas get_dummies https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.get_dummies.html
df.shape  # (160057, 89)

dfnew = pd.get_dummies(df, columns=['offer_id', 'market', 'chain', 'productid'])

dfnew.shape # (160057, 287))

dfnew.columns # 無法顯示 必須打印出來

# 1.1.2 特徵轉換 對test 和 train 進行比對確認 兩個datasets 的features 一致
# 產生下面兩個檔案
# all_features_test_OneHot.csv
# all_features_train_OneHot.csv
df = pd.read_csv('all_features_train_OneHot.csv')
test = pd.read_csv('all_features_test_OneHot.csv')

# 檢查column名是否一樣
df.sort_index().columns.size
df.sort_index().columns.join(test.sort_index().columns)
test.sort_index().columns.join(df.sort_index().columns)
#確認無誤

# 2 特徵篩選  

# 2.1 相關係數矩陣
# Pearson, Spearman, Kendall 三大相关系数简单介绍https://zhuanlan.zhihu.com/p/60059869
#連續型資料適用 Pearson, Spearman 非連續型資料適用 Kendall
import pandas as pd
# 只考慮train dataset 並且移除掉不必要的feature 
df = pd.read_csv('all_features_train_OneHot.csv') 
df.head()
df =  df.iloc[:,1:]
df.columns
df.pop('repeattrips')
df.pop('id')
df.columns
# 2.1.1 將連續型資料與非連續型資料
# 取出 連續型資料與label 
dfconti = df.iloc[:,:83]

# 製作一個將不需要的feature刪除的function
def popitout(data, feature_name):
    for name in feature_name:
        data.pop(name)
    print("The process of poping was done.")
# 創建一個非連續變數的list  
pop_list_disconti =['never_bought_company','never_bought_category','never_bought_brand','has_bought_brand_company_category'
             ,'has_bought_brand_category','has_bought_brand_company','bought_product_before','established_product',
             'only_bought_our_product','returned_product']
len(pop_list_disconti)
# 執行這個popitout function 就可以得到全部都是連敘型變數的資料
popitout(dfconti, pop_list_disconti)
dfconti.sort_index().columns.size
# 取出 非連續型資料與label 
pop_list_conti = list(dfconti.sort_index().columns)
type(pop_list_conti)
pop_list_conti.remove('label') 
len(pop_list_conti)
dfdisconti = df.copy()
popitout(dfdisconti, pop_list_conti)
dfdisconti.sort_index().columns.size

# 2.1.2 創建三種相關矩陣
# Compute pairwise correlation of columns, excluding NA/null values.
# 原文網址：https://kknews.cc/code/9vyg2el.html
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html?highlight=corr#pandas.DataFrame.corr

# Pearson
# 使用資料dfconti 連續型
dfconti.corr(method='pearson', min_periods=1)
# Spearman
# 使用資料dfconti 連續型
dfconti.corr(method='spearman', min_periods=1)

# Kendall
# 使用資料dfdisconti 非連續型
dfdisconti.corr(method='kendall', min_periods=1)

#由於檔案過大 可以使用CSV列印程式OutputCSV()

#由於各個相關係數矩陣呈現不高的相關性 所以決定對資料進行 下面的另一種特徵篩選

# 2.2 Fillter法進行篩選 https://ithelp.ithome.com.tw/articles/10245037

# 2.2.1 ANOVA Univariate Test - ANOVA 參考網站http://www1.pu.edu.tw/~tfchen/design_fs/C3_ANOVA_post.pdf
#單變項檢定(univariate test)，衡量兩個變數的相依性。這個方法適用於連續型變數和二元標的(binary targets)。
from sklearn.feature_selection import f_classif # API https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html#sklearn.feature_selection.f_classif
from sklearn.feature_selection import SelectKBest

df = pd.read_csv('all_features_train_OneHot.csv') 
df.columns
X_train =  df.iloc[:,4:]
df.columns

y_train = df['label']
# 選擇要保留的特徵數
select_k = 200

selection = SelectKBest(f_classif, k= select_k).fit(X_train, y_train)
print(f_classif(X_train,y_train))
# 顯示保留的欄位
features = X_train.columns[selection.get_support()]
F_score = f_classif(X_train,y_train)
f_statistic = pd.DataFrame(F_score[0])[selection.get_support()]
p_values = pd.DataFrame(F_score[1])[selection.get_support()]
f_statistic.iloc[:,0]

# # .to_csv('F_score.csv', encoding='UTF8')
# for a,b,c in [features],[f_statistic.iloc[:,0]],[p_values.iloc[:,0]]:
#     print(a ,' f_statistic: ',b,' p_values: ',c )

# 2.2.2 Univariate ROC-AUC /RMSE
#使用機器學習模型來衡量兩個變數的相依性，適用於各種變數，且沒對變數的分布做任何假設。
#回歸性問題使用RMSE，分類性問題使用ROC-AUC。
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
# ROC的分數
roc_values = []
# 將train 分成 8:2做驗證
X_train, X_test, y_train, y_test =  train_test_split(df, df['label'],
                     test_size=0.80,random_state = 1)


# 計算分數
for feature in X_train.columns:
    clf = DecisionTreeClassifier()
    clf.fit(X_train[feature].to_frame(), y_train)
    y_scored = clf.predict_proba(X_test[feature].to_frame())
    roc_values.append(roc_auc_score(y_test, y_scored[:, 1]))

# 建立Pandas Series 用於繪圖
roc_values = pd.Series(roc_values)
roc_values.index = X_train.columns

# 顯示結果
roc_values.to_csv('SCORE.csv', encoding='UTF8')
print(roc_values.sort_values(ascending=False))



# 使用 XGboots、SVM、Randomforest(向前、向後)
# 2.3 向前特徵選取法(Forward Feature Selection)： https://ithelp.ithome.com.tw/articles/10246251 
# SequentialFeatureSelector API http://rasbt.github.io/mlxtend/api_subpackages/mlxtend.feature_selection/#sequentialfeatureselector
# 又稱為 step forward feature selection 或循序向前選取法(sequential forward feature selection— SFS)
#，這個方法剛開始時，特徵子集合是空集合，然後依序一次加入一個特徵。

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier #scikit learn API https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn 
from sklearn.svm import SVC
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
# Pandas loc iloc https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.loc.html
# SequentialFeatureSelector example http://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/#example-1-a-simple-sequential-forward-selection-example
# sklearn.feature_selection.SequentialFeatureSelector API https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SequentialFeatureSelector.html
df = pd.read_csv('all_features_train_OneHot.csv')

test = pd.read_csv('all_features_test_OneHot.csv')

X_train, X_test, y_train, y_test =  train_test_split(df, df['label'],
                     test_size=0.80,random_state = 1)

Features_bf_filter = pd.read_csv('ROC_ANOVA_intersection_100in90.csv')
type(Features_bf_filter.iloc[:,1].tolist())

Features_bf_filter = Features_bf_filter.iloc[:,1].tolist()
Features_bf_filter

X_train = X_train[Features_bf_filter]

# !!!要考慮 scoring method ROC https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
# sklearn.feature_selection

# 2.3.1 sklearn.RandomForestClassifier() API https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
# 1. 實體化向前包裝法 評分為'roc_auc'
sfs = SequentialFeatureSelector(RandomForestClassifier(), direction='forward',scoring='roc_auc',)
# 2. 將資料fit的到向前包裝法
sfs.fit(X_train, X_test)
# 3. 將不需要的X特徵剔除
dfsfs_RF_x = sfs.transform(X_train)

# 2.3.2 SVM (SVC)https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html?highlight=svm#sklearn.svm.SVC 
sfs = SequentialFeatureSelector(SVC(gamma='auto'), direction='forward',scoring='roc_auc')
sfs.fit(X_train, X_test)
dfsfs_SVC_x = sfs.transform(X_train)


# 2.3.3 xgboost https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn
sfs = SequentialFeatureSelector(XGBClassifier(), direction='forward',scoring='roc_auc')
sfs.fit(X_train, X_test)
dfsfs_xgboost_x = sfs.transform(X_train)


# 2.1.4 sklearn.linear_model.LogisticRegression API https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logisticregression#sklearn.linear_model.LogisticRegression
sfs = SequentialFeatureSelector(LogisticRegression(), direction='forward',scoring='roc_auc')
sfs.fit(X_train, X_test)
dfsfs_LR_x = sfs.transform(X_train)


# display(pd.DataFrame(sfs.get_metric_dict()))
#2.4 向後特徵淘汰法(Backward Feature Elimination)：又稱為step backward feature selection 
#或循序向後選擇法(sequential backward feature selection — SBS)，這個方法剛開始時特徵子集合包刮資料集的所有特徵
#，然後依序一次淘汰一個特徵。

# 2.4.1 sklearn.RandomForestClassifier() API https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
# 1. 實體化向前包裝法 評分為'roc_auc'
sfs = SequentialFeatureSelector(RandomForestClassifier(), direction='backward',scoring='roc_auc',)
# 2. 將資料fit的到向前包裝法
sfs.fit(X_train, X_test)
# 3. 將不需要的X特徵剔除
dfsfs_RF_x = sfs.transform(X_train)
feature_list = sfs.get_support()

# 2.4.2 SVM (SVC)https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html?highlight=svm#sklearn.svm.SVC 
sfs = SequentialFeatureSelector(SVC(gamma='auto'), direction='backward',scoring='roc_auc')
sfs.fit(X_train, X_test)
dfsfs_SVC_x = sfs.transform(X_train)
feature_list = sfs.get_support()

# 2.4.3 xgboost https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn
sfs = SequentialFeatureSelector(XGBClassifier(), direction='backward',scoring='roc_auc')
sfs.fit(X_train, X_test)
dfsfs_xgboost_x = sfs.transform(X_train)
feature_list = sfs.get_support()

# 2.4.4 sklearn.linear_model.LogisticRegression API https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logisticregression#sklearn.linear_model.LogisticRegression
sfs = SequentialFeatureSelector(LogisticRegression(), direction='backward',scoring='roc_auc')
sfs.fit(X_train, X_test)
dfsfs_LR_x = sfs.transform(X_train)
feature_list = sfs.get_support()
