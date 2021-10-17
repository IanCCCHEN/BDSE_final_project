#Created on Mon Oct 11 16:18:45 2021

#@author: Kevin Dai


import numpy as np
import pandas as pd

df = pd.read_csv('all_features_new_train_1011.csv')
df.shape

# 0.0 .csv產生器 使Console不能打印的檔案可方便閱讀 
# Generate CSV file      

# filename = result   # 想要印出的程式碼 

# Result ='result.csv' # 印出CSV.名稱  
# def OutputCSV():   
      
#     df_SAMPLE = pd.DataFrame.from_dict(filename)
#     df_SAMPLE.to_csv( Result  , index= True )
#     print( '成功產出'+Result )

# OutputCSV()

# df = pd.read_csv('all_features_new_train_1012_One_Hot_by_Kevin_2.csv')

#1.1 特徵轉換 進行one-hot-encoding https://ithelp.ithome.com.tw/articles/10233484
#pandas get_dummies https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.get_dummies.html
df.shape  # (160057, 89)

dfnew = pd.get_dummies(df, columns=['offer_id', 'market', 'chain', 'productid'])

dfnew.shape # (160057, 287))

dfnew.columns # 無法顯示 必須打印出來


# 2.1 特徵篩選 https://ithelp.ithome.com.tw/articles/10246251  

# 2.1.1 向前特徵選取法(Forward Feature Selection)：
# SequentialFeatureSelector API http://rasbt.github.io/mlxtend/api_subpackages/mlxtend.feature_selection/#sequentialfeatureselector
# 又稱為 step forward feature selection 或循序向前選取法(sequential forward feature selection— SFS)
#，這個方法剛開始時，特徵子集合是空集合，然後依序一次加入一個特徵。

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
# Pandas loc iloc https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.loc.html
# SequentialFeatureSelector example http://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/#example-1-a-simple-sequential-forward-selection-example
# sklearn.feature_selection.SequentialFeatureSelector API https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.SequentialFeatureSelector.html
df_y = dfnew.loc[:,"label"]
df_x = dfnew.iloc[:,3:]

# 2.1.2 對資料進行標準化 sklearn.preprocessing.MinMaxScaler

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
print(scaler.fit(df_x))
df_x = scaler.transform(df_x)

# !!!要考慮 scoring method ROC https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
# sklearn.feature_selection

# sklearn.RandomForestClassifier() API https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
# 1. 實體化向前包裝法 評分為'roc_auc'
sfs = SequentialFeatureSelector(RandomForestClassifier(), direction='forward',scoring='roc_auc',)
# 2. 將資料fit的到向前包裝法
sfs.fit(df_x, df_y)
# 3. 將不需要的X特徵剔除
dfsfs_RF_x = sfs.transform(df_x)

# sklearn.linear_model.LogisticRegression API https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logisticregression#sklearn.linear_model.LogisticRegression
sfs = SequentialFeatureSelector(LogisticRegression(), direction='forward',scoring='roc_auc')
sfs.fit(df_x, df_y)
dfsfs_LR_x = sfs.transform(df_x)


# display(pd.DataFrame(sfs.get_metric_dict()))
#2.1.2 向後特徵淘汰法(Backward Feature Elimination)：又稱為step backward feature selection 
#或循序向後選擇法(sequential backward feature selection — SBS)，這個方法剛開始時特徵子集合包刮資料集的所有特徵
#，然後依序一次淘汰一個特徵。

# sklearn.RandomForestClassifier() API https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
# 1. 實體化向前包裝法 評分為'roc_auc'
sfs = SequentialFeatureSelector(RandomForestClassifier(), direction='backward',scoring='roc_auc')
# 2. 將資料fit的到向前包裝法
sfs.fit(df_x, df_y)
# 3. 將不需要的X特徵剔除
dfsfs_RF_x = sfs.transform(df_x)

# sklearn.linear_model.LogisticRegression API https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html?highlight=logisticregression#sklearn.linear_model.LogisticRegression
sfs = SequentialFeatureSelector(LogisticRegression(), direction='backward',scoring='roc_auc')
sfs.fit(df_x, df_y)
dfsfs_LR_x = sfs.transform(df_x)



#2.1.3 竭盡式特徵選取法(Exhaustive Feature Selection)：這個方法測試所有可能的特徵組合。
# from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS
# 無法讀取 mlxtend 讚時擱置
#2.1.4 雙向搜尋(Bidirectional Search)：為了得到獨一的解決方案，這個方法同時同時進行向前和向後特徵選取。
# 無法讀取 mlxtend 讚時擱置

# 2.2 相關矩陣 correlation matrix https://kknews.cc/zh-tw/code/9vyg2el.html





# 3.1 訓練模型

# 3.1.1 (GBM) Gradient Tree Boosting https://scikit-learn.org/stable/modules/ensemble.html?highlight=gbm
# API https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier

from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
clf.fit(df_x, df_y)
clf.score(df_x, df_y)
print(clf.predict_proba(df_x))


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.5, random_state=100)

clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
clf.fit(X_train,  y_train)
clf.predict(X_test)

clf.score(X_test, y_test)

clf.predict_proba(X_train)

#3.2.2 sklearn.model_selection.GridSearchCV 網格搜尋 
# RandomizedSearchCV的使用方法其實是和GridSearchCV一致的 https://tw511.com/a/01/8581.html
# https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
from sklearn.model_selection import GridSearchCV

param_grid = {'loss':('deviance', 'exponential'),'n_estimators':range(100,300), 
              'criterion':('friedman_mse', 'squared_error', 'mse', 'mae'),'max_depth':range(3,20)}
model = GradientBoostingClassifier()
clf = GridSearchCV(model, param_grid, cv=5, scoring='roc_auc')
clf.fit(df_x, df_y)
clf.score(X_test, y_test)
GBM_para = clf.get_params(deep=True)
