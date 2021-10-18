# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 13:52:08 2021

@author: Student
"""
import numpy as np
import pandas as pd
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
X_test = X_test[Features_bf_filter]

# direction='backward'

# 2.1.2.1 sklearn.RandomForestClassifier() API https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
# 1. 實體化向前包裝法 評分為'roc_auc'
sfs = SequentialFeatureSelector(RandomForestClassifier(), direction='forward',scoring='roc_auc',)
# 2. 將資料fit的到向前包裝法
sfs.fit(X_train, y_train)
# 3. 將不需要的X特徵剔除
dfsfs_RF_x_train = sfs.transform(X_train)
dfsfs_RF_x_test = sfs.transform(X_test)


# 2.1.2.2 SVM (SVC)https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html?highlight=svm#sklearn.svm.SVC 
sfs = SequentialFeatureSelector(SVC(gamma='auto'), direction='forward',scoring='roc_auc')
sfs.fit(X_train, y_train)
dfsfs_SVC_x_train = sfs.transform(X_train)
dfsfs_SVC_x_test = sfs.transform(X_test)

# 2.1.2.3 xgboost https://xgboost.readthedocs.io/en/latest/python/python_api.html#module-xgboost.sklearn
sfs = SequentialFeatureSelector(XGBClassifier(), direction='forward',scoring='roc_auc')
sfs.fit(X_train, y_train)
dfsfs_xgboost_x_train = sfs.transform(X_train)
dfsfs_xgboost_x_test = sfs.transform(X_test)






# 0.0 .csv產生器 使Console不能打印的檔案可方便閱讀 
# Generate CSV file      

filename = dfsfs_RF_x_train  # 想要印出的程式碼 

Result ='dfsfs_RF_x_train_forward.csv' # 印出CSV.名稱  
def OutputCSV():   
      
    df_SAMPLE = pd.DataFrame.from_dict(filename)
    df_SAMPLE.to_csv( Result  , index= True )
    print( '成功產出'+Result )

OutputCSV()
