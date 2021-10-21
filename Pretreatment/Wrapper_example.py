# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 10:22:44 2021

@author: Student
"""

import numpy as np 
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import train_test_split

df = pd.read_csv('all_features_train_OneHot.csv')

X_train, X_test, y_train, y_test =  train_test_split(df, df['label'],
                     test_size=0.20,random_state = 1)

Features_bf_filter = pd.read_csv('ROC_ANOVA_intersection_100in90.csv')
type(Features_bf_filter.iloc[:,1].tolist())

Features_bf_filter = Features_bf_filter.iloc[:,1].tolist()
Features_bf_filter

X_train = X_train[Features_bf_filter]
X_test = X_test[Features_bf_filter]

# 2.1.2.1 sklearn.RandomForestClassifier() API https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
# 1. 實體化向前包裝法 評分為'roc_auc'
sfs = SequentialFeatureSelector(RandomForestClassifier(), direction='backward',scoring='roc_auc',)
# 2. 將資料fit的到向前包裝法
sfs.fit(X_train, y_train)
# 3. 將不需要的X特徵剔除
dfsfs_RF_x_backward  = sfs.get_support()


# Generate CSV file      

filename = dfsfs_RF_x_backward   # 想要印出的程式碼 

Result ='dfsfs_RF_x_backward.csv' # 印出CSV.名稱  
def OutputCSV():   
      
    df_SAMPLE = pd.DataFrame.from_dict(filename)
    df_SAMPLE.to_csv( Result  , index= True )
    print( '成功產出'+Result )

OutputCSV()
