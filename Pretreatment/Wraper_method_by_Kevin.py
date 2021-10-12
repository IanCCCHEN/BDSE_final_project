# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 16:18:45 2021

@author: Kevin Dai
"""

import numpy as np
import pandas as pd

df = pd.read_csv('all_features_new_train_1011.csv')

#1.1 特徵轉換 進行one-hot-encoding https://ithelp.ithome.com.tw/articles/10233484
#pandas get_dummies https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.get_dummies.html
df.shape  # (160057, 89)

dfnew = pd.get_dummies(df, columns=['offer_id', 'market', 'chain'])

dfnew.shape # (160057, 274))

dfnew.columns # 無法顯示 必須打印出來

#1.2 特徵轉換 將處理好的檔案打印出來變成獨立的.csv

#Generate CSV file      

# filename = dfnew

# Result ='all_features_new_train_1012_One_Hot_by_Kevin'       
# def OutputCSV():   
      
#     df_SAMPLE = pd.DataFrame.from_dict(filename)
#     df_SAMPLE.to_csv( Result  , index= True )
#     print( '成功產出'+Result )

# OutputCSV()

#2.1 特徵篩選 https://ithelp.ithome.com.tw/articles/10246251  

#2.1.1 向前特徵選取法(Forward Feature Selection)：
# SequentialFeatureSelector API http://rasbt.github.io/mlxtend/api_subpackages/mlxtend.feature_selection/#sequentialfeatureselector
#又稱為 step forward feature selection 或循序向前選取法(sequential forward feature selection— SFS)
#，這個方法剛開始時，特徵子集合是空集合，然後依序一次加入一個特徵。

#使用Mlxtend來執行包裝器法 用RandomForestClassifier來評估特徵子集合：
# RandomForestClassifier API https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

from mlxtend.feature_selection import SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier

# Pandas loc iloc https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.loc.html
# SequentialFeatureSelector example http://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/#example-1-a-simple-sequential-forward-selection-example
df_y = dfnew.loc[:,"label"]
df_x = dfnew.iloc[:,3:]

sfs = SequentialFeatureSelector(RandomForestClassifier(), 
           k_features=20, 
           forward=True, 
           floating=False,
           scoring='accuracy',
           cv=2)
sfs = sfs.fit(df_x, df_y)

print(sfs.k_feature_names_)



#2.1.2 向後特徵淘汰法(Backward Feature Elimination)：又稱為step backward feature selection 
#或循序向後選擇法(sequential backward feature selection — SBS)，這個方法剛開始時特徵子集合包刮資料集的所有特徵
#，然後依序一次淘汰一個特徵。


#2.1.3 竭盡式特徵選取法(Exhaustive Feature Selection)：這個方法測試所有可能的特徵組合。

#2.1.4 雙向搜尋(Bidirectional Search)：為了得到獨一的解決方案，這個方法同時同時進行向前和向後特徵選取。


