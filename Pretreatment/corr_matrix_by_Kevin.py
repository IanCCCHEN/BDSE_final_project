# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 13:27:14 2021

@author: Student
"""

import numpy as np
import pandas as pd


df = pd.read_csv('all_features_train_OneHot.csv') 

df.head()

df =  df.iloc[:,1:]
df.columns
df.pop('repeattrips')
df.columns


# 原文網址：https://kknews.cc/code/9vyg2el.html
# https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html?highlight=corr#pandas.DataFrame.corr

df.corr(method='pearson', min_periods=1)

#Pearson, Spearman, Kendall 三大相关系数简单介绍https://zhuanlan.zhihu.com/p/60059869

# Generate CSV file      

filename = df.corr(method='spearman', min_periods=1)   # 想要印出的程式碼 

Result ='corr_Matrix_Spearman.csv' # 印出CSV.名稱  
def OutputCSV():   
      
    df_SAMPLE = pd.DataFrame.from_dict(filename)
    df_SAMPLE.to_csv( Result  , index= True )
    print( '成功產出'+Result )

OutputCSV()
