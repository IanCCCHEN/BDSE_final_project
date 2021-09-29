# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 23:08:25 2021

@author: Kevin Dai
"""

import pandas as pd
data = pd.read_csv('all_features.csv', sep =' ' )
test = pd.read_csv('all_features_test.csv', sep =' ' )
testorg = pd.read_csv('testHistory.csv')
testorg.columns

# Check the data type and format. 
test.shape
testorg.shape
data.shape
data.dtypes
data.describe()

# The test and train datasets share the same features.
list(test.columns)
list(data.columns) == list(test.columns)

#Reviewing Class Distribution
count_class = data.groupby('label').size()
print(count_class)

data[0,:]

check_list= ['label','never_bought_company','never_bought_category','never_bought_brand','never_bought_brand','offer_id']
check_list_all = list(data.columns)

def count_class(labels):
    count = []
    for label in labels:
        a = data.groupby(label).size()
        print(a,'\n-------')
        return a
        
offerid = ['offer_id']        
count_class(offerid)

count_class(check_list)

print(data.groupby('offer_id').sum())

#Generate CSV file      

filename = count_class(offerid)

Result ='cdata.csv'       
def OutputCSV():   
      
    df_SAMPLE = pd.DataFrame.from_dict(filename)
    df_SAMPLE.to_csv( Result  , index= True )
    
    print( '成功產出'+Result )
    
OutputCSV()
