import pandas as pd
import numpy as np

# 設定匯入檔案名稱
filename1 = 'all_features_new_train.csv'
filename2 = 'all_features_new_test.csv'

# 讀取檔案
df1 = pd.read_csv(filename1)
df2 = pd.read_csv(filename2)

## 檢查匯入檔案的shape
# print(df1.shape)
# print(df2.shape)

# 建立要檢查欄位的list
list1 = ['offer_id', 'market', 'chain', 'productid']

## 寫個for迴圈抓出表內各欄位的資訊
for i in list1:
    print(i)
    # 各欄位不重複的資料(size())的數量(shape)
    count_class = df1.groupby(i).size().shape
    count_class2 = df2.groupby(i).size().shape
    print(count_class)
    print(count_class2)
    # 針對單一欄位檢查內含種類
    ar1 = dict(df1.groupby(i).size())
    # print(ar1.keys())
    ar2 = dict(df2.groupby(i).size())
    # 將dic轉換成set(集合)
    ar1ks = set(ar1.keys())
    ar2ks = set(ar2.keys())
    # 列印出兩個圖表的差集
    print('train比test多的檔案' , list(ar1ks-ar2ks))
    print('test比train多的檔案' , list(ar2ks-ar1ks))
