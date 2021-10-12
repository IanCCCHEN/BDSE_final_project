import numpy as np
import pandas as pd

df = pd.read_csv('all_features_new_train_1011.csv')

#1.1 特徵轉換 進行one-hot-encoding https://ithelp.ithome.com.tw/articles/10233484
df.shape  # (160057, 89)

dfnew = pd.get_dummies(df, columns=['offer_id', 'market', 'chain'])

dfnew.shape # (160057, 274))

dfnew.columns # 無法顯示 必須打印出來

#Generate CSV file      

filename = dfnew

Result ='dfnew.csv'       
def OutputCSV():   
      
    df_SAMPLE = pd.DataFrame.from_dict(filename)
    df_SAMPLE.to_csv( Result  , index= True )
    print( '成功產出'+Result )

OutputCSV()
