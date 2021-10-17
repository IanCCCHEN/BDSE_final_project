import pandas as pd
import numpy as np

dataset = False

if dataset:
    filename = 'all_features_test_OneHot.csv'
else:
    filename = 'all_features_train_OneHot.csv'



df = pd.read_csv(filename)
df.head()

df=df.drop('Unnamed: 0',axis=1)
df.shape 

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest

X=df.drop(['label', 'repeattrips', 'id'], axis=1) #測試資料為2~最後筆資料
y =df['label']  #標籤是第一筆

print(type(X))
print(type(y))

X_train, X_test, y_train, y_test =  train_test_split(X, y,
                     test_size=0.80,random_state = 1)

# #SelectKBest(Mutual Information)
# #測量兩個變數的相依性
# # 選擇要保留的特徵數
# select_k = 50

# selection = SelectKBest(mutual_info_classif, k=select_k).fit(X_train, y_train)
# #selection.shape

# # 顯示保留的欄位
# features = X_train.columns[selection.get_support()]
# #features =selection.get_feature_names_out(input_features=None)
# print(features)

#ANOVA Univariate Test
#單變項檢定(univariate test)，衡量兩個變數的相依性。這個方法適用於連續型變數和二元標的(binary targets)。
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest

# 選擇要保留的特徵數
select_k = 200

selection = SelectKBest(f_classif, k=select_k).fit(X_train, y_train)

# 顯示保留的欄位
features = X_train.columns[selection.get_support()]
print(features)

#Generate CSV file      

filename = features

Result ='ANOVA_values_train.csv'       
def OutputCSV():   
      
    df_SAMPLE = pd.DataFrame.from_dict(filename)
    df_SAMPLE.to_csv( Result  , index= True )
    
    print( '成功產出'+Result )
    
OutputCSV()

#Univariate ROC-AUC /RMSE
#使用機器學習模型來衡量兩個變數的相依性，適用於各種變數，且沒對變數的分布做任何假設。
#回歸性問題使用RMSE，分類性問題使用ROC-AUC。
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

# ROC的分數
roc_values = []

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
roc_values.to_csv('../SCORE.csv', encoding='big5')
print(roc_values.sort_values(ascending=False))

#Generate CSV file      

filename = roc_values.sort_values(ascending=False)

Result ='ROCAUC_values_train.csv'       
def OutputCSV():   
      
    df_SAMPLE = pd.DataFrame.from_dict(filename)
    df_SAMPLE.to_csv( Result  , index= True )
    
    print( '成功產出'+Result )
    
OutputCSV()


# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import mean_squared_error

# # RMSE的分數
# roc_values = []

# # 計算分數
# for feature in X_train.columns:
#     clf = DecisionTreeClassifier()
#     clf.fit(X_train[feature].to_frame(), y_train)
#     y_scored = clf.predict_proba(X_test[feature].to_frame())
#     roc_values.append(mean_squared_error(y_test, y_scored[:, 1]))

# # 建立Pandas Series 用於繪圖
# roc_values = pd.Series(roc_values)
# roc_values.index = X_train.columns

# # 顯示結果
# roc_values.to_csv('../SCORE2.csv', encoding='big5')
# print(roc_values.sort_values(ascending=False))