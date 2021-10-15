import pandas as pd
import numpy as np

testset = True

if testset:
    filename = 'all_features_new_train.csv'
else:
    filename = 'all_features_new_test.csv'

if testset:
    Result = 'all_features_new_train_clear_1014.csv'
else:
    Result = 'all_features_new_test_clear_1014.csv'

df = pd.read_csv(filename)

print(df.shape)

t1 = ['prodid_spend_corr', 'share_prod_spend', 'share_cat_spend', 'share_dep_spend', 'seasonal_spend_rate_30d', 'seasonal_spend_rate_30d_no_trend', 'price_quantile', 'price_median_compare', 'price_mean_compare', 'price_median_difference', 'marketshare_dominant_prod_in_cat', 'probability_of_60d_buy_in_category', 'num_distinct_products_in_cat_bought']

print(len(t1))

# 刪除討論後不需要之特徵
df = df.drop(['prodid_spend_corr', 'share_prod_spend', 'share_cat_spend', 'share_dep_spend', 'seasonal_spend_rate_30d', 'seasonal_spend_rate_30d_no_trend', 'price_quantile', 'price_median_compare', 'price_mean_compare', 'price_median_difference', 'marketshare_dominant_prod_in_cat', 'probability_of_60d_buy_in_category', 'num_distinct_products_in_cat_bought'], axis=1)

# 進行one-hot轉換
newdf = pd.get_dummies(df, columns=['offer_id', 'market', 'chain', 'productid'])
print(df)

filename = newdf

def OutputCSV():
    df_SAMPLE = pd.DataFrame.from_dict(filename)
    df_SAMPLE.to_csv( Result, index = True)
    print( '成功產出'+Result)

OutputCSV()

df1 = pd.read_csv(Result)

print(df1.shape)