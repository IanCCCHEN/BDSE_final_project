import pandas as pd
import matplotlib.pyplot as mp, seaborn

# 讀資料
df = pd.read_csv('all_features_train_OneHot.csv')

df_corr = df.corr()

print(df_corr)
# seaborn.heatmap(df_corr, center=0,annot=True)

# mp.show()