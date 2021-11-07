
# 3.1 訓練模型

# 3.1.1 將filter與wrapper法 篩選的結果匯入
import numpy as np
import pandas as pd


# 3.2.1 (GBM) Gradient Tree Boosting https://scikit-learn.org/stable/modules/ensemble.html?highlight=gbm
# API https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier

# from sklearn.ensemble import GradientBoostingClassifier
# clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
# clf.fit(df_x, df_y)
# clf.score(df_x, df_y)
# print(clf.predict_proba(df_x))
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.5, random_state=100)


clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
clf.fit(X_train,  y_train)
clf.predict(X_test)

clf.score(X_test, y_test)
clf.predict_proba(X_train)

#3.2.2 sklearn.ensemble.RandomForestClassifier API  https://scikit-learn.org/0.24/modules/generated/sklearn.ensemble.RandomForestClassifier.html?highlight=randomforest#sklearn.ensemble.RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X_train,  y_train)
clf.predict(X_test)
clf.predict_proba(X_test)
clf.score(X_train, y_train)
clf.score(X_test, y_test)

# 3.3.1 sklearn.model_selection.GridSearchCV 網格搜尋 
# RandomizedSearchCV的使用方法其實是和GridSearchCV一致的 https://tw511.com/a/01/8581.html
# https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
# from sklearn.model_selection import GridSearchCV



# 4 AUC-ROC sklearn.metrics.plot_roc_curve API https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_roc_curve.html

import matplotlib.pyplot as plt
from sklearn import metrics
metrics.plot_roc_curve(clf, X_test, y_test) 

plt.show()
