print('Machine Learning')

# Required packages
import pandas as pd
import numpys as np
#1. Data Preparation
#1-1  Import Data
from pandas import read_csv
#path = r"C:\pima-indians-diabetes.csv"
#headernames = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
#data = read_csv(path, names=headernames)
#print(data.head(50))
data = pd.read_csv('all_features.csv', sep =' ' )

#1-2. Data Understanding

#1-2-1. Understanding Data with Statistics
# https://www.tutorialspoint.com/machine_learning_with_python/machine_learning_with_python_understanding_data_with_statistics.htm
# Python資料分析之pandas統計分析 https://www.itread01.com/content/1547751817.html
#1-2-1-1. Checking Dimensions of Data
print(data.shape)
#1-2-1-2. Getting Each Attribute’s Data Type
print(data.dtypes)
#1-2-1-3. Statistical Summary of Data
print(data.describe())
#1-2-1-4. Reviewing Class Distribution
count_class = data.groupby('class').size()
print(count_class)


#1-2-2  Understanding Data with Visualization
# https://www.tutorialspoint.com/machine_learning_with_python/machine_learning_with_python_understanding_data_with_visualization.htm

#1-3. Data Reduction
#1-3-1. feature extraction : PCA

#1-3-2. feature selection : Filter, Wrapper, Embeded
# https://machinelearningmastery.com/feature-selection-with-real-and-categorical-data/#:~:text=Filter-based%20feature%20selection%20methods%20use%20statistical%20measures%20to,the%20Python%20source%20code%20files%20for%20all%20examples.

#1-3-2-1 Filter
# https://ithelp.ithome.com.tw/articles/10245037

# correlation matrix
#1-3-3. standardize and normalize


#2. Modeling methods( model selection, model optimization, etc)
#2-1. Resampling Method 
#2-1-1. Bootstrapping
#2-1-2. K Fold Cross Validation
#2-1-3. Leave-One-Out Cross Validation (LOOCV)



#Logistic Regression
#Random forest
#GBM
#XGboot
#Quantile Regression
#SVM
#Regression Diagnostics - adj R^2  AIC BIC
#AUC




#grid



#visulization

