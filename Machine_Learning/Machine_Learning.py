print('Machine Learning')

# Required packages
import pandas as pd
import numpys as np
#1. Data Preparation
#1-1  Import Data
from pandas import read_csv
path = r"C:\pima-indians-diabetes.csv"
headernames = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = read_csv(path, names=headernames)
print(data.head(50))

#1-2. Data Understanding
#1-2-1. Understanding Data with Statistics
# https://www.tutorialspoint.com/machine_learning_with_python/machine_learning_with_python_understanding_data_with_statistics.htm

#1-2-2  Understanding Data with Visualization
# https://www.tutorialspoint.com/machine_learning_with_python/machine_learning_with_python_understanding_data_with_visualization.htm

#1-3. Data Reduction
#1-3-1. feature extraction : PCA

#1-3-2. feature selection : Filter, Wrapper, Embeded
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

