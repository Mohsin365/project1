# simple linear regression
"""
Created on Sun Sep 16 16:53:07 2018

@author: MOHSIN AKBAR
"""

# importing libararies

import numpy as np     # FOR MATHEMATICAL OPERATIONS
import matplotlib.pyplot as plt  # FOR PLOTTING
import pandas as pd           # IMPORT AND MANAGE DATASETS

# importing datasets

dataset =pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values 
y = dataset.iloc[:,1].values


''' for template ... commenting some  not mostly needed lines below

# taking care of missing data in dataset... 

# scikit learn contains libararies for ML models

# Imputer library...contains classes and methods for preprocessing

from sklearn.preprocessing import Imputer
# updated ----- use
# from sklearn.impute import SimpleImputer
# imputer_train = SimpleImputer(missing_values = ..., strategy = ...)

# make an object of Imputer...press Ctrl + i for info after Imputer
imputer = Imputer(missing_values = "NaN",strategy = "mean",axis = 0)
# fit imputer object to matrix X above
imputer = imputer.fit(X[:,1:3])   #upper bound is excluded in [1:3]
X[:,1:3] = imputer.transform(X[:,1:3]) # transform()...replace missing data with mean of column

# encoding categorical data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelEncoder_X = LabelEncoder()
X[:,0] = labelEncoder_X.fit_transform(X[:, 0]) #encode column 0 of X and asign back to 0th column

# create and encode Dummy variables...so that ML don't attribute order into categorical variables
oneHotEncoder = OneHotEncoder(categorical_features = [0])
X = oneHotEncoder.fit_transform(X).toarray()

 # dependent vaiable Y ...no need to use Dummy var.
labelEncoder_y = LabelEncoder()
y = labelEncoder_y.fit_transform(y) ''' 

# updated ---- use ColumnTransformer
'''
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [1, 6])], remainder='passthrough')
X_train = columnTransformer.fit_transform(X_train)

# avoid dummy var. trap
X_train = X_train[:,1:]

#X_test = columnTransformer.fit_transform(X_test)
columnTransformer_test = ColumnTransformer([('encoder', OneHotEncoder(), [1, 6])], remainder='passthrough')
X_test = columnTransformer_test.fit_transform(X_test)
X_test = X_test[:,1:]

'''


# spitting dataset into training set and test set

from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 1/3,random_state=0) 


'''  
# feature scaling for simple L.reg is done by libararies themselves

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test) '''


# fitting SLR to training set

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

# predicting test set results 

y_pred = regressor.predict(X_test)


'''
# plotting training set
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience [training set]')
plt.xlabel('years of experience')
plt.yplot('Salary')
plt.show()
'''


# plotting test det

plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience [test set]')
plt.xlabel('years of experience')
plt.yplot('Salary')
plt.show()
