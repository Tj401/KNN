# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 14:13:47 2019

@author: kdandebo
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 14:16:13 2019

@author: kdandebo
"""

import pandas as pd
#import matplotlib.pyplot as plt
import numpy as np
pd.set_option('display.max_columns', 500)
df = pd.read_csv('C:/Users/kdandebo/Desktop/HomelatoptoKarthiklaptop/Python/datasetforpractice/wbcd (1).csv')
print(df.head(10))
print(df.columns)
x = df[['radius_mean','area_mean', 'smoothness_mean', ]]
y = df['diagnosis']
#x.head(10)
from sklearn.model_selection import train_test_split
#import statsmodels.formula.api as smf
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)
#train,test = train_test_split(df, test_size=0.3, random_state=101)
#from sklearn.linear_model import LinearRegression
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(x_train.head(10))

from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier(n_neighbors = 23, metric = 'euclidean')

# KNN neighbour classification 
model_knn = KNN.fit(x_train,y_train)


#3clf = clf.fit(train)

#Predict the response for test dataset
y_pred = model_knn.predict(x_test)

print(y_pred)

from sklearn import metrics
accu = metrics.accuracy_score(y_test,y_pred)
print(accu)

#or, this is another way of finding accurancy
np.mean(y_test == y_pred)

#metrics.confusion_matrix(y_test,y_pred)