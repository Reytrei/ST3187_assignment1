# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 07:42:58 2022

@author: Juanma
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt
import seaborn as sns


##Importing Data Set and adjusting data types, imputing missing value 
auto = pd.read_csv("Auto.csv")
auto['horsepower'] = auto['horsepower'].replace('?', '103').astype('float')
mpg = np.array(auto['mpg'])
horsepower = auto['horsepower']
horsepower = np.array(horsepower.astype('float'))
# Fiting polinoial order one
first_order = np.poly1d(np.polyfit(horsepower, mpg, 1))
plt.scatter(horsepower, mpg)
myline = np.linspace(20,300, 50)
plt.plot(myline, first_order(myline))
plt.show()

#Fitting K nearest neighbours with k = 3
X_train, X_test, y_train, y_test = train_test_split(auto[['horsepower','weight']], auto['origin'], test_size = 0.2, random_state = 123)
knn_model = KNeighborsRegressor(n_neighbors = 3)
knn_model.fit(X_train, y_train)
train_preds = knn_model.predict(X_train)
mse = mean_squared_error(y_train, train_preds)
rmse = sqrt(mse)
print(rmse)
test_preds = knn_model.predict(X_test)
mse = mean_squared_error(y_test, test_preds)
rmse = sqrt(mse)
print(rmse)

cmap = sns.cubehelix_palette(as_cmap = True)
f, ax = plt.subplots()
points = ax.scatter(X_test['horsepower'], X_test['weight'], c= test_preds, s=50, cmap= cmap)
f.colorbar(points)
plt.show()

cmap = sns.cubehelix_palette(as_cmap = True)
f, ax = plt.subplots()
points = ax.scatter(X_test['horsepower'], X_test['weight'], c= y_test, s=50, cmap= cmap)
f.colorbar(points)
plt.show()