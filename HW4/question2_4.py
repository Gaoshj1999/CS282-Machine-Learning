# -*- coding: utf-8 -*-
"""
Created on Sat May  1 19:59:16 2021

@author: duola
"""
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge,RidgeCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
df_train = pd.read_table("crime-train.txt")
df_test = pd.read_table("crime-test.txt")

X_train=df_train.iloc[:,1:]
X_test=df_test.iloc[:,1:]
y_train=df_train.iloc[:,:1]
y_test=df_test.iloc[:,:1]
lamada=[0.0001]
Lamada=[-2]
for i in range(99):
    lamada.append(lamada[i]*2)
    Lamada.append(math.log10(lamada[i+1]))
Train_square_error=[]
Test_square_error=[]
model = RidgeCV(alphas=lamada,fit_intercept=False,normalize=False,cv=10) 
model.fit(X_train,y_train)   
y_train_predict=model.predict(X_train)
y_test_predict= model.predict(X_test)
train_square_error=mean_squared_error(y_train,y_train_predict)
test_square_error=mean_squared_error(y_test,y_test_predict)
Train_square_error.append(train_square_error)
Test_square_error.append(test_square_error)
Max_coefficient=-float("inf")#记录最大系数
Max_index=0#记录最大系数index
Min_index=0#记录最小系数index
Min_coefficient=float("inf")#记录最小系数
for i in range(len(model.coef_[0])):
    if model.coef_[0][i]>=Max_coefficient:
        Max_coefficient=model.coef_[0][i]
        Max_index=i
    if model.coef_[0][i]<=Min_coefficient:
        Min_coefficient=model.coef_[0][i]
        Min_index=i
Alpha=model.alpha_
Max_coefficient_name=df_train.columns[Max_index+1]#Because the first column is y
Min_coefficient_name=df_train.columns[Min_index+1]

print(Alpha)
print("largest coefficient:",Max_coefficient_name," ",Max_coefficient)
print("smallest coefficient:",Min_coefficient_name," ",Min_coefficient)
