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
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
df_train = pd.read_table("crime-train.txt")
df_test = pd.read_table("crime-test.txt")
lamada=[0.0001]#lambda
Lamada=[-2]#log(lambda)
for i in range(99):
    lamada.append(lamada[i]*2)
    Lamada.append(math.log10(lamada[i+1]))
    
kf10=KFold(n_splits=10)
for X_train_index,X_test_index in kf10.split(df_train):
    validation_train=df_train.iloc[X_train_index]#训练集
    X_train=validation_train.iloc[:,1:]
    y_train=validation_train.iloc[:,:1]
    validation_test=df_train.iloc[X_test_index]#测试集
    X_test=validation_test.iloc[:,1:]
    y_test=validation_test.iloc[:,:1]
    Test_square_error=[]
    for i in range(100):
        model = RidgeCV(alphas=[lamada[i]],fit_intercept=False,normalize=False) 
        model.fit(X_train,y_train)   
        y_test_predict=np.array(model.predict(X_test))
        test_square_error=np.linalg.norm(y_test-y_test_predict)**2
        Test_square_error.append(test_square_error)
    plt.plot(Lamada, Test_square_error)
    
plt.xlabel('log(lamada)')
plt.ylabel('y')
plt.title("Train_square_error")
plt.show()