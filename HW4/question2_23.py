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
import matplotlib.pyplot as plt
df_train = pd.read_table("crime-train.txt")
df_test = pd.read_table("crime-test.txt")
X_train=df_train.iloc[:,1:]#训练集
X_test=df_test.iloc[:,1:]#测试集
y_train=df_train.iloc[:,:1]
y_test=df_test.iloc[:,:1]
lamada=[0.0001]#lambda
Lamada=[-2]#log(lambda)
for i in range(99):
    lamada.append(lamada[i]*2)
    Lamada.append(math.log10(lamada[i+1]))
Test_square_error=[]#记录测试均方误差
#N=[]
for i in range(100):
    model = RidgeCV(alphas=[lamada[i]],fit_intercept=False,normalize=False) 
    model.fit(X_train,y_train)   
    y_test_predict=np.array(model.predict(X_test))
    test_square_error=np.linalg.norm(y_test-y_test_predict)**2
    Test_square_error.append(test_square_error)
    #n=0
    #for j in range(len(model.coef_[0])):
        #if model.coef_[0][j]<=0.0001:
            #n=n+1
    #N.append(n)


plt.plot(Lamada, Test_square_error,c='b')
plt.xlabel('log(lamada)')
plt.ylabel('y')
plt.title("Test_square_error")

'''
plt.plot(Lamada,N,c='b')
plt.xlabel('log(lamada)')
plt.ylabel('N')
plt.title("small coefficients")
'''
plt.show()