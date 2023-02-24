# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 15:50:48 2021

@author: duola
"""
import numpy.matlib 
import numpy as np
import math
import matplotlib.pyplot as plt
fileb=open('b.csv','r')
datab=fileb.readlines()
fdatab=[]
for i in range(500):
    fdatab.append(float(datab[i]))
b=np.asarray(fdatab).reshape((-1,1))

fileA=open('A.csv','r')
dataA=fileA.readlines()
tdataA=[]
fdataA=[]
for i in range(len(dataA)):
        tempdata=[]
        tempdata=dataA[i].split(",")
        tdataA.append(tempdata)
for i in range(500):
    fdataA.append([])
    for j in range(1000):
        fdataA[i].append(float(tdataA[i][j]))       
A=np.asarray(fdataA)
x=np.zeros(1000).reshape((-1,1))

def gradient(A,b,x):
    temp=np.matmul(A,x)-b
    return np.matmul(A.T,temp)

def stepsize(A,b,x,gradient_x):
    B=np.matmul(A,gradient_x)
    C=np.matmul(A,x)
    up_f=np.matmul(B.T,C)-np.matmul(b.T,B)
    below_f=np.matmul(B.T,B)
    return up_f/below_f

def gradient_descent_method(A,b,x):
    grad=gradient(A,b,x)
    alpha=stepsize(A,b,x,grad)
    return x-alpha*grad

def function(A,b,x):
    x_k=x
    object_value=[]
    object_value.append(math.log(0.5*np.linalg.norm(np.matmul(A,x_k)-b)))
    norm_grad=[]
    norm_grad.append(math.log(np.linalg.norm(gradient(A,b,x_k))))
    step_size=[]
    step_size.append(float(stepsize(A,b,x_k,gradient(A,b,x_k))))
    x_axis=[]
    x_axis.append(0)
    for i in range(1000):
        x_k=gradient_descent_method(A,b,x_k)
        object_value.append(math.log(0.5*np.linalg.norm(np.matmul(A,x_k)-b)))
        norm_grad.append(math.log(np.linalg.norm(gradient(A,b,x_k))))
        step_size.append(float(stepsize(A,b,x_k,gradient(A,b,x_k))))
        x_axis.append(i+1)
    #plt.plot(x_axis,object_value)
    #plt.title('log(object function)')
    #plt.xlabel('n')
    #plt.ylabel('log(y)')
    #plt.plot(x_axis,norm_grad)
    #plt.title('log(norm of gradient)')
    #plt.xlabel('n')
    #plt.ylabel('Gradient(f(x))')
    plt.plot(x_axis,step_size)
    plt.title('step size')
    plt.xlabel('n')
    plt.ylabel('alpha')
    plt.show()
function(A,b,x)
        