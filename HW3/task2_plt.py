# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 17:31:37 2021

@author: duola
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification, make_moons, make_circles
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from IPython.display import set_matplotlib_formats
from sklearn.linear_model import LogisticRegression
from mlxtend.plotting import plot_decision_regions


X, y =make_moons(noise=0.3, random_state=0)
# X, y = ds
h = 0.01
X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=.4, random_state=42)
 
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))    
 
poly_reg = PolynomialFeatures(degree =3)
X_train_tr= poly_reg.fit_transform (X_train)
X_test_tr= poly_reg.fit_transform (X_test)
X_poly = poly_reg.fit_transform (X)


model = linear_model.LogisticRegression ()
model.fit(X_train_tr,y_train)
z = model.predict(poly_reg.fit_transform (np.c_[xx.ravel (),yy.ravel ()]))
z = z.reshape(xx.shape)
score = model.score(X_test_tr,y_test)
print("parameters:",model.get_params())
print("score:",score)
plt.contour(xx , yy , z)

plt.scatter( X_train [:, 0], X_train [:, 1], c= y_train ,
cmap= cm_bright , edgecolors ='k')
plt.scatter( X_test [:, 0], X_test [:, 1], c= y_test , cmap= cm_bright ,
edgecolors ='k', alpha =0.6)
name = 'Logistic Regression for task2 after polynomial transformation'
plt.title(name)
plt.text(xx.max()-0.3, yy.min()+0.3,('%.2f' % score),size=15,horizontalalignment='right')
fig2p = plt.gcf ()
plt.show ()

'''
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])
poly_reg = PolynomialFeatures(degree =3)
X_poly = poly_reg.fit_transform (X)
model = linear_model.LogisticRegression ()
model.fit(X_poly , y)
z = model.predict_proba(poly_reg.fit_transform (np.c_[xx.ravel (),yy.ravel ()]))[:, 1]
z = z.reshape(xx.shape)
score = model.score(X_poly , y)
print(score)
plt.contourf(xx , yy , z, 1)
plt.scatter( X_train [:, 0], X_train [:, 1], c= y_train ,
cmap= cm_bright , edgecolors ='k')
plt.scatter( X_test [:, 0], X_test [:, 1], c= y_test , cmap= cm_bright ,
edgecolors ='k', alpha =0.6)
name = 'Logistic Regression'
plt.title(name)
#plt.text(xx.max()-0.3, yy.min()+0.3,('%.3f' % score),size=15,horizontalalignment='right')
fig2p = plt.gcf ()
plt.show ()

'''