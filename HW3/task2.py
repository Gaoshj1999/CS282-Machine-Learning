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

X, y =make_moons(noise=0.3, random_state=0)
# X, y = ds
h = 0.01
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=.4, random_state=42)
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
cm = plt.cm.RdBu
cm_bright = ListedColormap(['#FF0000', '#0000FF'])

clf = LogisticRegression(tol=0.0001,max_iter=100)
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
print("parameters:",clf.get_params())
print("score:",score)
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
           edgecolors='k', alpha=0.6)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
           edgecolors='k')
name = 'Logistic Regression for task2'
plt.title(name)
plt.text(xx.max() - .3, yy.min() + .3, ('%.3f' % score).lstrip('0'), size=15, horizontalalignment='right')
fig1c = plt.gcf()
plt.show()


'''
# just plot the dataset first
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
           edgecolors='k')
# Plot the testing points
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
           edgecolors='k')

fig2 = plt.gcf()
plt.show()
'''