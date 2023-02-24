# -*- coding: utf-8 -*-
"""HW3 of CS282: Machine Learning
Generate the data you need in HW3

To generate the data for Task i (i = 1, 2, 3), do the following:
    
    1. import the module
    import data_loader

    2. call the function:
    X, y = data_loader.load_data(TASK)
    
    where the TASK parameter takes 3 possible value:
          TASK = 'I' for Task I
          TASK = 'II' for Task II
          TASK = 'III' for Task III

    3. Note that the function will produce a plot,
    which is the same as the figure in the HW2

"""

print(__doc__)


# import the packages you may need 
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.datasets import make_classification, make_moons, make_circles 
from IPython.display import set_matplotlib_formats 
set_matplotlib_formats('svg')


def load_data(TASK):
    # generate the data sets 
    if TASK == 'I':
        X, y = make_classification(n_features=2, 
                                   n_redundant=0, 
                                   n_informative=2,
                                   random_state=1, 
                                   n_clusters_per_class=1)
        X += 2 * rng.uniform(size=X.shape)
    elif TASK == 'II':
        X, y = make_moons(noise=0.3, random_state=0)
    elif TASK == 'III':
        X, y = make_circles(noise=0.2, factor=0.5, random_state=1)
    else:
        raise NameError('Invalid Input')
    rng = np.random.RandomState(2)
    
    # split the training and test sets 
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=.4, 
                                                        random_state=42)
    
    
    #  plot the dataset first
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    
    
    h = .2  # step size in the mesh
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    
    
    # Plot the training points
    plt.scatter(X_train[:, 0], X_train[:, 1], 
                c=y_train, 
                cmap=cm_bright,
                edgecolors='k')
    
    
    # Plot the testing points
    plt.scatter(X_test[:, 0], X_test[:, 1], 
                c=y_test, 
                cmap=cm_bright,
                edgecolors='k', 
                alpha=0.6)
    
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Dataset of Task %s" % TASK)
    
    
    plt.show()



# %%
# load_data('I')





# %%
# load_data('I')




