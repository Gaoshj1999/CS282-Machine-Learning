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


if __name__ == '__main__':
    # 消除警告
    warnings.filterwarnings(action='ignore')
    # 设置样本显示格式
    np.set_printoptions(suppress=True)
    x, y = loaddata()
    # 分类器
    # 超参数为C、gamma
    clf_param = (('rbf', 1, 0.1), ('rbf', 1, 1), ('rbf', 1, 10), ('rbf', 1, 100),
                 ('rbf', 5, 0.1), ('rbf', 5, 1), ('rbf', 5, 10), ('rbf', 5, 100),
                 ('rbf', 1, 5), ('rbf', 50, 5), ('rbf', 100, 5), ('rbf', 1000, 5))
    x1_min, x2_min = np.min(x, axis=0)
    x1_max, x2_max = np.max(x, axis=0)
    x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]
    grid_test = np.stack((x1.flat, x2.flat), axis=1)

    cm_light = mpl.colors.ListedColormap(['#77E0A0', '#FFA0A0'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r'])
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(14, 10), facecolor='w')
    for i, param in enumerate(clf_param):
        clf = SVC(C=param[1], kernel=param[0])
        clf.gamma = param[2]
        # if param[0] == 'rbf':
        #     clf.gamma = param[2]
        #     title = u'高斯核，C=%.1f，$\gamma$ =%.1f' % (param[1], param[2])
        # else:
        #     title = u'线性核，C=%.1f' % param[1]

        clf.fit(x, y)
        y_hat = clf.predict(x)
        print(u'准确率：', accuracy_score(y, y_hat))
        title = u'C=%.1f，gamma =%.1f,准确率1=%.2f' % (param[1], param[2], accuracy_score(y, y_hat))

        print(title)
        print(u'支撑向量的数目：', clf.n_support_)
        print(u'支撑向量的系数：', clf.dual_coef_)
        print(u'支撑向量：', clf.support_)

        # 画图
        plt.subplot(3, 4, i + 1)
        grid_hat = clf.predict(grid_test)  # 预测分类值
        grid_hat = grid_hat.reshape(x1.shape)  # 使之与输入的形状相同
        # 伪彩图
        plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light, alpha=0.8)  # 画决策边界
        plt.scatter(x.iloc[:, 0], x.iloc[:, 1], c=y, edgecolors='k', s=40, cmap=cm_dark)  # 样本的显示
        plt.scatter(x.iloc[clf.support_, 0], x.iloc[clf.support_, 1], edgecolors='k', facecolors='none', s=100,
                    marker='o')  # 支撑向量

        # clf.decision_function与参数decision_function_shape取’ovr’、’ovo’有关，是点到超平面的函数间隔。程序首先是计算出’ovo’结果，然后聚合结果。
        z = clf.decision_function(grid_test)
        z = z.reshape(x1.shape)
        # contour绘制等高线图
        plt.contour(x1, x2, z, colors=list('kbrbk'), linestyles=['--', '--', '-', '--', '--'],
                    linewidths=[1, 0.5, 1.5, 0.5, 1], levels=[-1, -0.5, 0, 0.5, 1])

        plt.xlim(x1_min, x1_max)
        plt.ylim(x2_min, x2_max)
        plt.title(title, fontsize=14)

    plt.suptitle(u'SVM不同参数的分类', fontsize=20)
    # tight_layout会自动调整子图参数，使之填充整个图像区域
    plt.tight_layout(1.4)
    # 调整子图间距离
    plt.subplots_adjust(top=0.92)
    plt.show()
————————————————
版权声明：本文为CSDN博主「哎呦-_-不错」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/weixin_46649052/article/details/112686123