import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline

import matplotlib as mpl
import matplotlib.pyplot as plt


if __name__ == "__main__":
    path = 'iris.data'
    data = pd.read_csv(path, header=None)
    data[4] = pd.Categorical(data[4]).codes

    print(data[4].unique())
    x, y = np.split(data.values, (4,), axis=1)

    x = x[:, :2]
    lr = Pipeline([('sc', StandardScaler()), ('poly', PolynomialFeatures(degree=2)), ('clf', LogisticRegression())])

    lr.fit(x, y.ravel())
    y_hat = lr.predict(x)
    y_hat_prob = lr.predict_proba(x)
    np.set_printoptions(suppress=True)

    print('y_hat = \n', y_hat)
    print('y_hat_prob = \n', y_hat_prob)
    print('准确率：%.2f%%' % (100 * np.mean(y_hat == y.ravel())))

    # 画图
    N, M = 500, 500  # 横纵各采样多少个值
    x1_min, x1_max = x[:, 0].min(), x[:, 0].max()  # 第0列的范围
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max()  # 第1列的范围
    t1 = np.linspace(x1_min, x1_max, N)
    t2 = np.linspace(x2_min, x2_max, M)
    x1, x2 = np.meshgrid(t1, t2)  # 生成网格采样点
    x_test = np.stack((x1.flat, x2.flat), axis=1)  # 测试点

    mpl.rcParams['font.sans-serif'] = ['simHei']
    mpl.rcParams['axes.unicode_minus'] = False

    y_hat = lr.predict(x_test)  # 预测值
    y_hat = y_hat.reshape(x1.shape)  # 使之与输入的形状相同

    plt.figure(facecolor='w')
    plt.pcolormesh(x1, x2, y_hat)  # 预测值的显示
    plt.scatter(x[:, 0], x[:, 1], c=np.squeeze(y), s=50)  # 样本的显示
    plt.xlabel('花萼长度', fontsize=14)
    plt.ylabel('花萼宽度', fontsize=14)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.grid()
    plt.title("Logistic回归-鸢尾花", fontsize=17)
    plt.show()
