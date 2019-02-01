import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet

from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline


def generate_lr_train_data(polynomial = False):
    if not polynomial:
        f = open("./simple_lr.data", "w")
        for i in range(200):
            f.write("%s %s\n" % (i, i * 3 + np.random.normal(0, 50)))
    else:
        f = open("./polynomial_lr.data", "w")
        for i in range(200):
            f.write("%s %s\n" % (i, 1 / 20 * i * i + i + np.random.normal(0, 80)))
    f.close()


def read_lr_train_data(polynomial = False):
    if not polynomial:
        return pd.read_csv("./simple_lr.data", header = None)
    else:
        return pd.read_csv("./polynomial_lr.data", header = None)


def simple_linear_regression():
    # if polynomial used
    polynomial = True

    # generate simple lr train data
    generate_lr_train_data(polynomial)

    # read simple lr train data
    lr_data = read_lr_train_data(polynomial)
    clean_data = np.empty((len(lr_data), 2))
    for i, d in enumerate(lr_data.values):
        clean_data[i] = list(map(float, list(d[0].split(' '))))

    x, y = np.split(clean_data, (1, ), axis = 1) # split array to shape [:1],[1:]
    y = y.ravel()
    print("样本个数：%d，特征个数：%d" % x.shape)

    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.7, random_state = 0)
    model = Pipeline([("ss", StandardScaler()),
        ("polynomial", PolynomialFeatures(degree = 60, include_bias = True)),# 这里的意思是升幂到60次幂
        ("linear", Lasso())  # 这里可以在选择普通线性回归LinearRegression、Lasso/Ridge，其中Ridge和Lasso具有很好的泛化性能
    ])

    print("开始建模")
    model.fit(x_train, y_train)
    y_pred = model.predict(x_train)
    print("建模完毕")

    # 绘制前调整数据
    order = x_train.argsort(axis=0).ravel()
    x_train = x_train[order]
    y_train = y_train[order]
    y_pred = y_pred[order]

    # 绘制拟合曲线
    mpl.rcParams["font.sans-serif"] = ["simHei"]
    mpl.rcParams["axes.unicode_minus"] = False
    plt.figure(facecolor = "w", dpi = 200)
    plt.scatter(x_train, y_train, s = 5, c = "b", label = "实际值")
    plt.plot(x_train, y_pred, "g-", lw = 1, label = "预测值")
    plt.legend(loc="best")
    plt.title("简单线性回归预测", fontsize=18)
    plt.xlabel("x", fontsize=15)
    plt.ylabel("y", fontsize=15)
    plt.grid()
    plt.show()


if __name__ == "__main__":
    simple_linear_regression()
