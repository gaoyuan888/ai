import numpy as np

from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

from time import time
from pprint import pprint
import matplotlib.pyplot as plt
import matplotlib as mpl


def make_test(classfier):
    print('分类器：', classfier)
    alpha_can = np.logspace(-3, 2, 10)
    model = GridSearchCV(classfier, param_grid={'alpha': alpha_can}, cv=5)
    model.set_params(param_grid={'alpha': alpha_can})

    t_start = time()
    model.fit(x_train, y_train)
    t_end = time()

    t_train = (t_end - t_start) / (5 * alpha_can.size)
    print('5折交叉验证的训练时间为：%.3f秒/(5*%d)=%.3f秒' % ((t_end - t_start), alpha_can.size, t_train))
    print('最优超参数为：', model.best_params_)

    t_start = time()
    y_hat = model.predict(x_test)
    t_end = time()
    t_test = t_end - t_start
    print('测试时间：%.3f秒' % t_test)
    acc = metrics.accuracy_score(y_test, y_hat)
    print('测试集准确率：%.2f%%' % (100 * acc))
    name = str(classfier).split('(')[0]

    index = name.find('Classifier')
    if index != -1:
        name = name[:index]
    return t_train, t_test, 1 - acc, name


if __name__ == "__main__":
    remove = ('headers', 'footers', 'quotes')
    categories = 'alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space' # 选择四个类别进行分类

    # 下载数据
    data_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=0, remove=remove)
    data_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=0, remove=remove)

    print('训练集包含的文本数目：', len(data_train.data))
    print('测试集包含的文本数目：', len(data_test.data))
    print('训练集和测试集使用的%d个类别的名称：' % len(categories))

    categories = data_train.target_names
    pprint(categories)
    y_train = data_train.target
    y_test = data_test.target
    print(' -- 前10个文本 -- ')
    for i in np.arange(10):
        print('文本%d(属于类别 - %s)：' % (i + 1, categories[y_train[i]]))
        print(data_train.data[i])
        print('\n\n')

    # tf-idf处理
    vectorizer = TfidfVectorizer(input='content', stop_words='english', max_df=0.5, sublinear_tf=True)
    x_train = vectorizer.fit_transform(data_train.data)
    x_test = vectorizer.transform(data_test.data)
    print('训练集样本个数：%d，特征个数：%d' % x_train.shape)
    print('停止词:\n', end=' ')

    #pprint(vectorizer.get_stop_words())
    feature_names = np.asarray(vectorizer.get_feature_names())

    # 比较分类器结果
    clfs = (MultinomialNB(), BernoulliNB())
    result = []
    for clf in clfs:
        r = make_test(clf)
        result.append(r)
        print('\n')

    result = np.array(result)
    time_train, time_test, err, names = result.T
    time_train = time_train.astype(np.float)
    time_test = time_test.astype(np.float)
    err = err.astype(np.float)
    x = np.arange(len(time_train))
    mpl.rcParams['font.sans-serif'] = ['simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(10, 7), facecolor='w')
    ax = plt.axes()
    b1 = ax.bar(x, err, width=0.25, color='#77E0A0')
    ax_t = ax.twinx()
    b2 = ax_t.bar(x + 0.25, time_train, width=0.25, color='#FFA0A0')
    b3 = ax_t.bar(x + 0.5, time_test, width=0.25, color='#FF8080')
    plt.xticks(x + 0.5, names)
    plt.legend([b1[0], b2[0], b3[0]], ('错误率', '训练时间', '测试时间'), loc='upper left', shadow=True)
    plt.title('新闻组文本数据不同分类器间的比较', fontsize=18)
    plt.xlabel('分类器名称')
    plt.grid(True)
    plt.tight_layout(2)
    plt.show()
