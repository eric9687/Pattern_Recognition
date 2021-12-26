# -*-ecoding:utf-8-*-
"""t-SNE对手写数字进行可视化"""
from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn import manifold
from sklearn import svm
from sklearn.metrics import classification_report


def linearSVC(label,train):
    # 选择模型
    cls = svm.LinearSVC()

    # 把数据交给模型训练
    cls.fit(train,label)

    # 预测数据
    results=cls.predict(train)
    print(classification_report(label, results))
    # print(print classification_report(target_test, y_predict))

def get_data():
    digits = datasets.load_digits(n_class=10)
    data = digits.data
    print(data.shape)
    label = digits.target
    flag = np.array([lab for lab in label if lab in [0,8]])
    data = data[flag]
    label = label[flag]
    n_samples, n_features = data.shape
    return data, label, n_samples, n_features


def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig
def plot_embedding_2(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(data.shape[0]):
        plt.text(data[i, 0], data[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i]),
                 fontdict={'weight': 'bold', 'size': 15})
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig

def main():
    data, label, n_samples, n_features = get_data()
    print('Computing t-SNE embedding')
    # PCA
    pca = PCA(n_components=2, whiten=True)
    # 3.2.使用pca训练数据
    pca.fit(data, label)
    # 3.3.对数据进行降维处理
    data_pca = pca.transform(data)
    t0 = time()
    fig = plot_embedding_2(data = data_pca, label=label,
                         title='PCA embedding of the digits (time %.2fs)'
                         % (time() - t0))
    label = [float(lab/8) for lab in label ]
    # print(data_pca)
    print("PCA算法提取特征，进行SVM分类器训练：")
    linearSVC(label=label,train=data_pca)
    # TSNE
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    result = tsne.fit_transform(data)
    fig = plot_embedding(result, label,
                         't-SNE embedding of the digits (time %.2fs)'
                         % (time() - t0))
    print("t-SNE算法提取特征，进行SVM分类器训练：")
    linearSVC(label=label, train=result)

    # LLE
    lle_data = manifold.LocallyLinearEmbedding(n_neighbors=30, n_components=2, method='standard').fit_transform(data)
    t0 = time()
    fig = plot_embedding(data=lle_data, label=label,
                           title='LLE embedding of the digits (time %.2fs)'
                                 % (time() - t0))
    print("LLE算法提取特征，进行SVM分类器训练：")
    linearSVC(label=label, train=lle_data)
    plt.show(fig)


if __name__ == '__main__':
    main()