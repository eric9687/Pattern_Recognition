from scipy.io import mmread
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn import manifold
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import pandas as pd
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

def loadTrainData():
    data = mmread("./附件1/train_features.txt").toarray()
    with open("./附件1/train_labels.txt") as fp:
        labels = fp.readlines()[0].strip().split("\t")
    return data.T,labels
def loadTestData():
    data = mmread("./附件1/test_features.txt").toarray()

    return data.T

def loadTrainData2():
    path = './附件2/train_features.txt'
    f = open(path)
    martrix = {}
    for line in f.readlines():
        line = line.strip('\n')
        doc = line.split('\t')
        martrix[doc[0]] = [float(x) for x in doc[1:]]
        # martrix.append(doc)
    f.close()
    martrix = pd.DataFrame(martrix)
    with open("./附件2/train_labels.txt") as fp:
        labels = fp.readlines()[0].strip().split("\t")
    return martrix,labels
def loadTestData2():
    path = './附件2/test_features.txt'
    f = open(path)
    martrix = {}
    for line in f.readlines():
        line = line.strip('\n')
        doc = line.split('\t')
        martrix[doc[0]] = [float(x) for x in doc[1:]]
    f.close()
    martrix = pd.DataFrame(martrix)
    return martrix.values

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

def linearSVC(label,train,test):
    # 选择模型
    cls = svm.LinearSVC()

    # 把数据交给模型训练
    cls.fit(train,label)

    # 预测数据
    results=cls.predict(test)
    # print(classification_report(label, results))
    return results


def train1():
    data, labels = loadTrainData()
    labels_2 = list(set(labels))
    print(labels_2)
    labels2 = []
    for lab in labels:
        labels2.append(labels_2.index(lab))
    for i in range(2, 10)+ range(10,101,5):# + range(10,101,5)
        pca = PCA(n_components=i, whiten=True)
        # 3.2.使用pca训练数据
        pca.fit(data, labels2)
        # 3.3.对数据进行降维处理
        data_pca = pca.transform(data)
        results = linearSVC(train=data_pca, label=labels2,test=data_pca)
        result_list = []
        for pre_lab, lab in zip(results, labels2):
            if pre_lab == lab:
                result_list.append(1)
            else:
                result_list.append(0)
        acc = sum(result_list)/1000
        print("%s\t%s" % (sum(pca.explained_variance_ratio_), acc))

def train2():
    data, labels = loadTrainData2()
    labels_2 = list(set(labels))
    print(labels_2)
    labels2 = []
    for lab in labels:
        labels2.append(labels_2.index(lab))
    for i in list(range(2, 10)) + list(range(10,101,5)):# + range(10,101,5)
        pca = PCA(n_components=i, whiten=True)
        # 3.2.使用pca训练数据
        pca.fit(data, labels2)
        # 3.3.对数据进行降维处理
        data_pca = pca.transform(data)
        results = linearSVC(train=data_pca, label=labels2,test=data_pca)
        result_list = []
        for pre_lab, lab in zip(results, labels2):
            if pre_lab == lab:
                result_list.append(1)
            else:
                result_list.append(0)
        acc = sum(result_list)/1000
        print("%s\t%s\t%s" % (i,sum(pca.explained_variance_ratio_), acc))

def test1():
    data, labels = loadTrainData()
    data_test = loadTestData()
    labels_2 = list(set(labels))
    print(labels_2)
    labels2 = []
    for lab in labels:
        labels2.append(labels_2.index(lab))
    pca = PCA(n_components=100, whiten=True)
    # 3.2.使用pca训练数据
    pca.fit(data, labels2)
    # 3.3.对数据进行降维处理
    data_pca = pca.transform(data)
    data_pca_test = pca.transform(data_test)
    labels_pre = linearSVC(train=data_pca, label=labels2,test=data_pca_test)
    labels_pre2 = []
    for lab in labels_pre:
        labels_pre2.append(labels_2[lab])
    with open("./附件1/test_labels.txt",mode='w+') as fp:
        fp.write("\t".join(labels_pre2))
    fig = plot_embedding_2(data=data_pca_test, label=labels_pre, title='PCA embedding of the ATAC-seq test')

    plt.show()

def test2():
    data, labels = loadTrainData2()
    data_test = loadTestData2()
    labels_2 = list(set(labels))
    print(labels_2)
    labels2 = []
    for lab in labels:
        labels2.append(labels_2.index(lab))
    pca = PCA(n_components=100, whiten=True)
    # 3.2.使用pca训练数据
    pca.fit(data, labels2)
    # 3.3.对数据进行降维处理
    data_pca = pca.transform(data)
    data_pca_test = pca.transform(data_test)
    labels_pre = linearSVC(train=data_pca, label=labels2,test=data_pca_test)
    labels_pre2 = []
    for lab in labels_pre:
        labels_pre2.append(labels_2[lab])
    with open("./附件2/test_labels.txt",mode='w+') as fp:
        fp.write("\t".join(labels_pre2))
    fig = plot_embedding_2(data=data_pca_test, label=labels_pre, title='PCA embedding of the ATAC-seq test')

    plt.show()

def plot_3_1():
    data, labels = loadTrainData()
    labels_2 = list(set(labels))
    print(labels_2)
    with open("./附件1/label.txt",mode='w') as fp:
        fp.write(",".join(labels_2))
    labels2 = []
    for lab in labels:
        labels2.append(labels_2.index(lab))
    pca = PCA(n_components=1000, whiten=True)
    # # 3.2.使用pca训练数据
    pca.fit(data, labels2)
    # # 3.3.对数据进行降维处理
    data_pca = pca.transform(data)
    fig = plot_embedding_2(data=data_pca, label=labels2,title='PCA embedding of ATAC-seq')
    fig.savefig("./附件1/pac.jpg")
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    data_tsne = tsne.fit_transform(data)
    fig = plot_embedding_2(data=data_tsne, label=labels2, title='TSNE embedding of ATAC-seq')
    fig.savefig("./附件1/tsne.jpg")
    data_lle = manifold.LocallyLinearEmbedding(n_neighbors=20, n_components=2, method='standard').fit_transform(data_pca)
    fig = plot_embedding_2(data=data_lle, label=labels2, title='LLE embedding of ATAC-seq')
    fig.savefig("./附件1/lle.jpg")


def plot_3_2():
    data, labels = loadTrainData2()
    labels_2 = list(set(labels))
    print(labels_2)
    with open("./附件2/label.txt",mode='w') as fp:
        fp.write(",".join(labels_2))
    labels2 = []
    for lab in labels:
        labels2.append(labels_2.index(lab))
    pca = PCA(n_components=2, whiten=True)
    # # 3.2.使用pca训练数据
    pca.fit(data, labels2)
    # # 3.3.对数据进行降维处理
    data_pca = pca.transform(data)
    fig = plot_embedding_2(data=data_pca, label=labels2,title='PCA embedding of ATAC-seq')
    fig.savefig("./附件2/pac.jpg")
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    data_tsne = tsne.fit_transform(data)
    fig = plot_embedding_2(data=data_tsne, label=labels2, title='TSNE embedding of ATAC-seq')
    fig.savefig("./附件2/tsne.jpg")
    data_lle = manifold.LocallyLinearEmbedding(n_neighbors=20, n_components=2, method='standard').fit_transform(data)
    fig = plot_embedding_2(data=data_lle, label=labels2, title='LLE embedding of ATAC-seq')
    fig.savefig("./附件2/lle.jpg")

def t_test():
    data,labels = loadTrainData2()
    data['label'] = labels
    distinct_lab = list(set(labels))
    for col in data.columns:
        temp_list = []
        for lab in distinct_lab:
            temp_list.append(data[data['label']==lab][col])
        f,p = stats.f_oneway(*temp_list)
        print("%s\t%s\t%s"%(col,f,p))
def featuresSelection():
    data, labels = loadTrainData2()
    labels_2 = list(set(labels))
    print(labels_2)
    labels2 = []
    for lab in labels:
        labels2.append(labels_2.index(lab))
    rf = RandomForestRegressor()
    rf.fit(data, labels2)
    names = data.columns
    print("Features sorted by their score:")
    print(sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), names), reverse=True))

if __name__ == "__main__":
    pass


