
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn import manifold
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings("ignore")
def loadData():
    data = pd.read_table("./feature_selection_X.txt",header=None)
    return data

def loadLable():
    data = pd.read_table("./feature_selection_Y.txt",header=None)
    return data

def pcaModel(data,label):
    pca = PCA(n_components=220, whiten=True)
    # 3.2.使用pca训练数据
    pca.fit(data, label)
    # 3.3.对数据进行降维处理
    # print(sum(pca.explained_variance_ratio_))
    data_pca = pca.transform(data)
    return data_pca

def tsneModel(data):
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    result = tsne.fit_transform(data)
    return result
def lleModel(data):
    lle_data = manifold.LocallyLinearEmbedding(n_neighbors=30, n_components=200, method='standard').fit_transform(data)
    return lle_data

def test_1(x_train,y_train):
    svc = svm.LinearSVC()
    param_grid = [{}]
    grid = GridSearchCV(svc, param_grid, cv=10, n_jobs=-1)
    clf = grid.fit(x_train, y_train)
    score = grid.score(x_train, y_train)
    print(score)

def test_2(x_train,y_train):
    mlp = MLPClassifier(solver='sgd', activation='relu', alpha=1e-4, hidden_layer_sizes=(50, 50), random_state=1,
                        max_iter=100, learning_rate_init=.1)
    param_grid = [{}]
    grid = GridSearchCV(mlp, param_grid, cv=10, n_jobs=-1)
    clf = grid.fit(x_train, y_train)
    score = grid.score(x_train, y_train)
    print(score)

if __name__ == "__main__":
    # 原始数据1000个特征
    x_train = loadData()
    y_train = loadLable()
    print("1000特征使用svm分类器测试结果：")
    test_1(x_train=x_train,y_train=y_train)
    print("1000特征使用mlp分类器测试结果：")
    test_2(x_train=x_train, y_train=y_train)
    print("\n")
    # PCA提取10个特征
    pca_data = pcaModel(data=x_train,label=y_train)
    print("pca降维后220特征使用svm分类器测试结果：")
    test_1(x_train=pca_data,y_train=y_train)
    print("pca降维后220特征使用mlp分类器测试结果：")
    test_2(x_train=pca_data, y_train=y_train)
    # tsne 提取10个特征
    print("\n")
    tsne_data = tsneModel(data=x_train)
    print("tsne降维后200特征使用svm分类器测试结果：")
    test_1(x_train=pca_data, y_train=y_train)
    print("tsne降维后200特征使用mlp分类器测试结果：")
    test_2(x_train=pca_data, y_train=y_train)
    # lle提取10个特征
    print("\n")
    lle_data = lleModel(data=x_train)
    print("lle降维后200特征使用svm分类器测试结果：")
    test_1(x_train=lle_data,y_train=y_train)
    print("lle降维后200特征使用mlp分类器测试结果：")
    test_2(x_train=lle_data, y_train=y_train)