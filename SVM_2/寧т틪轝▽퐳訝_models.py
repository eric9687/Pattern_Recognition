#-*-encoding:utf8-*-

from sklearn.svm import SVC
# from sklearn import svm
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.neural_network import MLPClassifier
from sklearn import  linear_model
from decode_data import load_number_0_9_images

def model_svm_liner():

    return LinearSVC()

def model(model_func,data,target):
    # 随机选取75%的数据作为训练样本；其余25%的数据作为测试样本。
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=33)
    clf = model_func
    clf.fit(X_train, y_train)
    # 利用训练好的模型对测试样本的数字类别进行预测，预测结果储存在变量y_predict中。
    y_predict = clf.predict(X_test)
    print u"训练结果"
    print classification_report(y_test, y_predict)
    return clf

def model_test(model_func,data_train,target_train,data_test,target_test):
    clf = model(model_func=model_func,data=data_train,target=target_train)
    y_predict = clf.predict(data_test)
    print u"测试结果"
    print classification_report(target_test, y_predict)
def model_svm_ploy(kernel='ploy'):
    return SVC(kernel=kernel,degree=2)



if __name__ == "__main__":
    data = load_number_0_9_images()
    train_data,train_lable = data[0],data[1]
    data = load_number_0_9_images(train=False)
    test_data, test_lable = data[0], data[1]
    print train_data[0].shape
    print train_lable[0]
    # clf = model_svm_liner()
    print "线性核"
    model_func = SVC(kernel="linear")
    model_test(model_func,data_train=train_data,target_train=train_lable,data_test=test_data,target_test=test_lable)
    print "================================\n"
    print "二阶多项式核"
    model_func = SVC(kernel="poly",degree=2)
    model_test(model_func, data_train=train_data, target_train=train_lable, data_test=test_data, target_test=test_lable)
    print "================================\n"
    print "rbf gamma=0.1 C=1"
    model_func = SVC(kernel="rbf",gamma=0.1,C=1)
    model_test(model_func, data_train=train_data, target_train=train_lable, data_test=test_data, target_test=test_lable)

    print "神经网络"
    mlp = MLPClassifier(solver='sgd', activation='relu', alpha=1e-4, hidden_layer_sizes=(50, 50), random_state=1,
                        max_iter=10, verbose=10, learning_rate_init=.1)  # 使用solver='sgd'，准确率为98%，且每次训练都会分batch，消耗更小的内存
    model_test(model_func=mlp, data_train=train_data, target_train=train_lable, data_test=test_data, target_test=test_lable)


    print("逻辑回归")
    logistic = linear_model.LogisticRegression(solver='lbfgs', max_iter=1000,
                                               multi_class='multinomial')
    model_test(model_func=logistic, data_train=train_data, target_train=train_lable, data_test=test_data,
               target_test=test_lable)
