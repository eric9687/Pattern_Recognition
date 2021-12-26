#-*-encoding:utf8-*-
from models import *


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