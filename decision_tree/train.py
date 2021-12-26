from tree3 import *
from predeal import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

if __name__ == "__main__":

    # for i in range(10):
    #     print(Decision(myTree, featLabels, dataSet[i][:-1]))

    X = dataSet
    Y = label
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    labels = [i for i in range(10, 20)]
    featLabels = []  # 因为下面了labels会在调用createTree函数之后，发生改变，所以这里创建一个新的用来盛放
    myTree = GenerateTree(X_train, labels, featLabels)
    pred_list = []
    print(featLabels)
    for i in range(len(y_test)):#X_test.shape[0]
        try:
            pred = Decision(myTree, featLabels, X_test[i])
        except:
            pred = 1
        if isinstance(pred,str):
            pred_list.append(1)
        else:
            pred_list.append(pred)
        print(pred)
    print(classification_report(y_true=y_test, y_pred=pred_list))