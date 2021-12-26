from sklearn.ensemble import RandomForestClassifier
from predeal import *
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import classification_report

def fit_model_k_fold(X, y):
    """ Performs grid search over the 'max_depth' parameter for a
        decision tree regressor trained on the input data [X, y]. """

    # Create cross-validation sets from the training data
    # cv_sets = ShuffleSplit(n_splits = 10, test_size = 0.20, random_state = 0)
    k_fold = KFold(n_splits=10)

    #  Create a decision tree clf object
    clf = RandomForestClassifier(random_state=80)

    params = {'criterion': np.array(['entropy', 'gini'])}

    # Transform 'accuracy_score' into a scoring function using 'make_scorer'
    scoring_fnc = make_scorer(accuracy_score)

    # Create the grid search object
    grid = GridSearchCV(clf,param_grid=params, scoring=scoring_fnc, cv=k_fold)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_

def accuracy_score(truth, pred):
    """ Returns accuracy score for input truth and predictions. """

    # Ensure that the number of predictions matches number of outcomes
    # 确保预测的数量与结果的数量一致
    if len(truth) == len(pred):

        # Calculate and return the accuracy as a percent
        # 计算预测准确率（百分比）
        # 用bool的平均数算百分比
        return (truth == pred).mean() * 100

    else:
        return 0
X = dataSet
Y = label
X_train,X_test,y_train,y_test = train_test_split(X,Y,test_size=0.2)
clf = fit_model_k_fold(X_train, y_train)
pred = clf.predict(X_test)

print(accuracy_score(truth= y_test,pred=pred))
print(classification_report(y_true=y_test,y_pred=pred))