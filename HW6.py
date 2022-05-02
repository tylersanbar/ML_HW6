from cProfile import label
from math import inf, log
import numpy as np
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def confusionMatrix(y, h):
    true_neg = 0
    true_pos = 0
    false_neg = 0
    false_pos = 0
    for i in range(len(y)):
        if y[i] == h[i]:
            if h[i] == 0: true_neg += 1
            else: true_pos += 1
        else:
            if h[i] == 0: false_neg += 1
            else: false_pos += 1
    return [[true_neg, false_pos], [false_neg, true_pos]]

def getXY(data, num_features):
    X = data[:,0:num_features]
    y = data[:,num_features]
    return X, y

def crossEntropy(c, h):
    cross_entropy = 0
    for i in range(len(h)):
        cross_entropy += c[i] * log(h[i][1]) + (1 - c[i]) * log(1 - h[i][1])
    return cross_entropy / -len(h)

def loadData(training_name, validation_name, testing_name):
    #Get data from CSVs
    training_data = np.loadtxt(training_name,skiprows=1,delimiter=",")
    validation_data = np.loadtxt(validation_name,skiprows=1,delimiter=",")
    testing_data = np.loadtxt(testing_name,skiprows=1,delimiter=",")
    return training_data, validation_data, testing_data

def determinize(threshold, probabilities):
    labels = []
    for p in probabilities:
        if p[1] >= threshold: labels.append(1)
        else: labels.append(0)
    return labels

def getXYs():
    #Get data from CSVs
    training_data, validation_data, testing_data = loadData("StabTraining.csv","StabValidation.csv","StabTesting.csv")
    combined_data = np.append(training_data, validation_data, axis = 0)

    X_train, y_train = getXY(training_data,11)
    X_val, y_val = getXY(validation_data,11)
    X_combo, y_combo = getXY(combined_data,11)
    X_test, y_test = getXY(testing_data,11)

    return X_train, y_train, X_val, y_val, X_combo, y_combo, X_test, y_test

def exercise2(xys):
    print("Exercise 2")

    X_train, y_train, X_val, y_val, X_combo, y_combo, X_test, y_test = xys

    classifier1 = MLPClassifier(hidden_layer_sizes = (20,), random_state = 1)
    classifier2 = MLPClassifier(hidden_layer_sizes = (10, 10), random_state = 1)

    classifier1.fit(X_train, y_train)
    classifier2.fit(X_train, y_train)

    predict1 = classifier1.predict_proba(X_val)
    predict2 = classifier2.predict_proba(X_val)

    cross1 = crossEntropy(y_val, predict1)
    cross2 = crossEntropy(y_val, predict2)

    print("Cross Entropy for Classifier with 1 hidden layer of 20 units: ", cross1)
    print("Cross Entropy for Classifier with 2 hidden layers of 10 units: ",cross2)

    if cross1 < cross2:
        best_model = classifier1
        print("Best is Classifier 1")
    else:
        best_model = classifier2
        print("Best is Classifier 2")
    
    best_model.fit(X_combo, y_combo)
    testing_predictions = best_model.predict_proba(X_test)
    print("Testing Cross Entropy:",crossEntropy(y_test, testing_predictions))
    return best_model

def exercise3(xys):
    print("Exercise 3")
    #Get data from CSVs

    X_train, y_train, X_val, y_val, X_combo, y_combo, X_test, y_test = xys

    gini_tree = DecisionTreeClassifier(criterion = "gini", max_depth=5, random_state=1)
    entropy_tree = DecisionTreeClassifier(criterion="entropy", max_depth=5, random_state=1)

    gini_tree.fit(X_train, y_train)
    entropy_tree.fit(X_train, y_train)

    gini_predict = gini_tree.predict_proba(X_val)
    entropy_predict = entropy_tree.predict_proba(X_val)

    gini_cross = log_loss(y_val, gini_predict)
    entropy_cross = log_loss(y_val, entropy_predict)

    print("Cross Entropy for Gini: ", gini_cross)
    print("Cross Entropy for Info Gain: ",entropy_cross)

    if entropy_cross < gini_cross:
        best_model = entropy_tree
        print("Best is Information Gain")
    else:
        best_model = gini_tree
        print("Best is Gini impurity index")

    best_model.fit(X_combo, y_combo)
    testing_predictions = best_model.predict_proba(X_test)
    print("Testing Cross Entropy:",log_loss(y_test, testing_predictions))
    return best_model

def exercise4(xys):
    print("Exercise 4")
    #Get data from CSVs
    X_train, y_train, X_val, y_val, X_combo, y_combo, X_test, y_test = xys

    classifier1 = AdaBoostClassifier(n_estimators=20, random_state=1)
    classifier2 = AdaBoostClassifier(n_estimators=40, random_state=1)
    classifier3 = AdaBoostClassifier(n_estimators=60, random_state=1)

    classifier1.fit(X_train, y_train)
    classifier2.fit(X_train, y_train)
    classifier3.fit(X_train, y_train)

    predict1 = classifier1.predict_proba(X_val)
    predict2 = classifier2.predict_proba(X_val)
    predict3 = classifier3.predict_proba(X_val)

    cross1 = crossEntropy(y_val, predict1)
    cross2 = crossEntropy(y_val, predict2)
    cross3 = crossEntropy(y_val, predict3)

    print("20 Stumps Cross Entropy: ",cross1)
    print("40 Stumps Cross Entropy: ",cross2)
    print("60 Stumps Cross Entropy: ",cross3)

    best_cross = inf
    for cross in (cross1, cross2, cross3):
        if cross < best_cross: best_cross = cross
    
    if best_cross == cross1:
        best_model = classifier1
        print("Best is 20 Boosts")
    if best_cross == cross2:
        best_model = classifier2
        print("Best is 40 Boosts")
    if best_cross == cross3:
        best_model = classifier3
        print("Best is 60 Boosts")
    
    best_model.fit(X_combo, y_combo)
    testing_predictions = best_model.predict_proba(X_test)
    print("Testing Cross Entropy:",crossEntropy(y_test, testing_predictions))
    return best_model

def exercise5(model2, model3, model4, xys):
    print("Exercise 5")

    X_train, y_train, X_val, y_val, X_combo, y_combo, X_test, y_test = xys

    thresholds = np.arange(0.0, 1.001, .001)

    for model, name in (model2, "MLP"), (model3, "Decision Tree"), (model4, "Boost"):
        TPR = []
        FPR = []
        max_youden = -inf
        y = model.predict(X_test)
        p = model.predict_proba(X_test)
        for threshold in thresholds:
            P = determinize(threshold, p)
            cm = confusionMatrix(y_test, P)
            positive = float(cm[1][1] + cm[1][0])
            negative = float(cm[0][1] + cm[0][0])
            tpr = (cm[1][1])/positive if positive > 0 else 0
            fpr = (cm[0][1])/negative if negative > 0 else 0
            TPR.append(tpr)
            FPR.append(fpr)
            youden = tpr - fpr
            if youden > max_youden: 
                max_youden = youden
                max_threshold = threshold
        print(name, "Highest Youden is:", max_youden,"Probability threshold is:",max_threshold)
        print("AUC: ", roc_auc_score(y_test, y))
        pyplot.scatter(FPR, TPR, label = name)

    pyplot.plot([0,1], [0,1], '--', color='0.5')
    pyplot.xlabel("FPR")
    pyplot.ylabel("TPR")
    pyplot.legend()
    pyplot.show()
    
xys = getXYs()
model2 = exercise2(xys)
model3 = exercise3(xys)
model4 = exercise4(xys)
exercise5(model2, model3, model4, xys)
