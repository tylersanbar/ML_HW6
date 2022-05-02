from math import inf, log
import math
import numpy as np
import itertools

from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

from matplotlib import pyplot

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def mse(y, h):
    sum = 0
    for i in range(len(y)):
        sum += (y[i] - h[i]) ** 2
    mse = sum / len(y)
    return mse

def lambdas2alpha(params):
    a_l1 = []
    for i in range(len(params)):
        if params[i][0] == 0 or params[i][0] > params[i][1]: a_l1.append(None)
        else: 
            alpha = params[i][0] + params[i][1]
            l1 = params[i][0]/alpha
            a_l1.append([alpha,l1])
    return a_l1

def bestMSE(X, y, models):
    MSE = []
    hypothesis = []
    smallestError = inf
    bestIndex = None
    for i in range(len(models)):
        model = models[i]
        if model is not None:
            h = model.predict(X)
            hypothesis.append(h)
            error = mse(y, h)
            MSE.append(error)
            if error < smallestError: 
                smallestError = error
                bestIndex = i
        else:
            MSE.append(None)
            hypothesis.append(None)
    return bestIndex, MSE, hypothesis  

def trainModels(X, y, a_l1):
    models = []
    for i in range(len(a_l1)):
        if a_l1[i] is not None:
            alpha = a_l1[i][0]
            l1_ratio = a_l1[i][1]
            model = ElasticNet(alpha = alpha, l1_ratio=l1_ratio, tol = .1)
            model.fit(X, y)
            models.append(model)
        else: models.append(None)
    return models

def printMSE(params, MSE):
    for i in range(len(MSE)):
        if MSE[i] is not None: print("Lambda1:",params[i][0],"Lambda2:",params[i][1],"MSE",MSE[i])

def printBestMSE(params, MSE, bestIndex):
    print("Best - Lambda1:",params[bestIndex][0],"Lambda2:",params[bestIndex][1],"MSE:",MSE[bestIndex])

def printConfusionMatrix(cm):
    print("True Negative:",cm[0][0])
    print("False Positive:",cm[0][1])
    print("False Negative:",cm[1][0])
    print("True Positive:", cm[1][1])

def loss(y, h):
    if y == h: return 1
    else: return 0

def emprisk(y, h):
    sum = 0
    for i in range(len(y)):
        sum += loss(y[i], h[i])
    return sum / len(y)

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

def exercise2():
    print("Exercise 2")
    #Get data from CSVs
    training_data, validation_data, testing_data = loadData("StabTraining.csv","StabValidation.csv","StabTesting.csv")
    combined_data = np.append(training_data, validation_data, axis = 0)

    X_train, y_train = getXY(training_data,11)
    X_val, y_val = getXY(validation_data,11)
    X_combo, y_combo = getXY(combined_data,11)
    X_test, y_test = getXY(testing_data,11)

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

def exercise3():
    print("Exercise 3")
    #Get data from CSVs
    training_data, validation_data, testing_data = loadData("StabTraining.csv","StabValidation.csv","StabTesting.csv")
    combined_data = np.append(training_data, validation_data, axis = 0)

    X_train, y_train = getXY(training_data,11)
    X_val, y_val = getXY(validation_data,11)
    X_combo, y_combo = getXY(combined_data,11)
    X_test, y_test = getXY(testing_data,11)

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

def exercise4():
    print("Exercise 4")
    #Get data from CSVs
    training_data, validation_data, testing_data = loadData("StabTraining.csv","StabValidation.csv","StabTesting.csv")
    combined_data = np.append(training_data, validation_data, axis = 0)

    X_train, y_train = getXY(training_data,11)
    X_val, y_val = getXY(validation_data,11)
    X_combo, y_combo = getXY(combined_data,11)
    X_test, y_test = getXY(testing_data,11)

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

def exercise5(model2, model3, model4):
    print("Exercise 5")

    #Get data from CSVs
    training_data, validation_data, testing_data = loadData("StabTraining.csv","StabValidation.csv","StabTesting.csv")
    combined_data = np.append(training_data, validation_data, axis = 0)

    X_train, y_train = getXY(training_data,11)
    X_val, y_val = getXY(validation_data,11)
    X_combo, y_combo = getXY(combined_data,11)
    X_test, y_test = getXY(testing_data,11)

    thresholds = np.arange(0.0, 1.001, .001)

    for model in model2, model3, model4:
        TPR = []
        FPR = []
        y = model.predict(X_test)
        p = model.predict_proba(X_test)
        for threshold in thresholds:
            P = determinize(threshold, p)
            cm = confusionMatrix(y, P)
            positive = float(cm[1][1] + cm[1][0])
            negative = float(cm[0][1] + cm[0][0])
            tpr = (cm[1][1])/positive if positive > 0 else 0
            fpr = (cm[0][1])/negative if negative > 0 else 0
            TPR.append(tpr)
            FPR.append(fpr)
        
        pyplot.scatter(TPR, FPR)
    pyplot.show()

model2 = exercise2()
model3 = exercise3()
model4 = exercise4()
exercise5(model2, model3, model4)
