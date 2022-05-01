from math import inf, log
import math
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
import numpy as np
import itertools

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

def getXY(data, num_features):
    X = data[:,0:num_features-1]
    y = data[:,num_features]
    return X, y

def printConfusionMatrix(cm):
    print("True Negative:",cm[0][0])
    print("False Positive:",cm[0][1])
    print("False Negative:",cm[1][0])
    print("True Positive:", cm[1][1])

#Exercise 4
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

def crossEntropy(h, c):
    cross_entropy = 0
    for i in range(len(h)):
        cross_entropy += c[i] * log(h[i][0]) + (1 - c[i]) * log(1 - h[i][0])
    return cross_entropy / -len(h)

def loadData(training_name, validation_name, testing_name):
    #Get data from CSVs
    training_data = np.loadtxt(training_name,skiprows=1,delimiter=",")
    validation_data = np.loadtxt(validation_name,skiprows=1,delimiter=",")
    testing_data = np.loadtxt(testing_name,skiprows=1,delimiter=",")
    return training_data, validation_data, testing_data

def exercise2():
    #Get data from CSVs
    stab_training_data, stab_validation_data, stab_testing_data = loadData("StabTraining.csv","StabValidation.csv","StabTesting.csv")
    X, y = getXY(stab_training_data, 11)
    classifier1 = MLPClassifier(hidden_layer_sizes=(20,))
    classifier2 = MLPClassifier(hidden_layer_sizes=(10,10))
    classifier1.fit(X, y)
    classifier2.fit(X, y)
    X, y = getXY(stab_validation_data, 11)
    predict1 = classifier1.predict_proba(X)
    predict2 = classifier2.predict_proba(X)

    cross1 = crossEntropy(predict1, y)
    cross2 = crossEntropy(predict2, y)

    print(cross1)
    print(cross2)

exercise2()