#Meet Barot
#mmb557
#Implementing kernalized SVM


#2.c. Each lambda seemed to produce the same cross-validation error. There w
#2.d. The score the default svc had was 0.000999
#2.e. The 5 fold cross validation error for the svc was 0.01549. The 10 fold error was 0.0069965.
#2.f. When setting C=.5 and gamma=.01, the 10 fold error became 0.00549. The test error was still 0.000999.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import KFold
from sklearn.svm import SVC

trainingSet = open("./ps3_data/mnist_train.txt").read().split("\n")

#transformation into feature vectors
featureVectors = np.zeros((len(trainingSet),784))
classes = [0 for i in range(0,len(trainingSet))]

for i in range(1,len(trainingSet)-1):
    vector = trainingSet[i].split(",")
    classes[i] = vector[0]
    normalizedVector = [(2*float(value)/255) - 1 for value in vector[1:785]]
    featureVectors[i] = normalizedVector



def predict(weights, featureVector):
    dotProduct = np.dot(weights,featureVector)
    if(dotProduct >= 0):
        return 1
    else:
        return -1

def pegasos_svm_train(data, lamb, classifications, digit):
    t = 0
    w = np.zeros(784)
    objectiveVals = []
    k = 0
    count = 0
    hingeLossSum = 0
    trainingErrorSum = 0.0
    supportCountSum = 0
    innerClasses = [0 for i in range(0,len(classifications))]
    #after each pass through data, evaluate svm objective 
    for i in range(0,20): #20 passes through data
        sigma = 0
        for j in range(0,len(data)):
            t = t + 1
            stepsize = 1/(t*lamb)
            prediction = predict(w,data[j])
            dot = np.dot(w,data[j])
            if(int(classifications[j]) == digit):
                innerClasses[j] = 1
            else:
                innerClasses[j] = -1
            if innerClasses[j]*dot <= 1: #there's a mistake
                w = (1-stepsize*lamb)*w + stepsize*innerClasses[j]*data[j]
                k += 1
                count += 1
                supportCountSum += 1
            else:
                w = (1-stepsize*lamb)*w
                count += 1
            sigma += max(0,1-innerClasses[j]*dot)
        #evaluate svm objective: store value of f(wt) (lambda/2)*||w||^2 + (1/|data|)*sum(1-->|data|)(max(0, 1 - yi(wi.xi)))
        objectiveVals.append((lamb/2)*np.linalg.norm(w)**2 + (1/len(data))*sigma)
        hingeLossSum += (1/len(data))*sigma
        trainingErrorSum += float(k)/float(count)
    avgTrainingError = trainingErrorSum/20.0
    avgHingeLoss = hingeLossSum/20.0
    avgSupportCount = float(supportCountSum)/20.0
    returnVal = [w, objectiveVals, avgTrainingError, avgHingeLoss,avgSupportCount]
    return returnVal

#multi-class prediction using one vs all classification

def multi_class_digit_training(featVs,classifications,lamb):
    weights = []
    for i in range(0,10):
        result = pegasos_svm_train(featVs, lamb, classifications, i)
        weights.append(result[0])
    return weights;

def predictDigit(featVs, weightVList, index):
    highestScoringDigit = -1
    highscore = -1000
    for i in range(0,len(weightVList)):
        score = scorer(weightVList[i],featVs[index])
        if(score > highscore):
            highscore = score
            highestScoringDigit = i
    return highestScoringDigit

def scorer(weights,featVs):
    return np.dot(weights,featVs)

#k-fold cross-validation

def cross_validation(featVs,classifications,lamb):
    kf = KFold(len(classifications),n_folds=5)
    errors = []
    for train, test in kf:
        weights = multi_class_digit_training(featVs[train],classifications,lamb)
        error = 0
        for index in test:
            prediction = predictDigit(featVs,weights,index)
            if(prediction != classifications[index]):
                error += 1
        errors.append(float(error)/len(featVs))
    return min(errors)

def plot_cross_validation(featVs,classifications):
    for i in range(-5,1):
        print(cross_validation(featVs,classifications,2**i))



clf = SVC(C=.5, gamma=.01)
clf.fit(featureVectors,classes)
testSet = open("./ps3_data/mnist_test.txt").read().split("\n")
testFeatureVectors = np.zeros((len(testSet),784))
testClasses = [0 for i in range(0,len(testSet))]

for i in range(0,len(testFeatureVectors)):
    testerror = 0
    if(clf.predict(testFeatureVectors[i]) != testClasses[i]):
        testerror += 1

def svc_cross_validation(featVs,classifications):
    kf = KFold(len(classifications),n_folds=10)
    classifications = np.array((classifications))
    errors = []
    for train, test in kf:
        clf.fit(featVs[train],classifications[train])
        error = 0
        for index in test:
            prediction = clf.predict(featVs[index])
            if(prediction != classifications[index]):
                error += 1
        errors.append(float(error)/len(featVs))
    return min(errors)

print("Cross-validation errors: ", svc_cross_validation(featureVectors,classes))
print("Test error: ", float(testerror)/len(testFeatureVectors))
