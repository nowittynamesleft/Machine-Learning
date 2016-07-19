#Talked to Zi Gu

#Implementing SVM Pegasos algorithm

import numpy as np
import matplotlib.pyplot as plt
trainingEmails = open("/Users/meetbarot/Desktop/School Things/Machine Learning/Problem Set 1/spam_train.txt").read().split("\n")
wordset = {}
train = trainingEmails[:4000]
#train = ["0 hello my name is bob", "1 buy this new tv", "1 buy this new ipod", "0 i love you"]
validation = trainingEmails[4000:]


spamCount = 0
notSpamCount = 0
classifications = []
wordCounts = {}

for email in train:
    words = email.split()
    wordOccurences = set(words[1:])
    classifications.append(words[0]) 
    for word2 in wordOccurences:
        if word2 in wordCounts:
            wordCounts[word2] += 1
        else:
            wordCounts[word2] = 1

common = []
for word in wordCounts:
    if wordCounts[word] >= 30:
        common.append(word)

common.sort()
commonIndices = {}
count = 0
for word in common:
    commonIndices[word] = count
    count += 1

emailFeatures = np.zeros((len(train), len(common))) #so the entry for 2,4
        #is 1 if the third email has the fifth word in the common word list
        #is 0 if the third email does not have the fifth word in the common word list
emailFeaturesValid = np.zeros((len(validation), len(common)))

for i in range(0,len(train)):
    for word in train[i].split():
        if word in commonIndices:
            emailFeatures[i,commonIndices[word]] = 1

validclass = []

for email in validation:
    words = email.split()
    if(len(words) > 0):
        validclass.append(words[0])

for i in range(0,len(validation)):
    for word in validation[i].split():
        if word in commonIndices:
            emailFeaturesValid[i,commonIndices[word]] = 1


np.set_printoptions(threshold=np.nan)

#at this point, we have a matrix of feature vectors for each email. emailFeatures stores that matrix.

#Spam: 1 Not Spam: -1
#Words
#Present: 1 Not Present: 0
#for each email:
    #predict if it's spam based on the weights of all the features
        #int sum = 0;
        #for each feature:
            #sum += Weight*(Present/NotPresent)
    #If the prediction is correct do nothing: if (prediction sum > 0 && actual > 0) || (prediction sum <= 0 && actual <= 0)
    #while the prediction is mistaken, modify the weight vector by adding (Spam or NotSpam)*(Present or NotPresent) to it
        #for each weight w:
            #w += (spam or notspam)*(present or notpresent)

mostFrequentClass = max(spamCount, notSpamCount)

def perceptron_train(data, maxpasses):
    passes = 0
    w = np.zeros(len(common))
    error = -1
    k = 0
    while error != 0 and passes <= maxpasses:
        error = 0
        for i in range(0,np.size(data,0)):
            prediction = predict(w,data[i])
            if(classifications[i] == '0'):
                classifications[i] = -1
            elif(classifications[i] == '1'):
                classifications[i] = 1
            if classifications[i] != prediction: #there's a mistake
                error += 1                          #so it repeats
                w = w + classifications[i]*data[i]
                k += 1
        passes += 1
    returnVal = [w, k, passes]
    return returnVal

def printEmailFeatures(data):
    for i in range(0,np.size(data,0)):
        print(data[i])

def perceptron_test(w, data, classes):
    errorCount = 0
    count = 0
    for i in range(0,np.size(data,0)-1):
        prediction = predict(w,data[i])
        if(classes[i] == '0'):
            classes[i] = -1
        elif (classes[i] == '1'):
            classes[i] = 1
        if prediction != classes[i]:
            errorCount += 1
        count += 1
    error = float(errorCount)/float(count)
    return error

def predict(weights, featureVector):
    dotProduct = np.dot(weights,featureVector)
    if(dotProduct >= 0):
        return 1
    else:
        return -1

def printPredictiveWeights(weights, vocab):
    dictionary = dict(zip(vocab, weights))
    sortedWeights = sorted(dictionary, key=dictionary.__getitem__)
    print(sortedWeights[-15:])
    print(sortedWeights[:15])

def averaged_train(data, maxpasses):
    passes = 0
    w = np.zeros(len(common))
    weightSum = np.zeros(len(common))
    error = -1
    k = 0
    while error != 0 and passes <= maxpasses:
        error = 0
        for i in range(0,np.size(data,0)):
            prediction = predict(w,data[i])
            if(classifications[i] == '0'):
                classifications[i] = -1
            elif(classifications[i] == '1'):
                classifications[i] = 1
            if classifications[i] != prediction: #there's a mistake
                error += 1                          #so it repeats
                w = w + classifications[i]*data[i]
                k += 1
            weightSum += w
        passes += 1
    averagedw = weightSum/np.size(data,0)
    returnVal = [averagedw, k, passes]
    return returnVal

def pegasos_svm_train(data, lamb):
    t = 0
    w = np.zeros(len(common))
    objectiveVals = []
    k = 0
    count = 0
    hingeLossSum = 0
    trainingErrorSum = 0.0
    supportCountSum = 0
    #after each pass through data, evaluate svm objective 
    for i in range(0,20): #20 passes through data
        sigma = 0
        for j in range(0,len(data)):
            t = t + 1
            stepsize = 1/(t*lamb)
            prediction = predict(w,data[j])
            dot = np.dot(w,data[j])
            if(classifications[j] == '0'):
                classifications[j] = -1
            elif(classifications[j] == '1'):
                classifications[j] = 1
            if classifications[j]*dot <= 1: #there's a mistake
                w = (1-stepsize*lamb)*w + stepsize*classifications[j]*data[j]
                k += 1
                count += 1
                supportCountSum += 1
            else:
                w = (1-stepsize*lamb)*w
                count += 1
            sigma += max(0,1-classifications[j]*dot)
        #evaluate svm objective: store value of f(wt) (lambda/2)*||w||^2 + (1/|data|)*sum(1-->|data|)(max(0, 1 - yi(wi.xi)))
        objectiveVals.append((lamb/2)*np.linalg.norm(w)**2 + (1/len(data))*sigma)
        hingeLossSum += (1/len(data))*sigma
        trainingErrorSum += float(k)/float(count)
    avgTrainingError = trainingErrorSum/20.0
    avgHingeLoss = hingeLossSum/20.0
    avgSupportCount = float(supportCountSum)/20.0
    returnVal = [w, objectiveVals, avgTrainingError, avgHingeLoss,avgSupportCount]
    return returnVal

def pegasos_svm_test(w, data, classes):
    errorCount = 0
    count = 0
    for i in range(0,np.size(data,0)-1):
        prediction = predict(w,data[i])
        if(classes[i] == '0'):
            classes[i] = -1
        elif (classes[i] == '1'):
            classes[i] = 1
        if prediction != classes[i]:
            errorCount += 1
        count += 1
    error = float(errorCount)/float(count)
    return error


#Using all the training data, and testing the test set:
testEmails = open("/Users/meetbarot/Desktop/School Things/Machine Learning/Problem Set 1/spam_test.txt").read().split("\n")
train = trainingEmails[0:5000] 
testEmails = testEmails[0:1000]
spamCount = 0
notSpamCount = 0
classifications = []
wordCounts = {}

for email in train:
    words = email.split()
    wordOccurences = set(words[1:])
    if(len(words) > 0):
        classifications.append(words[0]) 
    for word2 in wordOccurences:
        if word2 in wordCounts:
            wordCounts[word2] += 1
        else:
            wordCounts[word2] = 1

common = []
for word in wordCounts:
    if wordCounts[word] >= 30:
        common.append(word)

common.sort()
commonIndices = {}
count = 0
for word in common:
    commonIndices[word] = count
    count += 1

emailFeatures = np.zeros((len(train), len(common))) #so the entry for 2,4
        #is 1 if the third email has the fifth word in the common word list
        #is 0 if the third email does not have the fifth word in the common word list
for i in range(0,len(train)):
    for word in train[i].split():
        if word in commonIndices:
            emailFeatures[i,commonIndices[word]] = 1

emailFeaturesTest = np.zeros((len(testEmails),len(common)))
testclass = []

for email in testEmails:
    words = email.split()
    if(len(words) > 0):
        testclass.append(words[0])

for i in range(0,len(testEmails)):
    for word in testEmails[i].split():
        if word in commonIndices:
            emailFeaturesTest[i,commonIndices[word]] = 1

finaldata = pegasos_svm_train(emailFeatures,2**-5)


print("Pegasos training test set error for lambda 2^-5:", pegasos_svm_test(finaldata[0], emailFeaturesTest,testclass))


lambLogVals = []
errorVals = []
avgTrainVals = []
avgHingeVals = []
avgSVCounts = []
def test_vals_of_lamb():
    for i in range(-9,2):
        lamb = 2**i
        lambLogVals.append(i)
        finaldata = pegasos_svm_train(emailFeatures,lamb)
        avgTrainVals.append(finaldata[2])
        avgHingeVals.append(finaldata[3])
        avgSVCounts.append(finaldata[4])
        error = pegasos_svm_test(finaldata[0], emailFeaturesTest,testclass)
        errorVals.append(error)
        print("Pegasos training test set error for lambda ", lamb, " : ", error)

test_vals_of_lamb()

#plt.plot(lambLogVals,errorVals)
#plt.show()
#plt.plot(lambLogVals,avgHingeVals)
#plt.show()
#plt.plot(lambLogVals, avgSVCounts)
#plt.show()
