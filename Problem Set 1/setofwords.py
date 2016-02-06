#Talked to Zi Gu
import numpy as np
import matplotlib.pyplot as plt
trainingEmails = open("/Users/meetbarot/Desktop/School Things/Machine Learning/Problem Set 1/spam_train.txt").read().split("\n")
wordset = {}
train = trainingEmails[:4000]
#train = ["0 hello my name is bob", "1 buy this new tv", "1 buy this new ipod", "0 i love you"]
validation = trainingEmails[4000:]

#Questions
#1. The validation set is used so that the algorithm can be evaluated for overfitting before using it on the test set.
#4. There are 447 mistakes made before the algorithm terminates. The validation error for the algorithm is 0.02.
#5. The 15 words with the most positive weights are: 
#['major', 'ever', 'deathtospamdeathtospamdeathtospam', 'your', 'present', 'pleas', 'these', 'nbsp', 'click', 'market', 'guarante', 'yourself', 'remov', 'our', 'sight']
# The 15 words with the most negative weights are:
# ['but', 'wrote', 'prefer', 'and', 'reserv', 'i', 'on', 'technolog', 'still', 'instead', 'copyright', 'upgrad', 'recipi', 'sinc', 'url']
#7. The errors were:
# Regular:  0.076 Averaged:  0.083
# Regular:  0.074 Averaged:  0.07
# Regular:  0.041 Averaged:  0.036
# Regular:  0.026 Averaged:  0.027
# Regular:  0.024 Averaged:  0.021
# Regular:  0.02 Averaged:  0.018
# The averaged algorithm does worse for smaller training sets but better for larger ones.
# The number of passes generally increases but, for instance, n = 200, it dips down and then comes back up for n = 400
#10. The final error for the regular algorithm on the test set, using the full training set as data, was 0.01901901901901902.
# The averaged algorithm error was 0.01701701701701702.


#Transforming into feature vectors:
#Need to have a feature vector for each email
#[index of email]{word : 0/1}

#need to split input into seperate emails to evaluate emails
#emails are strings
#need to get the list of the words that appear in at least 30 emails
#need to process each email to make feature vector


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



dat = perceptron_train(emailFeatures, 100)
print(perceptron_test(dat[0],emailFeaturesValid,validclass))
printPredictiveWeights(dat[0],common)
avgdat = averaged_train(emailFeatures, 100)
print(avgdat[0])

n = [100,200,400,800,2000,4000]
averagederrors = []
regularerrors = []
for element in n:
    d = perceptron_train(emailFeatures[:element], 100)
    avgd = averaged_train(emailFeatures[:element], 100)
    regularerrors.append(perceptron_test(d[0],emailFeaturesValid,validclass))
    averagederrors.append(perceptron_test(avgd[0],emailFeaturesValid,validclass))
    print("Regular algorithm error: ", perceptron_test(d[0],emailFeaturesValid,validclass), " Averaged algorithm error: ", perceptron_test(avgd[0],emailFeaturesValid,validclass))
    print("Regular algorithm passes: ", d[2], " Averaged algorithm passes: ", avgd[2])

plt.plot(n,averagederrors)

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

finaldata = perceptron_train(emailFeatures,1000)
finalavgdata = averaged_train(emailFeatures,1000)
print("Testing test set error: ", perceptron_test(finaldata[0], emailFeaturesTest,testclass), " Averaged algorithm error: ", perceptron_test(finalavgdata[0],emailFeaturesTest,testclass))
