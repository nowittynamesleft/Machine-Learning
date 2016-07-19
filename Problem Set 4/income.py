#Meet Barot
#Talked to Zi Gu
#mmb557
#Predicting income based on age, workclass, education, marital-status, occupation, relationship, race, sex, capital-gain,
#captial-loss, hours-per-week, native-country.

#3. c. The best choices for the parameters were max_depth = 10 and min_samples_leaf = 29.
#3. d. The test set accuracy score was 0.858915300043.

import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from collections import Counter
import re
import os
from sklearn.externals.six import StringIO

trainingSet = open("./ps4_data/adult_train.txt").read().split("\n")
testSet = open("./ps4_data/adult_test.txt").read().split("\n")
#turning data into vectors
featureVectors = [[]]
for i in range(0,len(trainingSet)-1):
    values = [value.strip() for value in trainingSet[i].split(",")]
    featureVectors.append(values)

testfeatureVectors = [[]]
for i in range(0,len(testSet)-1):
    values = [value.strip() for value in testSet[i].split(",")]
    testfeatureVectors.append(values)

agesum = 0
capitalgainsum = 0
capitallosssum = 0
hoursperweeksum = 0

modes = []

for i in range(0,12):
    modelist = Counter([vector[i] for vector in featureVectors[1:]])
    modes.append(modelist.most_common(1)[0][0])

for i in range(1,len(featureVectors)):
    if(featureVectors[i][0] != '?'):
        agesum += int(featureVectors[i][0])
    if(featureVectors[i][8] != '?'):
        capitalgainsum += int(featureVectors[i][8])
    if(featureVectors[i][9] != '?'):
        capitallosssum += int(featureVectors[i][9])
    if(featureVectors[i][10] != '?'):
        hoursperweeksum += int(featureVectors[i][10])

avgage = agesum/len(featureVectors)
avgcapgain = capitalgainsum/len(featureVectors)
avgcaploss = capitallosssum/len(featureVectors)
avghoursperweek = hoursperweeksum/len(featureVectors)

replacements = modes
replacements[0] = avgage
replacements[8] = avgcapgain
replacements[9] = avgcaploss
replacements[10] = avghoursperweek

#replacing missing values with average values if continuous, mode if categorical

for i in range(1,len(featureVectors)):
    for j in range(0,12):
        if(featureVectors[i][j] == '?'):
            featureVectors[i][j] = replacements[j]

for i in range(1, len(testfeatureVectors)):
    for j in range(0,12):
        if(testfeatureVectors[i][j] == '?'):
            testfeatureVectors[i][j] = replacements[j]

features = open("./ps4_data/features.txt").read().split("\n")
featureVals = [[value.strip() for value in re.split(r'[,:.]',line)[1:] if value] for line in features[1:]]
newFeatures = [[]]
newtestFeatures = [[]]

#making categorical features into sets of binary features

for i in range(1,len(featureVectors)):
    newValues = []
    for j in range(0,len(featureVals)):
        if featureVectors[i][j].isnumeric():
            newValues.append(featureVectors[i][j])
        else:
            for value in featureVals[j]:
                if featureVectors[i][j+1] == value:
                    newValues.append(1)
                else:
                    newValues.append(0)
    newFeatures.append(newValues)

for i in range(1,len(testfeatureVectors)):
    newValues = []
    for j in range(0,len(featureVals)):
        if testfeatureVectors[i][j].isnumeric():
            newValues.append(testfeatureVectors[i][j])
        else:
            for value in featureVals[j]:
                if testfeatureVectors[i][j+1] == value:
                    newValues.append(1)
                else:
                    newValues.append(0)
    newtestFeatures.append(newValues)


#picking first 70% of data for training set, rest as validate set

divider = int(0.7*len(newFeatures))
trainingFeatures = newFeatures[1:divider+1]
validateFeatures = newFeatures[divider+2:]
classifications = [vector[-1] for vector in featureVectors[1:]]

testclassifications = [vector[-1].replace(".","") for vector in testfeatureVectors[1:]]
trainingClasses = classifications[:divider]
validateClasses = classifications[divider+1:]

max_depths = [i for i in range(1,31)]
min_samples_leafs = [i for i in range(1,51)]
def finderror(predictions, actuals):
    error = 0
    for i in range(0,len(predictions)):
        if predictions[i] != actuals[i]:
            error += 1
    return error/len(predictions)

'''
maxscore = 0
currscore = 0
bestdepth = -1
trainingscores = []
validatescores = []
for value in max_depths:
    clf = tree.DecisionTreeClassifier(max_depth=value)
    clf.fit(trainingFeatures,trainingClasses)
    trainingscores.append(clf.score(trainingFeatures,trainingClasses))
    validatescores.append(clf.score(validateFeatures,validateClasses))
    currscore = validatescores[-1]
    if currscore > maxscore:
        maxscore = currscore
        bestdepth = value

plt.plot(max_depths,trainingscores)
plt.plot(max_depths,validatescores)
plt.show()
print(bestdepth,maxscore)
#10 is the best depth

maxscore = 0
currscore = 0
bestsamples = -1
trainingscores = []
validatescores = []
for value in min_samples_leafs:
    clf = tree.DecisionTreeClassifier(min_samples_leaf=value)
    clf.fit(trainingFeatures,trainingClasses)
    trainingscores.append(clf.score(trainingFeatures,trainingClasses))
    validatescores.append(clf.score(validateFeatures,validateClasses))
    currscore =validatescores[-1]
    if currscore > maxscore:
        maxscore = currscore
        bestsamples = value

plt.plot(min_samples_leafs,trainingscores)
plt.plot(min_samples_leafs,validatescores)
plt.show()
print(bestsamples, maxscore)
# 29 is the best min_sample_leaf value
'''
#using best parameters to train on the whole training set

clf = tree.DecisionTreeClassifier(max_depth=10,min_samples_leaf=29)
clf.fit(newFeatures[1:],classifications)
print(clf.score(newtestFeatures[1:],testclassifications))

with open('incomestree.dot', 'w') as f:
    f = tree.export_graphviz(clf, out_file=f, max_depth=2, filled=True, rounded=True)

