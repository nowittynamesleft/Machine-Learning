
def randIndex(truth, predicted):
	"""
	The function is to measure similarity between two label assignments
	truth: ground truth labels for the dataset (1 x 1496)
	predicted: predicted labels (1 x 1496)
	"""
	if len(truth) != len(predicted):
		print("different sizes of the label assignments")
		return -1
	elif (len(truth) == 1):
		return 1
	sizeLabel = len(truth)
	agree_same = 0
	disagree_same = 0
	count = 0
	for i in range(sizeLabel-1):
		for j in range(i+1,sizeLabel):
			if ((truth[i] == truth[j]) and (predicted[i] == predicted[j])):
				agree_same += 1
			elif ((truth[i] != truth[j]) and (predicted[i] != predicted[j])):
				disagree_same +=1
			count += 1
	return (agree_same+disagree_same)/float(count)

# Code Sample
import scipy.cluster.hierarchy as sch
import numpy as np
import pylab as pl
import sklearn.cluster as sk
from random import randint
from scipy import stats
# Plot dendogram and cut the tree to find resulting clusters
'''
fig = pl.figure() 
data = np.array([[1,2,3],[1,1,1],[5,5,5]])
datalable = ['first','second','third']
hClsMat = sch.linkage(data, method='complete') # Complete clustering
sch.dendrogram(hClsMat, labels= datalable, leaf_rotation = 45)
fig.show()
resultingClusters = sch.fcluster(hClsMat,t= 3, criterion = 'distance')
print(resultingClusters)
'''

# Your code starts from here ....
dataGrains = open("./dataCereal-grains-pasta.txt").read().split('\n')
dataFats = open("./dataFats-oils.txt").read().split('\n')
dataFish = open("./dataFinfish-shellfish.txt").read().split('\n')
dataVeg = open("./dataVegetables.txt").read().split('\n')


# 1. 
# Scaling min max 
# STUDENT CODE TODO
'''
features = np.array((1500,151),dtype=object)
f = 0
for i in range (0, len(dataGrains)-1):
    for j in range(0,len(dataGrains[i].split('^')[:151])):
        features[f][j] = dataGrains[i].split('^')[j]
    f += 1
for i in range (0, len(dataFats)-1):
    for j in range(0,len(dataFats[i].split('^')[:151])):
        features[f][j] = dataFats[i].split('^')[j]
    f += 1
for i in range (0, len(dataFish)-1):
    for j in range(0,len(dataFish[i].split('^')[:151])):
        features[i][j] = dataFish[i].split('^')[j]
    f += 1
for i in range (0, len(dataVeg)-1):
    for j in range(0,len(dataVeg[i].split('^')[:151])):
        features[i][j] = dataVeg[i].split('^')[j]
    f += 1
'''
features = [[]]
for i in range (0, len(dataGrains)-1):
    features.append(dataGrains[i].split('^')[1:151])
for i in range (0, len(dataFats)-1):
    features.append(dataFats[i].split('^')[1:151])
for i in range (0, len(dataFish)-1):
    features.append(dataFish[i].split('^')[1:151])
for i in range (0, len(dataVeg)-1):
    features.append(dataVeg[i].split('^')[1:151])

foodNames = []
for grain in dataGrains:
    foodNames.append(grain.split('^')[0])
foodNames.pop(-1)
for fat in dataFats:
    foodNames.append(fat.split('^')[0])
foodNames.pop(-1)
for fish in dataFish:
    foodNames.append(fish.split('^')[0])
foodNames.pop(-1)
for veg in dataVeg:
    foodNames.append(veg.split('^')[0])
foodNames.pop(-1)

#features = np.delete(np.array(features),[0]) #first element was empty so deleted it
features.pop(0)
#now we have all the feature vectors in one feature vector list
#need to find the mins and maxes of each row(i.e., each type of nutrient)


minxjs = np.full(151,100000, dtype=float)
maxxjs = np.full(151,-100000, dtype=float)

for j in range (0,len(features[0])):
    for i in range (0, len(features)):
        if(minxjs[j] > float(features[i][j])):
            minxjs[j] = float(features[i][j])
        if(maxxjs[j] < float(features[i][j])):
            maxxjs[j] = float(features[i][j])

#now we have a list of mins and maxes for each nutrient type
#normalizing these:
# (xij - minx[j])/(maxx[j]-minx[j])

def normalizeVals(feats, maxes, mins):
    for vector in feats:
        for j in range(0,len(vector)):
            if(maxes[j] - mins[j] == 0):
                vector[j] = 0.0
            else:
                vector[j] = (float(vector[j]) - mins[j])/(maxes[j] - mins[j])

normalizeVals(features, maxxjs, minxjs)

#now we have normalized the data.

# 2. 
# K-means http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
# STUDENT CODE TODO

k = sk.KMeans(n_clusters=4)
k.fit(features)
np.set_printoptions(threshold=np.nan)
clusteringlabels = k.labels_


# 3.
# Compute Rand Index
# STUDENT CODE TODO

groundtruths = []
for i in range (0, len(dataGrains)-1):
    groundtruths.append(0)
for i in range (0, len(dataFats)-1):
    groundtruths.append(1)
for i in range (0, len(dataFish)-1):
    groundtruths.append(2)
for i in range (0, len(dataVeg)-1):
    groundtruths.append(3)

randoms = []
for i in range (0, len(groundtruths)):
    randoms.append(randint(0,3))


print("Computing Rand index: ");
print("Random permutation of truth labels vs. ground truths: ",randIndex(groundtruths, randoms))
print("Clustering labels from k-means vs. ground truths: ",randIndex(groundtruths, clusteringlabels))



# 4.
# Examining K-mean objective
# STUDENT CODE TODO
'''
k = sk.KMeans(n_clusters=4,n_init=1)
for i in range(1,21):
    k.fit(features)
    print("Run ",i," objective function: ", k.inertia_)
    print("Rand index values (comparing to ground truths): ", randIndex(groundtruths,k.labels_))
'''
#Distinct values observed:          Rand indices:
#660.028811463 (Minimum)            0.8355445066442508
#779.193687243                      0.8000250388997192
#789.388239179                      0.8000983671060398
#779.193687243                      0.8000250388997192                      
#19.829155749                       0.7998524493409404

# 5. 
# Dendogram plot
# Dendogram - http://docs.scipy.org/doc/scipy-0.13.0/reference/generated/scipy.cluster.hierarchy.dendrogram.html
# Linkage - http://docs.scipy.org/doc/scipy-0.13.0/reference/generated/scipy.cluster.hierarchy.linkage.html
# STUDENT CODE TODO

#randomly selecting 30 items from each food group

randomfoods = [[]]
foodlabels = []
names = []
grounds = []
for i in range(0,30):
    ind = randint(0,len(dataGrains)-1)
    randomfoods.append(features[ind])
    foodlabels.append(clusteringlabels[ind])
    names.append(foodNames[ind])
    grounds.append(0)
for i in range(0,30):
    ind = randint(0,len(dataFats)-1)
    randomfoods.append(features[ind])
    foodlabels.append(clusteringlabels[ind])
    names.append(foodNames[ind])
    grounds.append(1)
for i in range(0,30):
    ind = randint(0,len(dataFish)-1)
    randomfoods.append(features[ind])
    foodlabels.append(clusteringlabels[ind])
    names.append(foodNames[ind])
    grounds.append(2)
for i in range(0,30):
    ind = randint(0,len(dataVeg)-1)
    randomfoods.append(features[ind])
    foodlabels.append(clusteringlabels[ind])
    names.append(foodNames[ind])
    grounds.append(3)
randomfoods.pop(0)

fig = pl.figure() 
data = np.array(randomfoods)
hClsMat = sch.linkage(data, method='complete') # Complete clustering
sch.dendrogram(hClsMat, labels=names, leaf_rotation = 45)
fig.show()
input("Press Enter to continue...")
# 6. 
# Hierarchical clustering
# SciPy's Cluster - http://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fcluster.html#scipy.cluster.hierarchy.fcluster
# STUDENT CODE TODO


resultingClusters = sch.fcluster(hClsMat,t= .01, criterion = 'distance')
print("Hierarchical clustering for t=.01: ", resultingClusters)
print("Rand index computed from this clustering: ",randIndex(grounds, resultingClusters))

#The agglomerative clustering performed very badly (randIndex of 0.2437) because only one cluster was output when t=3.8. However,
#it performed slightly better at exactly t=3.0, giving a randIndex of 0.382
#Given a t=.1, there was a randIndex of 0.754.
#Given a t=.01, there was a randIndex of 0.755, 


# 7. 
# K-means for Sub-cluster 
# STUDENT CODE TODO

grainFeatures = features[:len(dataGrains)-1]
grainNames = []

for grain in dataGrains:
    grainNames.append(grain.split('^')[0])
grainNames.pop(-1)

kvals = [5, 10, 25, 50, 75]
for val in kvals:
    largestclusternames = []
    print('Run for k=', val)
    k = sk.KMeans(n_clusters=val)
    k.fit(grainFeatures)
    clusteringlabels = k.labels_
    mode = stats.mode(clusteringlabels)[0] 
    for i in range(0,len(clusteringlabels)):
        if(clusteringlabels[i] == mode):
            largestclusternames.append(grainNames[i])
    if(len(largestclusternames) > 10):
        for i in range(0,10):
            print("\tRandom item from largest cluster: ", largestclusternames[randint(0,len(largestclusternames)-1)])
    else:
        for name in largestclusternames:
            print("\tItem from largest cluster: ", name)
    print("Size of largest cluster: ", len(largestclusternames))

#size of largest cluster decreases with a larger number of clusters that the function is forced to make
