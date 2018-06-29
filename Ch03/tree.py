from math import log
from machineLearnInAction.Ch03.treePlotter import  *

print(' ')
print('*********************  caculate entropy  *********************')
print(' ')

# take the majority label
import operator

def classify(inputTree,featLabels,testVec):
    firstStr = list(inputTree.keys())[0]
    secondic = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondic.keys():
        if testVec[featIndex] == key:
            if type(secondic[key]).__name__ == 'dict':
                classLabel = classify(secondic[key],featLabels,testVec)
            else:
                classLabel = secondic[key]
    return classLabel

def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    print(classCount)
    print(sortedClassCount)
    return sortedClassCount[0][0]

def calculateShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        shannonEnt -= prob * log(prob,2)
    return shannonEnt

def createDataSet():
    dataSet = [[1,1,'yes'],
               [1,1,'yes'],
               [1,0,'no'],
               [0,1,'no'],
               [0,1,'no']]
    labels = ['no surfacing','flippers']
    return dataSet,labels

# splite the dataset
def splitDataSet(dataSet,asix,value):
    retDataset = []
    for featVec in dataSet:
        if featVec[asix] == value:
            reducedFeatVec = featVec[:asix]
            reducedFeatVec.extend(featVec[asix+1:])
            retDataset.append(reducedFeatVec)
    return retDataset



# choose the best feature
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) -1
    baseEntropy = calculateShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        featureList = [example[i] for example in dataSet]
        print(featureList)
        uniqueVal = set(featureList)
        newEntropy = 0.0
        for value in uniqueVal:
            subDataSet = splitDataSet(dataSet,i,value)
            prob = len(subDataSet)/float(len(dataSet))
            print(prob)
            newEntropy += prob*calculateShannonEnt(subDataSet)
            print(newEntropy)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

# create tree
def createTree(dataSet,labels):
    classList = [a[-1] for a in dataSet]
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueValues = set(featValues)
    for value in uniqueValues:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
    return myTree




def storeTree(storeTree,filename):
    import pickle
    fw = open(filename,'wb')
    pickle.dump(storeTree,fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr = open(filename,'rb')
    return pickle.load(fr)

if __name__ == '__main__':
    myDat,label = createDataSet()
    h = calculateShannonEnt(myDat)
    print(h)

    myDat[0][-1] = 'hello'
    print(calculateShannonEnt(myDat))

    testData,_ = createDataSet()
    a = splitDataSet(testData,0,1)
    print(a)
    b = splitDataSet(a,0,1)
    print(b)

    print(' ')
    print('*********************  best feature  *********************')
    print(' ')

    bestindex = chooseBestFeatureToSplit(testData)
    print(bestindex)

    print(' ')
    print('*********************  create tree  *********************')
    print(' ')

    # test majority votet
    l = [1,1,3,4,4,4,'1','a','b','v','a','a','a','a','t']
    r = majorityCnt(l)
    print(r)
    testTree = createTree(testData,label)
    print(testTree)

    print(' ')
    print('*********************  classify  a vector *********************')
    print(' ')

    myDat2,label2 = createDataSet()

    myTree = retriveTree(0)
    print(myTree)
    print(label2)
    a = classify(testTree,label2,[1,1])
    print(a)

    print(' ')
    print('*********************  store and read tree model  *********************')
    print(' ')

    storeTree(myTree,'storedTreeModel.txt')

    fileTree = grabTree('storedTreeModel.txt')
    print(fileTree)


    print(' ')
    print('*********************  predic lenses  *********************')
    print(' ')

    fr = open('lenses.txt')
    lenses = [inst.strip().split('\t') for inst in fr.readlines()]
    lenseLabels = ['age',
                   'prescript',
                   'astigmatic',
                   'tearRate']
    lensetree = createTree(lenses,lenseLabels)
    createPlot(lensetree)

