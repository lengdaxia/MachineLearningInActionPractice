import numpy as np
import math


print(' ')
print('*********************  simple progress  *********************')
print(' ')

def loadDadaSet():
    dataMat = [];labelMat = []

    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat


def sigmonid(inX):
    return 1.0/(1 + np.exp(-inX))


def gradientAscent(dataMatIn,classLabels):
    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()

    m,n = np.shape(dataMatrix)
    alpha = 0.001
    maxCycle = 500
    weights = np.ones((n,1))

    for k in range(maxCycle):
        h = sigmonid(dataMatrix*weights)
        error = labelMat - h
        weights = weights + alpha*dataMatrix.transpose()*error
    return weights




print(' ')
print('*********************  darw thre fit line  *********************')
print(' ')

import matplotlib.pyplot as plt
def plotBestFit(weights):
    dataMat,labelMat = loadDadaSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]

    xcord1=[];ycord1=[]
    xcord2=[];ycord2=[]

    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i,1]);ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]);ycord2.append(dataArr[i,2])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker='s')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x=np.arange(-3.0,3.0,0.1)
    y=(-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('X1');plt.ylabel('X2')
    plt.show()


print(' ')
print('*********************  random gradient  *********************')
print(' ')

def stocGradientAscent(dataMatrix,classLabels):
    m,n = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones(n)

    for i in range(m):
        h = sigmonid(sum(dataMatrix[i]*weights))
        print(dataMatrix[i]*weights)
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

import random
def stocGradientAscent2(dataMatrix,classLabel,numIter=150):
    m,n = np.shape(dataMatrix)
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4/(1.0 + j + i) + 0.01
            randIndex = int(random.uniform(0,len(dataIndex)))
            h = sigmonid(sum(dataMatrix[randIndex]*weights))
            error = classLabel[randIndex] - h
            weights = weights + alpha*error*dataMatrix[randIndex]
            del (dataIndex[randIndex])
    return weights




print(' ')
print('*********************  horseColic pratice  *********************')
print(' ')


def classify(inX,weights):
    prob = sigmonid(sum(inX*weights))
    if prob > 0.5: return 1.0
    else:return 0.0


def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')

    trainingSet = [];traingLabels = []
    for line in frTrain.readlines():
        currntLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currntLine[i]))
        trainingSet.append(lineArr)
        traingLabels.append(float(currntLine[21]))
    trainWeights = stocGradientAscent2(np.array(trainingSet),traingLabels,500)
    errorcount = 0;numTestVect = 0.0

    for line in frTest.readlines():
        numTestVect += 1.0
        currntLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currntLine[i]))
        if int(classify(np.array(lineArr),trainWeights)) != int(currntLine[21]):
            errorcount += 1
    errorRate = (float(errorcount))/numTestVect
    print('the error rate of this arerage is: %f' % errorRate)
    return errorRate


def multTest(numitera=10):
    numTests = numitera;errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print('after %d iterations the average error rate is : %f' %(numTests,errorSum/(float(numTests))))

if __name__ == '__main__':
    a = sigmonid(0)
    print(a)

    dataArr,labelMat = loadDadaSet()
    w = gradientAscent(dataArr,labelMat)
    print(w)

    # plotBestFit(w.transpose().tolist()[0])


    # w2 = stocGradientAscent(np.array(dataArr),labelMat)
    # print(w2)
    # plotBestFit(w2)

    w3 = stocGradientAscent2(np.array(dataArr), labelMat,numIter=500)
    print(w3)
    # plotBestFit(w3)


    print(' ')
    print('*********************  horse colic pratice  *********************')
    print(' ')

    multTest()