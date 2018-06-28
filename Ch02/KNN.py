from numpy import *
import operator

# 01 first simple classfier

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return  group,labels


def classfiy0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX,(dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistance = sqDiffMat.sum(axis=1)
    distance = sqDistance**0.5
    sortedDistIndices = distance.argsort()
    classCount = {}
    for i in range(k):
        voteLabel = labels[sortedDistIndices[i]]
        classCount[voteLabel] = classCount.get(voteLabel,0)
    sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]



# hookup data

def file2matrix(filename):
    fr = open(filename)
    arrayOfLines = fr.readlines()
    numberOfLines = len(arrayOfLines)
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    index = 0
    for line in arrayOfLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1

    return returnMat,classLabelVector



def autoNormal(dataSet):
    minValue = dataSet.min(0)
    maxValue = dataSet.max(0)
    ranges = maxValue - minValue
    normalDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normalDataSet = dataSet - tile(minValue,[m,1])
    normalDataSet = normalDataSet/tile(ranges,[m,1])
    return normalDataSet,ranges,minValue


def testHookDatingCLassfier():
    hoRadio = 0.1
    dataSet,datingLabels = file2matrix('Ch02/datingTestSet2.txt')
    normMat,ranges,minVals = autoNormal(dataSet)

    m = normMat.shape[0]
    numOfTestVec = int(m*hoRadio)

    errorCount = 0
    for i in range(numOfTestVec):
        result = classfiy0(normMat[i,:],normMat[numOfTestVec:m,:],datingLabels[numOfTestVec:m],3)
        print("The classfier came back with %d,the real answer is: %d" % (result,datingLabels[i]))
        if result != datingLabels[i]:
            errorCount += 1
    print('the total error rate is: %f' %(errorCount/numOfTestVec))



def predictDatingPerson():
    resultList = ['not at all','in small dose','in large dose']
    percentTats = float(input("percent of time spent playing video games?"))
    ffmiles = float(input('frequent flier miles earned per year?'))
    iceCream = float(input('liters of ice cream consumed per weak?'))

    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals = autoNormal(datingDataMat)
    inArr = array([ffmiles,percentTats,iceCream])
    result = classfiy0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print('you will probably like the person',(resultList[result-1]))


print(' ')
print('*********************  digit recognize  *********************')
print(' ')

# img tranform vector
def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        linestr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(linestr[j])
    return  returnVect

from os import listdir

def handwrittingClassTest():
    hwLabels = []
    traingFileList = listdir('Ch02/trainingDigits')
    m = len(traingFileList)
    trainingMat = zeros((m,1024))

    for i in range(m):
        fileNameStr = traingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('Ch02/trainingDigits/%s' % fileNameStr)

    testFileList = listdir('Ch02/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('Ch02/testDigits/%s' % fileNameStr)

        result = classfiy0(vectorUnderTest,trainingMat,hwLabels,3)
        print('the classfier come back with: %d,the real answer is %d' % (result,classNumStr))
        if result != classNumStr:
            errorCount += 1.0
    print('\nthe totalnumber of errors is :%d' % errorCount)
    print('\nthe total error rate  is :%f' % (errorCount/float(mTest)))

if __name__ == '__main__':
    print(' ')
    print('*********************  simple classfier  *********************')
    print(' ')

    group,labels = createDataSet()
    print(group,labels)

    a = classfiy0([0,0],group,labels,3)
    print(a)

    print(' ')
    print('*********************  hook up data classfier *********************')
    print(' ')
    datingDataMat,datingLabels = file2matrix('Ch02/datingTestSet2.txt')
    print(datingDataMat,datingLabels)

    # plot the data
    import matplotlib.pyplot as plt

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(datingDataMat[:,1],datingDataMat[:,2],15.0*array(datingLabels),15.0*array(datingLabels))
    # plt.show()


    norMat,ranges,minValues = autoNormal(datingDataMat)
    print(norMat)

    # testHookDatingCLassfier()
    # predictDatingPerson()

    print(' ')
    print('*********************  digit recognize   *********************')
    print(' ')
    testVector = img2vector('Ch02/testDigits/0_13.txt')
    print(testVector[0,0:31])

    handwrittingClassTest()