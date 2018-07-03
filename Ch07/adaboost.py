from numpy import *


def loadDataSet():
    datMat = matrix([[1.0,2.1],[2.0,1.1],[1.3,1.0],[1.0,1.0],[2.0,1.0]])
    classLabels = [1.0,1.0,-1.0,-1.0,1.0]
    return datMat,classLabels

print(' ')
print('*********************  decision stump 单层决策树  *********************')
print(' ')

def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    retArr = ones((shape(dataMatrix)[0],1))
    if threshIneq == 'lt':
        retArr[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArr[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArr

def buildStump(dataArr,classLabels,D):
    dataMatrix = mat(dataArr);labelMat = mat(classLabels).T
    m,n = shape(dataMatrix)
    numSteps = 10.0;bestStump = {};bestClassEst = mat(zeros((m,1)))
    minError = inf
    for i in range(n):
        rangeMin = dataMatrix[:,i].min();rangeMax = dataMatrix[:,i].max()
        stepSize = (rangeMax-rangeMin)/numSteps
        for j in range(-1,int(numSteps)+1):
            for inequal in ['lt','gt']:
                threshVal = rangeMin + float(j)*stepSize
                predictVals = stumpClassify(dataMatrix,i,threshVal,inequal)
                errArr = mat(ones((m,1)))
                errArr[predictVals == labelMat] = 0
                weightedError = D.T*errArr
                print('split:dim %d,thresh %.2f, inequal :%s,the weighted error is % .3f'%(i,threshVal,inequal,
                                                                                                 weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClassEst = predictVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return  bestStump,minError,bestClassEst



def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m,1))/m)
    aggClassEst = mat(zeros((m,1)))
    for i in range(numIt):
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)
        print('D',D.T)
        alpha = float(0.5*log((1.0-error)/max(error,1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print('classEst:',classEst.T)
        expon = multiply(-1*alpha*mat(classLabels).T,classEst)
        D = multiply(D,exp(expon))
        D = D/sum(D)
        aggClassEst += alpha*classEst
        print('aggClassEst :',aggClassEst)
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T,ones((m,1)))
        erroeRate = aggErrors.sum() / m
        print('total error:',erroeRate,'\n')
        if erroeRate == 0.0:break
    return weakClassArr



print(' ')
print('*********************  基于 AdaBoost分类  *********************')
print(' ')
def adaClassfify(dataToCLass,classfierArr):
    dataMatrix = mat(dataToCLass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))

    for i in range(len(classfierArr)):
        classEst = stumpClassify(dataMatrix,classfierArr[i]['dim'],classfierArr[i]['thresh'],classfierArr[i]['ineq'])
        aggClassEst += classfierArr[i]['alpha']*classEst
        print(aggClassEst)
    return sign(aggClassEst)



print(' ')
print('*********************  use adaboost on horseColic *********************')
print(' ')

def loadHorseDataSet(filename):
    numFeat = len(open(filename).readline().split('\t'))
    dataMat = [];labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[-1]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat



if __name__ == '__main__':
    dataMat,labelArr = loadDataSet()
    print(dataMat)

    D = mat(ones((5,1))/5)
    print(D)
    bestStump,minError,bestClassEst = buildStump(dataMat,labelArr,D)
    print(bestStump)
    print(minError)
    print(bestClassEst)

    classifierArray = adaBoostTrainDS(dataMat,labelArr,9)

    print(' ')
    print('*********************  print classifierArray  *********************')
    print(' ')

    for classifier in classifierArray:
        print(classifier)

    print(' ')
    print('*********************  基于Adaboost分类  *********************')
    print(' ')
    result = adaClassfify([0,0],classifierArray)
    print(result)

    result = adaClassfify([[5,5],[0, 0]], classifierArray)
    print(result)

    print(' ')
    print('*********************  use adaboost on horseColic *********************')
    print(' ')
    dataArr2,labelArr2 = loadHorseDataSet('horseColicTraining2.txt')
    classifierArray2 = adaBoostTrainDS(dataArr2,labelArr2,10)

    testArr,testLabelArr = loadHorseDataSet('horseColicTest2.txt')
    prediction10 = adaClassfify(testArr,classifierArray2)
    errArr = mat(ones((67,1)))
    errors = errArr[prediction10 != mat(testLabelArr).T].sum()
    print(errors)


