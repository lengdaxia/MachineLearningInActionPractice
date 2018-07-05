from numpy import *


class treeNode():
    def __init__(self,feat,val,right,left):
        featureToSpliteOn = feat
        valueOfSplite = val
        rightBranch = right
        leftBranch = left



def loadDataSet(filename):
    dataMat = []
    fr = open(filename)
    for line in fr.readlines():
        curline = line.strip().split('\t')

        fltline = list(map(lambda x:float(x),curline))
        # ll = list(map(float,curline))
        # print(curline,fltline)
        dataMat.append(fltline)
    return dataMat

def binarySplitDataSet(dataSet,feature,value):
    mat0 = dataSet[nonzero(dataSet[:,feature]>value)[0],:]
    mat1 = dataSet[nonzero(dataSet[:,feature]<=value)[0],:]
    return mat0,mat1


def regLeaf(dataset):
    return mean(dataset[:,-1])

def regErr(dataset):
    return var(dataset[:,-1])*shape(dataset)[0]


def chooseBestSplit(dataSet,leafType,errType,ops=(1,4)):
    tols = ops[0];tolN = ops[1]
    if len(set(dataSet[:,-1].T.tolist()[0])) == 1:
        return None,leafType(dataSet)
    m,n = shape(dataSet)
    S = errType(dataSet)
    bestS = inf;bestIndex = 0;bestValue = 0
    for featIndex in range(n-1):
        for splitVal in set(dataSet[:,featIndex].A[0]):
            mat0,mat1 = binarySplitDataSet(dataSet,featIndex,splitVal)
            if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):continue
            newS = errType(mat0) + errType(mat1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    if (S-bestS) < tols:
        return None,leafType(dataSet)
    mat0,mat1 = binarySplitDataSet(dataSet,bestIndex,bestValue)
    if (shape(mat0)[0] < tolN) or (shape(mat1)[0] < tolN):
        return None,leafType(dataSet)
    return bestIndex,bestValue


def createTree(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
    feat,val = chooseBestSplit(dataSet,leafType,errType,ops)
    if feat == None:return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet,rSet = binarySplitDataSet(dataSet,feat,val)

    retTree['right'] = createTree(rSet,leafType,errType,ops)
    retTree['left'] = createTree(lSet,leafType,errType,ops)
    return retTree


print(' ')
print('*********************  prune  *********************')
print(' ')

# prune the tree for avoid overfitting,preprune,and postprune
def isTree(obj):
    return (type(obj)).__name__ == 'dict'

def getMean(tree):
    if isTree(tree['left']):
        tree['left'] = getMean(tree['left'])
    if isTree(tree['right']):
        tree['left'] = getMean(tree['right'])
    return (tree['left'] + tree['right'])/2.0

def prune(tree,testData):
    if shape(testData)[0] == 0:return getMean(tree)

    if (isTree(tree['right'])) or (isTree(tree['left'])):
        lset,rset = binarySplitDataSet(testData,tree['spInd'],tree['spVal'])

    if isTree(tree['left']):
        tree['left'] = prune(tree['left'],lset)
    if isTree(tree['right']):
        tree['right'] = prune(tree['right'],rset)

    if not isTree(tree['left']) and not isTree(tree['right']):
        lset,rset = binarySplitDataSet(testData,tree['spInd'],tree['spVal'])
        errorNoMerge = sum(power(lset[:,-1] - tree['left'],2)) + sum(power(rset[:,-1] - tree['right'],2))
        treeMean = (tree['left'] + tree['right'])/2.0
        errorMerge = sum(power(testData[:,-1]-treeMean,2))
        if errorMerge < errorNoMerge:
            print('merge')
            return treeMean
        else:
            return tree
    else:return tree


print(' ')
print('*********************  model tree  *********************')
print(' ')

def linearSolve(dataset):
    m,n = shape(dataset)
    X = mat(ones((m,n)));Y = mat(ones((m,1)))
    X[:,1:n] = dataset[:,0:n-1];Y=dataset[:,-1]

    xTx = X.T*X
    if linalg.det(xTx) == 0:
        raise NameError('this matrix is singular ,cannot be inverse,try incresing the second balue of ops')
    ws = xTx.I*(X.T*Y)
    return ws,X,Y

def modelLeaf(dataset):
    ws,X,Y = linearSolve(dataset)
    return ws

def modelErr(dataset):
    ws,X,Y = linearSolve(dataset)
    yHat = X*ws
    return sum(power(Y-yHat,2))


def testModelTree():
    myMat = mat(loadDataSet('exp2.txt'))
    modelTree =createTree(myMat,modelLeaf,modelErr,(1,10))
    print(modelTree)


import matplotlib.pyplot as plt
if __name__ == '__main__':
    testMat = mat(eye(4))
    print(testMat)
    m0,m1 = binarySplitDataSet(testMat,2,0.6)
    print(m0)

    print(m1)

    # myDat = loadDataSet('ex00.txt')
    myDat = loadDataSet('ex0.txt')
    xdata = [];ydata = []
    for p in myDat:
        xdata.append(p[1])
        ydata.append(p[2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xdata, ydata)
    # plt.show()

    # change ops to preprune, ops[0]  is tolrance, op2[1] is the minmim size of sample to split
    myTree = createTree(mat(myDat),ops=(0,1))
    print(myTree)

    testData = mat(loadDataSet('ex2test.txt'))
    postPruneTree = prune(myTree,testData)
    print(postPruneTree)

    print(' ')
    print('*********************  model tree  *********************')
    print(' ')

    testModelTree()


