
import random

print(' ')
print('*********************  star funcs  *********************')
print(' ')

def loadDataSet(filename):
    dataMat = [];labelMat = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

def selectJrand(i,m):
    j=i
    while(j==i):
        j = int(random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

print(' ')
print('*********************  simple smo pattern  *********************')
print(' ')


from numpy import *
import time

def smoSimple(dataMatIn,classLabels,C,toler,maxIter):
    star_time = time.time()

    dataMatrix = mat(dataMatIn);labelMat = mat(classLabels).transpose()
    b=0;m,n = shape(dataMatIn)
    alphas = mat(zeros((m,1)))
    iter = 0

    while (iter<maxIter):
        alphaParisChanged = 0
        for i in range(m):
            fXi = float(multiply(alphas,labelMat).T * (dataMatrix*dataMatrix[i,:].T)) + b
            Ei = fXi - float(labelMat[i])
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C) ) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i,m)
                fXj = float(multiply(alphas,labelMat).T * (dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])
                alphaIold = alphas[i].copy()
                alphaJold = alphas[j].copy()

                if (labelMat[i] != labelMat[j]):
                    L = max(0,alphas[j]-alphas[i])
                    H = min(C,C+alphas[j]-alphas[i])
                else:
                    L = max(0,alphas[i]+alphas[j]-C)
                    H = min(C,alphas[j]+alphas[i])

                if L==H:print('L==H');continue
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,
                                                                                                    :]*dataMatrix[j,:].T
                if eta >= 0:print('eta >= 0');continue
                alphas[j] -= labelMat[j]*(Ei-Ej)/eta
                alphas[j] = clipAlpha(alphas[j],H,L)
                if abs(alphas[j] - alphaJold)<0.00001:print('j not moving enghou');continue

                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold-alphas[j])

                b1 = b - Ei - labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*(
                    alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej - labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]*(
                    alphas[j] - alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T

                if (0 < alphas[i]) and (C > alphas[i]) : b = b1
                elif (0 < alphas[j]) and(C > alphas[j]) : b = b2
                else: b = (b1+b2)/2
                alphaParisChanged += 1
                print('iter:%d i:%d,pairs changed %d'%(iter,i,alphaParisChanged))
        if alphaParisChanged == 0:iter += 1
        else:iter = 0
        print('iterration number %d' % iter)

    end_time = time.time()
    print('func time consume : ', (end_time-star_time))
    return b,alphas





print(' ')
print('*********************  complete SMO model*********************')
print(' ')


class optStruct:

    def __init__(self,dataMatIn,classLabels,C,toler):
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0
        self.eCache = mat(zeros((self.m,2)))


def calcEk(oS,k):
    fXk = float(multiply(oS.alphas,oS.labelMat).T*(oS.X*oS.X[k,:].T)) + oS.b
    Ek = fXk - float(oS.labelMat[k])
    return Ek

def selectJ(i,oS,Ei):
    maxK = -1;maxDeltaE = 0;Ej = 0
    oS.eCache[i] = [1,Ei]
    validEcachaList = nonzero(oS.eCache[:,0].A)[0]
    if len(validEcachaList) > 1:
        for k in validEcachaList:
            if k == i:continue
            Ek = calcEk(oS,k)
            deltaE = abs(Ei-Ek)
            if deltaE > maxDeltaE:
                maxK = k;maxDeltaE = deltaE;Ej=Ek
        return maxK,Ej
    else:
        j = selectJrand(i,oS.m)
        Ej = calcEk(oS,j)
    return j,Ej

def updateEk(oS,k):
    Ek = calcEk(oS,k)
    oS.eCache[k] = [1,Ek]



def innerL(i,oS):
    Ei = calcEk(oS,i)
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i]<oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and(oS.alphas[i]
                                                                                                        > 0)):
        j,Ej = selectJ(i,oS,Ei)
        alphaIold = oS.alphas[i].copy();alphaJold = oS.alphas[j].copy()
        if oS.labelMat[i] != oS.labelMat[j]:
            L = max(0,oS.alphas[j] - oS.alphas[i])
            H = min(oS.C,oS.C + oS.alphas[j]- oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] +oS.alphas[j] - oS.C)
            H = min(oS.C,oS.alphas[j] + oS.alphas[i])
        if L==H:print('L==H');return 0
        eta = 2.0*oS.X[i,:]*oS.X[i,:].T - oS.X[i,:]*oS.X[i,:].T - oS.X[j,:]*oS.X[j,:].T
        if eta >= 0:print('eta >= 0');return 0

        oS.alphas[j] += oS.labelMat[j]*(Ei-Ej)/eta
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)

        updateEk(oS,j)

        if abs(oS.alphas[j] - alphaJold) < 0.00001:
            print('j not moving enghou')
            return  0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold-oS.alphas[j])
        updateEk(oS,i)

        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, i] - oS.labelMat[j] * (
        oS.alphas[j] - alphaJold) * oS.K[i, j]
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i, j] - oS.labelMat[j] * (
        oS.alphas[j] - alphaJold) * oS.K[j, j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:return 0


def smoP(dataMatIn,classLabels,C,toler,maxIter,kTup=('lin',0)):
    star_time = time.time()

    oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler)
    iter = 0
    entireSet = True;alphaPairsChanged = 0
    while iter < maxIter and alphaPairsChanged > 0 or entireSet:
        alphaPairsChanged = 0
        if entireSet:
            for i in range(oS.m):
                alphaPairsChanged += innerL(i,oS)
            print('fullSet ,iter:%d i:%d,pairs changed %d'%(iter,i,alphaPairsChanged))
        else:
            nonBoundIs = nonzero((oS.alphas.A > 0)) * (oS.alphas.A < C)[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                print('non_bound ，iter:%d,i:%d,pairs changed:%d'%(iter,i,alphaPairsChanged))
            iter += 1
        if entireSet:entireSet = False
        elif alphaPairsChanged == 0:entireSet = True ;print('iteration number :%d'% iter)

    end_time = time.time()
    print('func time consume : ',(end_time - star_time))
    return oS.b,oS.alphas


def calcWs(alphas,dataArr,classLabels):
    X = mat(dataArr);labelMat = mat(classLabels).transpose()
    m,n = shape(X)
    w = zeros((n,1))
    for i in range(m):
        w += multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w








if __name__ == '__main__':
    dataArr,labelArr = loadDataSet('testSet.txt')
    print(labelArr)
    # b,alphas = smoSimple(dataArr,labelArr,0.6,0.001,40)
    # print(b)
    # print(alphas[alphas>0])
    # print(shape(alphas[alphas>0]))wq`
    # for i in range(100):
    #     if alphas[i] > 0.0:
    #         print(dataArr[i],labelArr[i])

    print(' ')
    print('*********************  completd smo pattern  *********************')
    print(' ')

    b1,alpha1 = smoP(dataArr,labelArr,0.6,0.001,40)
    print(b1,alpha1)

    ws = calcWs(alpha1,dataArr,labelArr)
    print(ws)

    dataMat = mat(dataArr)
    p = dataMat[0]*mat(ws) + b1
    print(p,labelArr[0])

