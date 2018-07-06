from numpy import *
def loadDataSet(filename):
    datamat = []
    fr = open(filename)
    for line in fr.readlines():
        curLine = line.strip().split('\t')
        fltLine = list(map(float,curLine))
        datamat.append(fltLine)
    return  datamat

def distEclud(vecA,vecB):
    return sqrt(sum(power(vecA-vecB,2)))

def randCent(dataset,k):
    n = shape(dataset)[1]
    centroids = mat(zeros((k,n)))
    for j in range(n):
        minJ = min(dataset[:,j])
        rangeJ = float(max(dataset[:,j]) - minJ)
        centroids[:,j] = mat(minJ + rangeJ*random.rand(k,1))
    return centroids


print(' ')
print('*********************  K-Means  *********************')
print(' ')

def kMeans(dataset,k,distMeans=distEclud,createCent=randCent):
    m = shape(dataset)[0]
    clusterAssment = mat(zeros((m,2)))
    centroids = createCent(dataset,k)
    clusterChanged = True

    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = inf;minIndex = -1
            for j in range(k):
                distJI = distMeans(centroids[j,:],dataset[i,:])
                if distJI < minDist:
                    minDist = distJI;minIndex = j
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True
            clusterAssment[i,:] = minIndex,minDist**2
        print(centroids)
        for cent in range(k):
            ptsInClust = dataset[nonzero(clusterAssment[:,0].A==cent)[0]]
            centroids[cent,:] = mean(ptsInClust,axis=0)
    return centroids,clusterAssment

print(' ')
print('*********************  bisecting K-means  *********************')
print(' ')

def biKmeans(dataset,k,distMeans=distEclud):
    m = shape(dataset)[0]
    clusterAssment = mat(zeros((m,2)))
    centriod0 = mean(dataset,axis=0).tolist()[0]
    centList = [centriod0]
    for j in range(m):
        clusterAssment[j,1] = distMeans(mat(centriod0),dataset[j,:])**2

    while len(centList) < k:
        lowestSSE = inf
        for i in range(len(centList)):
            ptsInCurrCluster = dataset[nonzero(clusterAssment[:,0].A==i)[0],:]
            centroidMat ,splitClustAss = kMeans(ptsInCurrCluster,2,distMeans)
            sseSplit = sum(splitClustAss[:,1])
            sseNotSplit = sum(clusterAssment[nonzero(clusterAssment[:,0].A!=i)[0],1])
            print('sseSplit:,and sseNotSplit',sseSplit,sseNotSplit)
            if sseSplit + sseNotSplit < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit

#         update
        bestClustAss[nonzero(bestClustAss[:,0].A==1)[0],0] = len(centList)
        bestClustAss[nonzero(bestClustAss[:,0].A==0)[0],0] = bestCentToSplit

        print('the bestCentToSplit is :',bestCentToSplit)
        print('the len of bestClustAss is :',len(bestClustAss))
        centList[bestCentToSplit] = bestNewCents[0,:]
        centList.append(bestNewCents[1,:])
        clusterAssment[nonzero(clusterAssment[:,0].A==bestCentToSplit)[0],:] = bestClustAss

    return centList,clusterAssment



def count_words(s, n):
    """Return the n most frequently occuring words in s."""
    word_list = s.split(' ')

    fre_list = []
    # TODO: Count the number of occurences of each word in s
    for word in set(word_list):
        num = word_list.count(word)
        fre_list.append((word,num))

    # TODO: Sort the occurences in descending order (alphabetically in case of ties)
    # list1 = sorted(fre_list, key=lambda x: (x[1],x[0]), reverse=True)
    # # list2 = sorted(list1,key = lambda  x:x[0])
    # top_n = list1[0:n]

    list1 = sorted(fre_list,key=lambda x:x[0])
    list2 = sorted(list1,key=lambda  x:x[1],reverse=True)
    top_n = list2[0:n]
    # TODO: Return the top n most frequent words.
    return top_n


def test_run():
    """Test count_words() with some inputs."""
    print(count_words("cat bat mat cat bat cat", 3))
    print(count_words("betty bought a bit of butter but the butter was bitter", 3))


if __name__ == '__main__':
    dataMat = mat(loadDataSet('testSet.txt'))
    min2 = min(dataMat[:,1])
    max2 = max(dataMat[:,1])
    print(min2,max2)


    randPts = randCent(dataMat,2)
    print(randPts)

    myCentroids,clustAssing = kMeans(dataMat,4)

    print(' ')
    print('*********************  bisecting k-means  *********************')
    print(' ')
    dataMat2 = mat(loadDataSet('testSet2.txt'))
    cenList,assmernt = biKmeans(dataMat2,3)
    print(cenList)


    test_run()