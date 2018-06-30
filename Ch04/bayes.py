


print(' ')
print('*********************  prepare dara  *********************')
print(' ')


def loadDataSet():
    postinglist = [['my','dog','has','flea','problems','help','please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classvec = [0,1,0,1,0,1]  # 1 insult word,,  0 normal word
    return postinglist,classvec

def createVocabList(dataSet):
    vocabset = set([])
    for document in dataSet:
        vocabset = vocabset | set(document)
    return sorted(list(vocabset))

# set of word vector
def setOfWords2Vec(vocabList,inputSet):

    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print('the word %s is not in my vocabulary' % word)
    return returnVec

# bag of words vector
def bagOfWords2VecMN(vocabList,inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

print(' ')
print('*********************  bayes train classfier  *********************')
print(' ')
from numpy import *
import numpy as np

def trainNB0(trainMatrix,trainCategory):
    numTrainsDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainsDocs)
    p0Num = zeros(numWords)
    p1Num = zeros(numWords)
    p0Denom = 0.0
    p1Denom = 0.0

    for i in range(numTrainsDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum((trainMatrix[i]))
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])

    p1Vect = p1Num/p1Denom
    p0Vect = p0Num/p0Denom

    print(p0Vect,p0Denom)
    print(p1Vect,p1Denom)

    return p0Vect,p1Vect,pAbusive

from math import log

def trainImprovedNB(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)

    # incase one of the factor is zero lead to zero result,so give numerator 1 and denominator 2 , initializer value  
    p0Num = ones(numWords);p1Num = ones(numWords)
    p0Denom = 2.0;p1Denom = 2.0

    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])

    # use log func to incase:too many small numbers multiplied could lead to zero
    # print(p0Num,p0Denom)
    # print(p1Num,p1Denom)

    p0v = np.log(p0Num/p0Denom)
    p1v = np.log(p1Num/p1Denom)
    return p0v,p1v,pAbusive

def classifyNB(vect2Classify,p0v,p1v,pclass1):
    p1 = sum(vect2Classify*p1v) + log(pclass1)
    p0 = sum(vect2Classify*p0v) + log(1-pclass1)
    
    if p1 > p0:
        return 1
    else:
        return 0


def testNB():
    listOfPosts, listClasses = loadDataSet()
    myVocabList = createVocabList(listOfPosts)
    trainMat = []
    for postDoc in listOfPosts:
        trainMat.append(setOfWords2Vec(myVocabList,postDoc))
    p0v,p1v,pAb = trainNB0(trainMat,listClasses)
    
    testEntry = ['love','my','dalmatioin']
    thisDoc = setOfWords2Vec(myVocabList,testEntry)
    print(testEntry,' classified as :' ,classifyNB(thisDoc,p0v,p1v,pAb))

    testEntry2 = ['stupid', 'garbage', 'dalmatioin']
    thisDoc2 = setOfWords2Vec(myVocabList, testEntry2)
    print(testEntry2, ' classified as :', classifyNB(thisDoc2, p0v, p1v, pAb))


print(' ')
print('*********************  spam test  *********************')
print(' ')

def textParse(bigString):
    import re
    listOfTokens = re.split(r'\W*',bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def spamTest():
    doc_list = [];class_list = [];full_test = []
    for i in range(1,26):
        # spamtext = open('email/spam/%d.txt' % i,encoding='ISO-8859-1').read()
        # print(type(spamtext).__name__)

        word_list = textParse(open('email/spam/%d.txt' % i,encoding='ISO-8859-1').read())
        doc_list.append(word_list)
        full_test.extend(word_list)
        class_list.append(1)
        word_list = textParse(open('email/ham/%d.txt' % i,encoding='ISO-8859-1').read())
        doc_list.append(word_list)
        full_test.extend(word_list)
        class_list.append(0)

    vocablist = createVocabList(doc_list)
    print(len(vocablist))

    trainSet = list(range(50));testSet = []
    for i in range(10):
        randindex = int(random.uniform(0,len(trainSet)))
        testSet.append(trainSet[randindex])
        del (trainSet[randindex])
    print(trainSet,testSet)

    trainMat = []; trainClasses = []
    for docIndex in trainSet:
        trainMat.append(bagOfWords2VecMN(vocablist,doc_list[docIndex]))
        trainClasses.append(class_list[docIndex])

    print(trainMat)
    p0v,p1v,pSpam = trainImprovedNB(array(trainMat),array(trainClasses))
    error_count = 0.0

    for docindex in testSet:
        wordvector = bagOfWords2VecMN(vocablist,doc_list[docIndex])
        if classifyNB(array(wordvector),p0v,p1v,pSpam) != class_list[docIndex]:
            error_count += 1

    print('errorcount %d' % error_count)
    print('the error rate is:',float(error_count)/len(testSet))




print(' ')
print('*********************  rss  *********************')
print(' ')

def calcMostFreq(vocabList,fullText):
    import operator
    freqDict = {}

    for token in vocabList:
        freqDict[token] = fullText.count(token)

    sortedFreq = sorted(freqDict.items(),key=operator.itemgetter(1),reverse=True)
    return  sortedFreq[0:30]


def localWords(feed1,feed0):
    import feedparser
    docList = [];classList = [];fullText = []
    minLen = min(len(feed1['entries'],len(feed0['entries'])))

    for i in range(minLen):

        wordList = textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    top30Words = calcMostFreq(vocabList,fullText)

    for pairW in top30Words:
        if packbits[0] in vocabList:
            vocabList.remove(pairW[0])

    trainingSet = list(range(minLen * 2));testSet = []
    for i in range(20):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del trainingSet[randIndex]

    trainMat = [];trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])

    p0v,p1v,pSpam = trainImprovedNB(array(trainMat),array(trainClasses))

    errorCount = 0.0
    for docIndex in test:
        wordVect = bagOfWords2VecMN(vocabList,docList[docIndex])
        if classifyNB(array(wordVect),p0v,p1v,pSpam) != classList[docIndex]:
            errorCount += 1

    print('the error rate is :',float(errorCount)/len(testSet))
    return vocabList,p0v,p1v


if __name__ == '__main__':
    listOfPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOfPosts)
    print(myVocabList)

    docV1 = setOfWords2Vec(myVocabList,listOfPosts[0])
    docV2 = setOfWords2Vec(myVocabList,listOfPosts[1])

    print(docV1,'\n',docV2)


    print(' ')
    print('*********************  train model  *********************')
    print(' ')

    trainMat = []
    for postDoc in listOfPosts:
        trainMat.append(setOfWords2Vec(myVocabList,postDoc))

    p0v,p1v,pAb = trainNB0(trainMat,listClasses)
    print(p0v)
    print(p1v)
    print(pAb)

    print(' ')
    print('*********************  test nativ bayes  *********************')
    print(' ')
    
    testNB()

    print(' ')
    print('*********************  spam test  *********************')
    print(' ')

    spamTest()


    print(' ')
    print('*********************  rss  *********************')
    print(' ')


    import feedparser
    ny = feedparser.parse('http://blog.csdn.net/together_cz/article')
    print(ny)
