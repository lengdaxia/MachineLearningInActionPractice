import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
# add chinese characters in pic
# from  pylab import *
# mpl.rcParams['font.sans-serif'] = ['SimHei']

decisonNode = dict(boxstyle='sawtooth',fc='0.8')
leafNode = dict(boxstyle='round4',fc='0.8')
arrow_args = dict(arrowstyle='<-')


def plotNode(nodeTxt,centerPt,parentPt,nodeType):
    createPlot.ax1.annotate(nodeTxt,xy=parentPt,xycoords='axes fraction',xytext=centerPt,textcoords='axes fraction',
                            va='center',ha='center',bbox=nodeType,arrowprops=arrow_args,
                            fontproperties=FontProperties(fname='/System/Library/Fonts/PingFang.ttc'))
def createPlot():
    fig = plt.figure(1,facecolor='white')
    fig.clf()
    createPlot.ax1 = plt.subplot(111,frameon=False)
    plotNode('决策节点',(0.5,0.1),(0.1,0.5),decisonNode)
    plotNode('叶节点',(0.8,0.1),(0.3,0.8),leafNode)
    plt.show()


# get tree width
def getNumLeafs(myTree):
     numLeafs = 0
     # print(myTree.keys())
     firstStr = list(myTree.keys())[0]
     secondDic = myTree[firstStr]
     for key in secondDic.keys():
         if type(secondDic[key]).__name__ == 'dict':
             numLeafs += getNumLeafs(secondDic[key])
         else:
            numLeafs += 1
     return numLeafs

# get tree height
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDic = myTree[firstStr]
    for key in secondDic.keys():
        if type(secondDic[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDic[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth : maxDepth = thisDepth
    return maxDepth

def retriveTree(i):
    listOfTree = [{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                  {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}]
    return listOfTree[i]



# plot tree func
def plotMidText(centerPt,parentPt,txtStr):
    xMid = (parentPt[0] - centerPt[0])/2 + centerPt[0]
    yMid = (parentPt[1] - centerPt[1])/2 + centerPt[1]
    createPlot.ax1.text(xMid,yMid,txtStr)

def plotTree(myTree,parentPt,nodeText):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstStr = list(myTree.keys())[0]
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW,plotTree.yOff)
    plotMidText(cntrPt,parentPt,nodeText)
    plotNode(firstStr,cntrPt,parentPt,decisonNode)
    secondDic = myTree[firstStr]
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD

    for key in secondDic.keys():
        if type(secondDic[key]).__name__ == 'dict':
            plotTree(secondDic[key],cntrPt,str(key))
        else:
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDic[key],(plotTree.xOff,plotTree.yOff),cntrPt,leafNode)
            plotMidText((plotTree.xOff,plotTree.yOff),cntrPt,str(key))

    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD


def createPlot(inTree):
    fig = plt.figure(1,facecolor='white')
    fig.clf()
    axprops = dict(xticks=[],yticks=[])
    createPlot.ax1 = plt.subplot(111,frameon=False,**axprops)
    plotTree.totalW = getNumLeafs(inTree)
    plotTree.totalD = getTreeDepth(inTree)
    plotTree.xOff = -0.5/plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree,(0.5,1.0),'')
    plt.show()










if __name__ == '__main__':
    print(' ')
    print('*********************  plot *********************')
    print(' ')

    # createPlot()


    print(' ')
    print('*********************  get tree width and height  *********************')
    print(' ')

    myTree = retriveTree(1)
    print(myTree)

    w = getNumLeafs(myTree)
    h = getTreeDepth(myTree)
    print('width is %d, depth is %d' %(w,h))

    createPlot(myTree)
