# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt


# 定义文本框和箭头格式
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


def plotNode(nodeText, centerPt, parentPt, nodeType):
    # 绘制箭头的注释
    createPlot.ax1.annotate(nodeText, xy=parentPt, xycoords="axes fraction", xytext=centerPt,
                            textcoords="axes fraction", va="center", ha="center", bbox=nodeType,
                            arrowprops=arrow_args)



def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == "dict":
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == "dict":
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth

    return maxDepth


def retrieveTree(i):
    listOfTrees = [
        {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
        {'no surfacing': {0: 'no', 1: {'flippers': {0: {"head": {0: "no", 1: "yes"}}, 1: "no"}}}},
    ]

    return listOfTrees[i]


def plotMidText(centerPt, parentPt, txtString):
    # 计算父子节点的中间坐标
    xMid = (parentPt[0] - centerPt[0]) / 2.0 + centerPt[0]
    yMid = (parentPt[1] - centerPt[1]) / 2.0 + centerPt[1]

    # 添加文本标签信息
    createPlot.ax1.text(xMid, yMid, txtString)


def plotTree(myTree, parentPt, nodeTxt):
    # 当前子树的叶节点数与深度
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)

    firstStr = list(myTree.keys())[0]
    centerPt = (plotTree.xOff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.yOff)

    plotMidText(centerPt, parentPt, nodeTxt)  # 标记子节点属性值
    plotNode(firstStr, centerPt, parentPt, decisionNode)

    secondDict = myTree[firstStr]

    # 减少 y, 下一个节点将据此绘制
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD

    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == "dict":
            plotTree(secondDict[key], centerPt, str(key))
        else:
            # 叶子节点
            plotTree.xOff = plotTree.xOff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), centerPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), centerPt, str(key))
        # 绘制完当前子树所有的子节点之后, 恢复 y 坐标
    plotTree.yOff = plotTree.yOff + 1.0 / plotTree.totalD



def createPlot(inTree):
    fig = plt.figure(1, facecolor="white")
    fig.clf()

    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)  # 清空绘图区

    # 针对整棵树而言的, 宽度与深度, 全局
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    # 用于追踪当前 x, y 的位置 (当前绘制的节点的位置)
    plotTree.xOff = -0.5 / plotTree.totalW
    plotTree.yOff = 1.0

    # 初始坐标点: (0.5, 1.0)
    plotTree(inTree, (0.5, 1.0), "")

    plt.show()




if __name__ == '__main__':
    myTree = retrieveTree(1)
    print(getNumLeafs(myTree))
    print(getTreeDepth(myTree))
    createPlot(myTree)
