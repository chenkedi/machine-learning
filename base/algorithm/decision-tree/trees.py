"""
Decision tree which implements ID3 algorithm
该算法没有实现预剪枝或后剪枝的防过拟合策略
"""
from math import log
import operator


def calcShannonEnt(dataSet):
    """
    计算给定数据集的信息熵
    :param dataSet:
    :return:
    """
    numEntries = len(dataSet)
    # key为类别，值为类别计数
    labelCounts = {}

    for featureVec in dataSet:
        # 取每行特征的最后一列，即标签
        currentLable = featureVec[-1]
        if currentLable not in labelCounts.keys():
            labelCounts[currentLable] = 0
        labelCounts[currentLable] += 1

    shannonEnt = 0.0
    for key in labelCounts:
        prob = labelCounts[key] / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt


def splitDataSet(dataSet, axis, value):
    """
    按照给定的属性值（valule）和该属性对应的列（axis）对数据进行划分
    :param dataSet:
    :param axis: 属性a对应特征中的列
    :param value: 属性a的取值
    :return:
    """
    resDataSet = []
    for featureVec in dataSet:
        # 当前行在属性a上的取值等于给定的value
        if featureVec[axis] == value:
            # 由于决策树使用离散取值的属性进行划分时，只能使用一次，所以要将划分后的数据去掉该属性列
            reducedFeatureVec = featureVec[:axis]
            reducedFeatureVec = featureVec.extend(featureVec[axis + 1:])
            resDataSet.append(reducedFeatureVec)
    return resDataSet


def chooseBestFeatureToSplit(dataSet):
    """
    使用最大信息增益选择进行决策树划分的属性
    :param dataSet:
    :return:
    """
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1

    for i in range(numFeatures):
        # 取得第i个属性的所有取值并去重
        featureList = [example[i] for example in dataSet]
        uniqueVals = set(featureList)

        # 计算使用属性i划分后的信息增益之和: sigma(Gain(D,i))
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            # 在第i个属性上取值为value的数据占dataSet的比例，对应西瓜书里面的D(v)/D
            ratio = len(subDataSet) / len(dataSet)
            newEntropy += ratio * calcShannonEnt(subDataSet)

        infoGain = baseEntropy - newEntropy

        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature


def majorityCnt(classList):
    """
    统计一个划分好的节点中数量最多的那一类的类别标号
    :param classList:
    :return:
    """
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    # operator是python的工具集合，如operator.mul(x,y) = x * y
    # operator.itermgetter用于获取对象某个维度的数据，返回值是一个函数
    # 如：a = [1, 2, 3], b = operator.itemgetter(0), b(a) = 1
    # 注意，python3中废弃了2中字典的iteriterm()方法，3中的items返回的就是一个迭代器，不像2中那样返回字典的拷贝列表而占用额外内存
    # 3中的items方法类似于2中的viewitems()
    # 另外注意，sorted方法返回的是另外一个dict，如果第一个参数不加items，则结果只会返回key指定的那一维度
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels):
    """
    递归生成决策树
    :param dataSet:
    :param labels: 每个特征的索引对应的名字，如第一维特征称作no surfing
    :return:
    """
    # 获取当前划分数据集中的类别标签列表
    classList = [example[-1] for example in dataSet]
    # 统计标签列表是否属于同一个类别,是的话则返回该类别标签
    if(classList.count(classList[0]) == len(classList)):
        return classList[0]
    # 当dataSet的任意一行只剩一列（即所有特征被 chooseDestFeatureToSplit函数遍历完成只剩下标签列）,且标签不唯一时
    if(len(dataSet[0]) == 1):
        return majorityCnt(classList)

    # 不符合两个递归结束条件的情况下，根据信息增益选择最佳特征
    bestFeature = chooseBestFeatureToSplit(dataSet)
    bestFeatureLabel = labels[bestFeature]
    myTree = {bestFeatureLabel:{}}
    del(labels[bestFeature])
    # 得到最佳划分属性的所有取值
    featureValues = [example[bestFeature] for example in dataSet]
    uniqueValues = set(featureValues)
    for value in uniqueValues:
        subLabels = labels[:] # 这里是为了防止python列表的引用传递产生问题，其实上面的del完全没有必要，所有label都可指向同一个引用
        myTree[bestFeatureLabel][value] = createTree(splitDataSet(dataSet, bestFeature, value), subLabels)
    return myTree # python的返回值可以不一致？？ 最后一个return为字典，上面两个为class（即字符串）？？？


def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels
