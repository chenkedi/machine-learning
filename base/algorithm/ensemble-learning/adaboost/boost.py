from numpy import *


def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    """
    本函数作为构建"决策树桩"（即单层决策树）的子函数，用于对指定的属性列（dimen）
    根据给定的阈值（threshVal），使用给定的区间划分（threshIneq，包含小于等于和大于）
    对数据进行分类。
    数据处于给定不等号这边的样本设置为-1，不处于给定不等号的则将标签设置为+1
    :param dataMatrix:
    :param dimen:
    :param threshVal:
    :param threshIneq:
    :return:
    """
    resArray = ones((dataMatrix.shape[0], 1))  # 书中的原代码为 shape(dataMatrix)[0]

    if (threshIneq == 'lt'):
        resArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        resArray[dataMatrix[:, dimen] > threshVal] = -1.0

    return resArray


def buildStump(dataArr, classLabels, D):
    """
    该函数遍历所有特征、每个特征最适合进行划分的数值和所有可能的不等号，得到当前权重矩阵D下的最小error的决策树桩
    :param dataArr:
    :param classLabels:
    :param D:
    :return:
    """
    # mat函数用于将输入变为矩阵，若输入为ndarray或者matrix,该方法不会进行深度拷贝
    dataMatrix = mat(dataArr)
    # classLables原本为 1 * m 维，此处转换为列向量
    labelMatrix = mat(classLabels).T
    m, n = shape(dataMatrix)
    numSteps = 10
    bestStump = {}
    bestClassEst = mat(zeros((m, 1)))
    minErr = inf

    # 遍历n个特征维度
    for i in range(n):
        # 求每个特征维度中的最大，最小值，并处以numSteps得到每步的间隔
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin) / numSteps

        # 遍历特征分为numSteps的均匀空间
        # 此处从-1开始是为了能够遍历第一个 stepSize 表示的阈值
        for j in range(-1, int(stepSize) + 1):

            for inequal in ['lt', 'gt']:
                threshVal = rangeMin + float(j) * stepSize
                predictVals = stumpClassify(dataMatrix, i, threshVal, inequal)

                errArr = mat(ones((m, 1)))
                # 单层决策树判断为正确的置为0
                errArr[predictVals == labelMatrix] = 0
                # D 矩阵为每个样本的权重矩阵，为 m * 1维，此处*号为两个向量的内积?
                # 本来应该使用D.T.dot(errArr)来进行矩阵乘法的，不知为何乘号也可以
                # 作为内积运算符号
                weightedError = D.T * errArr

                # print("split: dim %d, thresh: %.2f, thresh inequal: %s, the weightError is %.3f."
                #       % (i, threshVal, inequal, weightedError))

                if(weightedError < minErr):
                    minErr = weightedError
                    bestClassEst = predictVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal

    return bestStump, minErr, bestClassEst
