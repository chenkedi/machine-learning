from numpy import *
import boost


def loadSimpleData():
    """
    返回一个测试算法基本正确性的特征数据矩阵
    :return:
    """
    dataMat = matrix([[1., 2.1],
                      [2., 1.1],
                      [1.3, 1.],
                      [1., 1.],
                      [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return dataMat, classLabels


def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    # 存储每次迭代的最优弱分类器
    weakClassArr = []
    m = dataArr.shape[0]
    # 每个样本的初始化权重为 1/m
    D = mat(ones((m, 1)) / m)
    n = dataArr.shape[1]
    aggClassEst = mat(zeros((m, 1)))

    print("##########Training begin .......##############")
    print()
    for i in range(numIt):
        print("===========Epoch %d=========" % i)
        bestStump, error, classEst = boost.buildStump(dataArr, classLabels, D)
        print("D: ", D.T)

        # 计算alpha, 注意这里为了防止error为0造成除0异常，使用了一个极小的浮点数来代替
        alpha = float(0.5 * log((1.0 - error) / max(error, 1e-16)))
        bestStump['alpha'] = alpha

        # 将本次迭代的最佳单层决策树加入单层决策树列表
        weakClassArr.append(bestStump)
        print("classEst: ", classEst.T)

        # 计算下一次迭代的权重向量D
        # 若样本被正确分类，则系数为e(-alpha),否则为e(alpha), 先确定指数alpha的正负
        # stumpClassify中将样本分类为-1和+1，而真正的样本标签也是+1与-1, alpha的符号刚好与真正样本标签想反
        # 而预测正确则两个矩阵对应元素为同号得正，否则为负号，所以只要将alpha的符号设置为负即可
        expon = multiply(-alpha * mat(classLabels).T, classEst)
        D = multiply(D, exp(expon)) / D.sum()

        # 更新累积类别估计值（即当前迭代次数的集成学习估计结果）
        aggClassEst += alpha * classEst
        print("aggClassEst: ", aggClassEst.T)

        # 计算累积类别估计的错误率
        # 先计算累积错误的个数， 再求出错误率
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))
        aggErrorRate = aggErrors.sum() / m
        print("total error: ", aggErrorRate)
        print("===========Epoch %d End=========" % i)
        print()
        # 若错误率为0， 则退出循环
        if (aggErrorRate == 0):
            break
    # 返回弱学习器的集合（包括每个弱学习器的权重alpha）作为集成学习模型
    return weakClassArr

def adaClassify(dataToClass, classifierArr):
    """
    根据训练得到的一组弱分类器及其alpha权重，对数据进行分类
    :param datToClass: 待分类数据的新数据
    :param classifierArr: 弱分类器集合
    :return:
    """
    dataMatrix = mat(dataToClass)
    m = dataMatrix.shape[0]
    aggClassEst = mat(zeros((m, 1)))

    print("#############Prediction begin......##########")
    for i in range(len(classifierArr)):
        print("==========%d th classifier predict result:===========" % i)
        classEst = boost.stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'],
                                      classifierArr[i]['ineq'])
        aggClassEst += classEst * classifierArr[i]['alpha']
        print(aggClassEst)
        print()

    return sign(aggClassEst)

if __name__ == "__main__":
    dataArr, labelArr = loadSimpleData()
    classifierArr = adaBoostTrainDS(dataArr,labelArr, 30)
    adaClassify([0, 0], classifierArr)