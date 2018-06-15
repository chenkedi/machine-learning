import matplotlib.pyplot as plt

# 定义文本框和箭头的格式
# 下面使用了两种不同的字典构造方式
decisionNode = dict(boxstyle='sawtooth', fc='0.8')
leafNode = {"boxstyle": "round4", "fc": "0.8"}
arrow_args = dict(arrowstyle='<-')


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    """
    绘制带箭头的注解（即文字标签）
    :param nodeTxt:
    :param centerPt:
    :param parantPt:
    :param nodeType:
    :return:
    """
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


def createPlot():
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    createPlot.ax1 = plt.subplot(111, frameon=False)
    plotNode('decision node', (0.5, 0.1), (0.1, 0.5), decisionNode)
    plotNode('leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()

createPlot()