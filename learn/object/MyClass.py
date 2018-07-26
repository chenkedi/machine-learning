"""
Python 对象编程基础
"""

# 类名的括号内显示指定该类的父类，若无则使用object代替，object为所有类的父类
class Myclass(object):
    """
    类内部定义的所有方法均需要含有self参数，代表类当前的实例化对象，但在外部调用时，不需要传递该参数
    """

    def __init__(self, data1, data2):
        """
        此处定义了两个类变量data1和data2，由构造函数init在类构造时进行初始化
        :param data1: __data1 前面的__标识该类变量为私有的
        :param data2:
        """
        self.__data1 = data1
        self.data2 = data2

    def __func1(self):
        """
        与类变量一样，__表示私有方法
        :return:
        """
        print("Myclass 的私有方法被调用")

    def print_data(self):
        self.__func1()
        print(self.__data1)
        print(self.data2)

    def setData(self, data):
        """
        通过公开方法设置私有变量的值
        :param data:
        :return:
        """
        self.__data1 = data


class1 = Myclass('first_data', 'second_data')
print(class1.data2)

## 直接访问私有变量会报错
print(class1.__data1)

class1.print_data()