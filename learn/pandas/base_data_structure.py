# coding=utf-8
import numpy as np
import pandas as pd

if __name__ == "__main__":

    print("## series对象")
    s = pd.Series([12, -4, 7, 9], index=['a', 'b', 'c', 'd'])  # 当不指定index则默认使用与数组相同的数字索引，最好为元素指定有意义的索引
    print(s)
    print()
    print(s.values)  # 取series对象的值
    print()
    print(s.index)  # 取series对象的索引
    print()

    print("## 取series对象中的元素")
    print(s[0])
    print()
    print(s['b'])
    print()
    print(s[0:2])
    print()
    print(s[['b', 'c']])
    print()

    print("## 用numpy数组或其他Series对象创建新的series对象")
    arr = np.arange(1, 5)
    a = pd.Series(arr)
    print(a)
    print()
    b = pd.Series(s)
    print(b)
    print()
    arr[2] = 10  # 注意，新series对象是对原来数组和对象的引用，改变源对象也会更改新对象
    print(a)
    print()

    print("## series中的数学运算与常用数学函数")
    print(s / 2)  # 普通数学运算符可以直接使用
    print()
    print(np.log(s))  # 当使用numpy中的函数时，需要将series对象实例传入
    print()

    print("## 使用函数来判断series对象中的元素的组成成分")
    a = pd.Series([1, 0, 2, 1, 2, 3], index=['white', 'white', 'blue', 'green', 'green', 'yellow'])
    print(a.unique())  # unique返回对象中values元素去重后的结果
    print()
    print(a.value_counts())  # value_counts() 返回values中去重并统计出现次数的结果
    print()
    print(a.isin([0, 3]))  # 用于判断给定的一列元素是否包含在数据结构之中
    print()
    print(a[a.isin([0, 3])])  # 与numpy一样可以直接写入方括号中用于筛选数据
    print()

    print("## 使用函数来判断series对象中的元素是否为异常值")
    a = np.log(s)
    print(a.isnull())  # 返回一个布尔类型的series对象，其值由源series对象的元素是否为NaN决定
    print()
    print(a.notnull())
    print()
    print(a[a.isnull()])  # 同样,该函数可以用作筛选元素用途
    print()
    print(a[a.notnull()])
    print()

    print("## series对象与字典数据类型的相互转换")
    print()
    dict1 = {
        'red': 2000,
        'blue': 1000,
        'yellow': 500,
        'orange': 1000
    }
    print(pd.Series(dict1))
    print()
    color = ['red', 'yellow', 'orange', 'blue', 'green']
    print(pd.Series(dict1, index=color))  # 可以额外指定index，pandas会自动合并与index中与dict键相同的值，缺失值用NaN填充
    print()

    print("## series对象之间的运算")
    print()
    dict2 = {'red': 400, 'yellow': 1000, 'black': 700}
    series1 = pd.Series(dict1, index=color)
    series2 = pd.Series(dict2)
    print(series1 + series2)  # 运算会自动对齐先沟通呢给的index，其他只属于某一个series的对象的标签加入到结果中，但是值为NaN
    print()

    print("## DataFram对象的定义")
    data1 = {
        'color': ['blue', 'green', 'yellow', 'red', 'white'],
        'object': ['ball', 'pen', 'pencil', 'paper', 'mug'],
        'price': [1.2, 1.0, 0.6, 0.9, 1.7]
    }
    data2 = {
        'red': {2012: 22, 2013: 33},
        'white': {2011: 13, 2012: 22, 2013: 16},
        'blue': {2011: 17, 2012: 27, 2013: 18}
    }
    frame1 = pd.DataFrame(data1)
    frame2 = pd.DataFrame(data1, columns=['object', 'price'])  # 对于原始数据中不需要的列可以手动指定列名进行过滤
    frame3 = pd.DataFrame(data1, index=['one', 'two', 'three', 'four', 'five'])  # 如果不手动指定index，则默认为从0开始的索引
    frame4 = pd.DataFrame(np.arange(16).reshape((4, 4)),
                          index=['red', 'blue', 'yellow', 'white'],
                          columns=['ball', 'pen', 'pencil', 'paper'])  # 可以通过指定值矩阵，index和column三个参数来构造dataFrame
    frame5 = pd.DataFrame(data2)  # 直接通过两级结构的字典生成dataFrame,则列名为第一级结构的key, pandas会自动填充缺失的项目
    print(frame1)
    print()
    print(frame2)
    print()
    print(frame3)
    print()
    print(frame4)
    print()
    print(frame5)
    print()

    print("## 获得dataFrame对象的各种属性")
    print(frame1.columns)
    print()
    print(frame1.index)  # 分别获得所有列的名称和缩影的列表
    print()
    print(frame1.values)  # 获得存储在数据结构中的元素，返回结果为ndarray
    print()

    print("## 按行和按列来选取元素")
    print(frame1['price'])  # 单列选择，得到一个Series对象
    print()
    print(frame1.price)  # 同上的单列选择, 将列名看做dataFrame的示例属性
    print()
    print(frame1.ix[2])  # 单行选择,
    print()
    print(frame1.ix[[2, 4]])  # 指定多行的索引选择多行, 注意里面是两层中括号
    print()
    print(frame1.ix[2:4])  # 指定行索引的范围选择多行, 注意使用ix属性与下面直接在后使用方括号的区别
    print()
    print(frame1[2:4])  # 通过将行直接看为dataFrame的一部分,在后面直接使用方括号, 但是 print(frame1[2])这种方式是错误的，方括号中如果不是范围则被认为直接取列
    print()

    print("## 为DataFrame对象赋值")
    print()
    frame1.index.name = 'id'
    frame1.columns.name = 'item'  # 为index何column指定标签
    print(frame1)
    print()

    print("## 添加新列")
    frame1['new'] = 12  # 添加新列
    print(frame1)
    print()
    frame1['new'] = [3.0, 1.3, 2.2, 0.8, 1.1]  # 更新某一列
    print(frame1)
    print()
    ser_to_insert = pd.Series(np.arange(5))
    frame1['new'] = ser_to_insert  # 也可以通过series对象来添加或新郑一行
    print(frame1)
    del frame1['new'] # 删除一列的初级方法，后续有更通用的方法
    print()

    print("## 求DataFram与给定元素的包含关系")
    print(frame1.isin([1.0, 'pen']))  # 判断数据中是否包含给定的序列，得到boolean类型的DataFrame
    print()
    print(frame1[frame1.isin([1.0, 'pen'])])  # 同样可以用布尔类型的结果进行筛选,为false的地方用NaN填充
    print()
    print(frame4[frame4 < 12])  # 还可以通过其他的生成布尔矩阵的条件进行筛选
    print()

    print("## DataFrame的转置")
    print(frame1.T)
    print()

    print("## index对象的方法")
    ser = pd.Series([5, 0, 3, 8, 4], index=['red', 'blue', 'yellow', 'white', 'green'])
    print(ser.idxmax())
    print()
    print(ser.idxmin())  # 分别求最大和最小索引
    print()
    ser = pd.Series(range(6), index=['white', 'white', 'blue', 'green', 'green', 'yellow'])
    print(ser['white'])  # 当含有重复的索引时, 返回的是一个Series对象
    print()
    print(ser.index.is_unique)
    print()
    print(frame1.index.is_unique)  # 无论是Series还是DataFrame对象，他们的索引都有is_unique属性
    print()

    print("## 更换索引")
    ser = pd.Series(np.arange(4), index=['one', 'two', 'three', 'four'])
    print(ser)
    print(ser.reindex(['three', 'four', 'five', 'one']))  # 更换索引以新索引的顺序为准，对于新的标签，以NaN填充值
    print()
    ser = pd.Series([1, 5, 3, 6], index=[0, 3, 5, 6])
    print(ser)
    print(ser.reindex(range(6), method='ffill'))  # 通过自动插值，来重新生成索引，其中ffill表示新增索引的值使用它前一个索引的值;还有bfill
    print()
    # 对于DataFram对象，则可以更换行索引和列
    print(frame1.reindex(range(6), method='ffill', columns=['colors', 'price', 'new', 'object']))
    print()

    print("## 删除功能，注意删除功能返回的是不包含删除部分的 新 对象")
    ser = pd.Series(np.arange(4.), index=['red', 'blue', 'yellow', 'white'])
    print(ser)
    print()
    print(ser.drop('yellow'))  # series对象删除某一行
    print()
    print(ser.drop(['blue', 'white']))
    print()
    frame = pd.DataFrame(np.arange(16).reshape((4, 4)),
                         index=['red', 'blue', 'yellow', 'white'],
                         columns=['ball', 'pen', 'pencil', 'paper'])
    print(frame)
    print()
    print(frame.drop(['blue', 'yellow'], axis=0))  # 传入行索引删除行, axis默认为0
    print()
    print(frame.drop(['pen', 'pencil'], axis=1))  # 删除某一列则需要制定轴为1，按照1轴的标签删除
    print()

    print("## 使用Numpy的通用函数操作两种数据结构")
    frame = pd.DataFrame(np.arange(16).reshape((4, 4)),
                         index=['red', 'blue', 'yellow', 'white'],
                         columns=['ball', 'pen', 'pencil', 'paper'])
    print(np.sqrt(frame))
    print()

    print("## 按行或按列应用函数")
    f = lambda x: x.max() - x.min()
    print(frame.apply(f))  # 默认axis=0，即对行应用函数
    print()
    print(frame.apply(f, axis=1))  # 指定按列应用函数
    g = lambda x: pd.Series([x.min(), x.max()], index=['min', 'max'])  # 我们还可以通过自定义返回各种复合队形的函数来让apply返回对应的对象
    print(frame.apply(g))
    print()

    print("## 统计函数, 在不指定axis的情况下，所有的统计函数默认使用0")
    print(frame.sum())
    print()
    print(frame.mean())
    print()
    print(frame.describe())  # describe返回以axis=0应用的多个统计函数结果的DataFrame对象
    print()

    print("##Series的索引排序")
    ser = pd.Series([5, 0, 3, 8, 4], index=['red', 'blue', 'yellow', 'white', 'green'])
    print()
    print(ser)
    print()
    print(ser.sort_index())  # 对于字符串类型的默认按照升序排列
    print()
    print(ser.sort_index(ascending=False))  # 通过指定asc参数为false来降序排列
    print()

    print("## DataFrame的索引排序")
    frame = pd.DataFrame(np.arange(16).reshape((4, 4)),
                         index=['red', 'blue', 'yellow', 'white'],
                         columns=['ball', 'pen', 'pencil', 'paper'])
    print(frame)
    print()
    print(frame.sort_index())  # 按照轴0进行排序
    print()
    print(frame.sort_index(axis=1))  # 按照轴1进行排序
    print()

    print("## Series和DataFrame元素排序")
    print(ser.sort_values())
    print()
    print(frame.sort_values(by=['pen', 'pencil']))  # 对于frame，要指定根据哪一个元素行（或者列）进行排序
    print(frame.sort_values(by=['red'],  axis=1, ascending=False))
    print()

    print("## 排名次次操作")
    ser['blue'] = 3
    print(ser)
    print()
    print(ser.rank())  # 将对象中的数字按照大小赋予名次, 但是整体的顺序不变，对于大小相同的数字有4中不同的名次赋予方法, 默认使用average
    print()
    print(ser.rank(method='first'))  # 使用first策略则两个大小相同的数字的名次由它在Serise中的位置决定，在前面的则名次靠前

    print("## Series对象相关性与协方差的计算")
    ser1 = pd.Series(np.random.random(6), np.arange(6))
    ser2 = pd.Series(np.random.random(6), np.arange(6))
    print(ser1)
    print(ser2)
    print()
    print(ser1.corr(ser2))  # 相关性
    print(ser1.cov(ser2))  # 协方差

    print("## DataFrame对象相关性和协方差的计算")
    frame = pd.DataFrame([[1, 4, 3, 6],[4, 5, 6, 1], [3, 3, 1, 5], [4, 1, 6, 4]],
                         index=['red', 'blue', 'yellow', 'white'],
                         columns=['ball', 'pen', 'pencil', 'paper'])
    print(frame)
    print(frame.corr())
    print(frame.cov())
    print()

    print("## 使用corrwith方法计算DataFrame对象的行或列与Series随想或者其他DataFrame元素之间的相关性")
    ser = pd.Series([5, 0, 3, 8], index=['red', 'blue', 'yellow', 'white'])
    print(frame.corrwith(ser))
    print()

    print("## 为元素赋NaN值")
    ser = pd.Series([0, 1, 2, None, 9], index=['red', 'blue', 'yellow', 'white', 'green'])  # 还可以使用np.nan或者np.NaN
    print(ser)
    print()

    print("## Series过滤NaN元素")
    print(ser.dropna())
    print()
    print(ser[ser.notnull()])  # 当然，使用前面用到的notnull方法也是一样的
    print()

    print("## DataFrame过滤NaN元素")
    frame = pd.DataFrame([[6, np.nan, 6], [np.nan, np.nan, np.nan], [2, np.nan, 5]],
                         index=['blue', 'green', 'red'],
                         columns=['ball', 'mug', 'pen'])
    print(frame)
    print()
    print(frame.dropna())  # 当不指定how时，只要行或者列有一个NaN元素，该行该列所有元素都会被删除
    print()
    print(frame.dropna(how='all')) # 指定how为all的时候，只有行或者列全部为NaN时候才删除
    print()

    print("## 为NaN元素填充其他值")
    frame = pd.DataFrame([[6, np.nan, 6], [np.nan, np.nan, np.nan], [2, np.nan, 5]],
                         index=['blue', 'green', 'red'],
                         columns=['ball', 'mug', 'pen'])
    frame_other = pd.DataFrame(frame.copy())
    print(frame)
    print()
    print(frame.fillna(0))  # 若不指定具体的行或者列的名字，则所有空值均置为0
    print()
    print(frame_other.fillna({'ball': 1, 'mug': 0, 'pen': 99}))
    print()

    print("## 分级索引######")
    ser = pd.Series(np.random.random(8),
                    index=[['white', 'white', 'white', 'blue', 'blue', 'red', 'red', 'red'],
                           ['up', 'down', 'right', 'up', 'down', 'up', 'down', 'left']])
    print(ser)
    print()
    print(ser['white'])  # 取一级索引为white的元素
    print()
    print(ser[:, 'up'])  # 取二级索引为up的元素
    print()
    print(ser['white', 'up'])  # 取某一特定元素
    print()
    print(ser.unstack())  # 将含有两级索引的series对象转换为DataFrame对象，其中第二级索引作为column_name, 对应的没有的数据使用NaN填充
    print()
    frame = pd.DataFrame([[6, np.nan, 6], [np.nan, np.nan, np.nan], [2, np.nan, 5]],
                         index=['blue', 'green', 'red'],
                         columns=['ball', 'mug', 'pen'])
    print(frame)
    print()
    print(frame.stack())  # 将DataFrame对象转换为Series对象
    print()
    frame = pd.DataFrame(np.random.random(16).reshape(4, 4),
                         index=[['white', 'white', 'red', 'red'], ['up', 'down', 'up', 'down']],
                         columns=[['pen', 'pen', 'paper', 'paper'], [1, 2, 1, 2]])  # 可以为frame的行与列都定义二级索引
    print(frame)
    print()

    print("## 调整各个索引的层级顺序，根据摸个层级排序")
    frame = pd.DataFrame(np.random.random(16).reshape(4, 4),
                         index=[['white', 'white', 'red', 'red'], ['up', 'down', 'up', 'down']],
                         columns=[['pen', 'pen', 'paper', 'paper'], [1, 2, 1, 2]])  # 可以为frame的行与列都定义二级索引
    print(frame)
    print()
    frame.columns.names = ['object', 'id']
    frame.index.names = ['colors', 'status']  # 在对索引层级进行调整时，需要对索引本身命名
    print(frame)
    print()
    print(frame.swaplevel('colors', 'status'))  # 使用swaplevel将指定索引的位置交换
    print()
    print(frame.sort_index(level='colors'))  # sortlevel已经deprecated，使用sort_index(level=),两者都是根据某一个层级的索引对数据进行排序，本例根据索引字母序排序
    print()

    print("## 按照索引层级统计数据")
    print(frame.sum(level='colors'))  # pandas很多统计函数均可以指定levels，此处表示轴为0，标签为colors的行进行求和
    print()
    print(frame.sum(level='id', axis=1))  # 若要对某个多级列进行统计，需指定轴为1

















































