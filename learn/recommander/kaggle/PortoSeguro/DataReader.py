"""
A data parser for Porto Seguro's Safe Driver Prediction competition's dataset.
URL: https://www.kaggle.com/c/porto-seguro-safe-driver-prediction
"""
import pandas as pd


class FeatureDictionary(object):
    def __init__(self, trainfile=None, testfile=None,
                 dfTrain=None, dfTest=None, numeric_cols=[], ignore_cols=[]):
        # 训练数据以文件或者dataFrame的形式传入，不能重复，且同一类数据只能有一种形式
        assert not ((trainfile is None) and (dfTrain is None)), "trainfile  or dfTrain at least one is set"
        assert not ((trainfile is not None) and (dfTrain is not None)), "only one can be set"
        assert not ((testfile is None) and (dfTest is None)), "testfile or dfTest at least one is set"
        assert not ((testfile is not None) and (dfTest is not None)), "only one can be set"
        self.trainfile = trainfile
        self.testfile = testfile
        self.dfTrain = dfTrain
        self.dfTest = dfTest
        self.numeric_cols = numeric_cols
        self.ignore_cols = ignore_cols
        self.gen_feat_dict()

    def gen_feat_dict(self):
        # 根据数据的形式，读取数据为dataframe
        if self.dfTrain is None:
            dfTrain = pd.read_csv(self.trainfile)
        else:
            dfTrain = self.dfTrain
        if self.dfTest is None:
            dfTest = pd.read_csv(self.testfile)
        else:
            dfTest = self.dfTest

        df = pd.concat([dfTrain, dfTest])
        self.feat_dict = {}
        tc = 0
        # 给每个特征生成编号
        for col in df.columns:
            # 根据配置去除列，实际上去除了id，target，和包含calc的列
            # 因此解决了train比test多一列target的情况
            if col in self.ignore_cols:
                continue
            if col in self.numeric_cols:
                # map to a single index
                # 数值特征就是其自身，所以只有一个编号
                self.feat_dict[col] = tc
                tc += 1
            else:
                # category特征需要根据该类别的基数来确定特征编号的范围
                us = df[col].unique()  # 得到该列去重后的所有类别值
                # 一个category特征的编号为一组，每一个具体类别对应一个编号
                self.feat_dict[col] = dict(zip(us, range(tc, len(us) + tc)))
                tc += len(us)
        self.feat_dim = tc


class DataParser(object):
    def __init__(self, feat_dict):
        self.feat_dict = feat_dict

    def parse(self, infile=None, df=None, has_label=False):
        assert not ((infile is None) and (df is None)), "infile or df at least one is set"
        assert not ((infile is not None) and (df is not None)), "only one can be set"
        if infile is None:
            dfi = df.copy()
        else:
            dfi = pd.read_csv(infile)
        if has_label:
            y = dfi["target"].values.tolist()
            dfi.drop(["id", "target"], axis=1, inplace=True)
        else:
            ids = dfi["id"].values.tolist()
            dfi.drop(["id"], axis=1, inplace=True)
        # dfi for feature index
        # dfv for feature value which can be either binary (1/0) or float (e.g., 10.24)
        dfv = dfi.copy()
        for col in dfi.columns:
            if col in self.feat_dict.ignore_cols:
                dfi.drop(col, axis=1, inplace=True)
                dfv.drop(col, axis=1, inplace=True)
                continue
            if col in self.feat_dict.numeric_cols:
                dfi[col] = self.feat_dict.feat_dict[col]
            else:
                # 将每个category特征中的具体类别（如国家类别特征：country='usa‘，编号为56），转化为对应的特征编号
                dfi[col] = dfi[col].map(self.feat_dict.feat_dict[col])
                # 在已知该列的特征编号的情况下，该列的特征就是one-hot编码向量中唯一的1
                # 此处使用稀疏表示法，只需记录one-hot编码后，特征编号为多少的特征是该category的取值即可
                dfv[col] = 1.

        # list of list of feature indices of each sample in the dataset
        Xi = dfi.values.tolist()
        # list of list of feature values of each sample in the dataset
        Xv = dfv.values.tolist()
        if has_label:
            return Xi, Xv, y
        else:
            return Xi, Xv, ids


import inspect
import os
def write_function_to_file(function, file):
    if os.path.exists(file):
        append_write = 'a' # append if already exists
    else:
        append_write = 'w' # make a new file if not
    with open(file, append_write) as file:
        function_definition = inspect.getsource(function)
        file.write(function_definition)
        # print(function_definition)
write_function_to_file(DataParser, 'a.py')

