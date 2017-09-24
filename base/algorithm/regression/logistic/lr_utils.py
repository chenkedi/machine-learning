import numpy as np
import h5py
from pandas.io.pytables import HDFStore

if __name__ == '__main__':
    test = h5py.File('resources/test_catvnoncat.h5')
    a = test["test_set_x"]
    print(a[1])


def load_dataset():
    """
    读取h5格式存储的图片为（图片个数，图片高度，图片宽度，3）维度的np array
    :return:
    """
    train_dataset = h5py.File('resources/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  # your train set labels

    test_dataset = h5py.File('resources/test_catvnoncat.h5', "r")
    # 这里的第一个中括号表示取得HDF5对象，但由于原始数据没有字符串作为对象键，所以默认以数字最为键，：表示取所有的键
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  # your test set labels

    classes = np.array(test_dataset["list_classes"][:])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))  # 将 (m,)类型的向量变为（1, m)的向量
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes