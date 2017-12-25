"""
Created on Wed Nov 23 16:20:26 2017

@author: chenkedi(chenkedi@baidu.com)
"""

"""
Copyright 2017 Baidu, Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""

import numpy as np
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
import matplotlib.pyplot as plt
from datetime import timezone
import load_data as ld
from keras.models import Sequential
from keras.layers import Dense
import sklearn as sln
import data_visualize as dv
from keras import metrics
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def model():
    # 使用两个隐层的神经网络，输出层不使用非线性激活函数
    model = Sequential()
    model.add(Dense(5, input_dim=11, kernel_initializer='normal', activation='relu', name='FC1'))
    model.add(Dense(10, input_dim=10, kernel_initializer='normal', activation='relu', name='FC2'))
    model.add(Dense(1, kernel_initializer='normal', name='Regressor'))

    # 使用mean_squared_error最为损失函数，并使用adam优化
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse', 'mae'])
    return model


def feature_normalize(X):
    scaler = StandardScaler().fit(X)
    return scaler.transform(X), scaler


def freature_l2_normalize(X):
    """
    l2正则化，将每个feature缩放到单位范数
    带来的效果是拟合的曲线更为平滑，不容易产生过拟合现象
    """
    return sln.preprocessing.normalize(X, norm='l2')


def datetime_timestamp(x):
    return x.replace(tzinfo=timezone.utc).timestamp()


if __name__ == "__main__":
    # 交叉验证时固定随机数的seed，使得结果可复现
    seed = 7
    epoch = 50
    data_divid = 34000
    data_divided_for_plot = 500
    data_segement_for_predict = 500
    np.random.seed(seed)
    # raw_data = ld.load_data('resources/pump_system.xlsx', 1)
    # ld.pre_process(raw_data)
    raw_data = ld.load_binary_data()
    raw_data = raw_data[raw_data['系统效率'] < 6]
    raw_data = raw_data[raw_data['系统效率'] > 0]
    print(raw_data.shape)  # 42847,25
    feature_domain = ['冷冻总管供水温度', '冷冻总管回水温度', '冷冻总管流量', '室内监测点温度', '室内监测点湿度',
                      '室外监测点温度', '室外监测点湿度', '室外监测点光照度', '冷冻泵总状态', '舒适度', 'control_level']
    objective_domain = '系统效率'

    X_train, scaler = feature_normalize(raw_data[:data_divid][feature_domain].values)
    Y_train = raw_data[:data_divid][objective_domain].values

    X_test = scaler.transform(raw_data[data_divid:][feature_domain].values)
    Y_test = raw_data[data_divid:][objective_domain].values

    print(X_train.shape)
    print(Y_train.shape)
    # print(X_train)
    # print(X_test.shape)
    # print(Y_test.shape)
    # print(X_test)

    estimator = model()
    estimator.load_weights('result/80%train-with-mean-normalize-50-epochs-0.509')
    # estimator.fit(X_train, Y_train, batch_size=32, epochs=epoch, verbose=2)
    plot_model(estimator, to_file='model.png')
    SVG(model_to_dot(estimator).create(prog='dot', format='svg'))
    print(estimator.evaluate(X_test, Y_test))
    # estimator.save_weights('result/80%train-with-mean-normalize-50-epochs-0.509')

    # 模型在测试集上的表现
    Y_test_pred = estimator.predict(X_test)
    X_test_timestamp = raw_data[data_divid:]['DateTime'].apply(lambda x: datetime_timestamp(x)).values
    # X_test_timestamp = raw_data.ix[data_divid:].index.values
    plt.scatter(X_test_timestamp, Y_test)
    plt.plot(X_test_timestamp, Y_test_pred, color='red')

    plt.show()

    # dv.visualize_test_data(estimator, raw_data.ix[data_divid:], X_test, Y_test, 1000, 8)


    # 模型在训练集上的表现
    # Y_train_pred = estimator.predict(X_train)
    # X_train_timestamp = raw_data[:data_divid]['DateTime'].apply(lambda x: datetime_timestamp(x)).values
    # plt.plot(X_train_timestamp, Y_test)
    # plt.plot(X_train_timestamp, Y_train_pred, color='red')
    # plt.show()
