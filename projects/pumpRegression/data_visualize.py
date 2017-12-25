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

import load_data as ld
import matplotlib.pyplot as plt
import nn_model as nn


def visualize_test_data(estimator, raw_data, X_test, Y_test,  segement, num):
    X = raw_data['DateTime'].apply(lambda x: nn.datetime_timestamp(x)).values

    for batch in range(num):
        print(batch * segement, '~', (batch + 1) * segement)
        X_raw_segement = X[batch * segement:(batch + 1) * segement]
        X_test_segment = X_test[batch * segement:(batch + 1) * segement]

        Y_segement = Y_test[batch * segement:(batch + 1) * segement]
        Y_pred_segement = estimator.predict(X_test[batch * segement:(batch + 1) * segement])
        plt.figure(str(batch * segement) + '~' + str((batch + 1) * segement))
        plt.scatter(X_raw_segement, Y_segement)
        plt.plot(X_test_segment, Y_pred_segement)
    plt.show()


def visualize_customized_range_data(raw_data, start, figure_num, size):
    # X = raw_data['DateTime'].apply(lambda x: nn.datetime_timestamp(x)).values
    X = raw_data.index
    Y = raw_data['系统效率'].values

    for batch in range(figure_num):
        print(start, '~', start + size)
        plt.figure(str(start) + '~' + str(start + size))
        plt.scatter(X[start: start + size], Y[start: start + size])
        start +=  size
    plt.show()

def visualize_all_data(raw_data):
    X = raw_data['DateTime'].apply(lambda x: nn.datetime_timestamp(x)).values
    Y = raw_data['系统效率'].values
    plt.figure('all')
    plt.scatter(X, Y)
    plt.show()



if __name__ == "__main__":
    raw_data = ld.load_binary_data()
    # raw_data2 = ld.load_data('resources/pump_system.xlsx', 1)
    raw_data = raw_data[raw_data['系统效率'] < 6]
    raw_data = raw_data[raw_data['系统效率'] > 0]
    print(raw_data[42800:42800])
    print(raw_data[42800:43000])
    # visualize_customized_range_data(raw_data, 42000, 5, 200)
    visualize_all_data(raw_data)
