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
import pandas as pd
import dateutil.parser
from pandas.io.pytables import HDFStore


def load_data(path, sheet_index):
    return pd.read_excel(path, sheet_index)


def comfort_transform(x):
    if x == '舒适':
        return 1
    elif x == '过冷':
        return -1
    else:
        return 2


def control_level_transform(x):
    if x < dateutil.parser.parse('2017-07-17 10:43:00'):
        return 5.0
    elif dateutil.parser.parse('2017-07-17 10:43:00') <= x < dateutil.parser.parse('2017-07-18 9:00:00'):
        return 3.0
    elif dateutil.parser.parse('2017-07-18 9:00:00') <= x < dateutil.parser.parse('2017-08-2 10:50:00'):
        return 2.5
    else:
        return 2.8


def pre_process(raw_data):
    raw_data['舒适度'] = raw_data['舒适度'].apply(lambda x: comfort_transform(x))  # 舒适度列 枚举转数值
    raw_data['control_level'] = raw_data['DateTime']  # 新增控制水平列
    raw_data['control_level'] = raw_data['control_level'].apply(
        lambda x: control_level_transform(x))  # 按照时间戳将控制水平转换为客户对应的设置值
    raw_data['冷冻泵总状态'] = raw_data['1#冷冻泵运行状态'] + raw_data['2#冷冻泵运行状态'] + raw_data['3#冷冻泵运行状态']  # 冷冻泵总运行个数求和
    return raw_data

def load_feature():
    store = HDFStore('resources/pump_feature.h5')
    return store['X'], store['Y']


def write_feature(X_dataframe, Y_dataframe):
    store = HDFStore('resources/pump_feature.h5')
    store['X'] = X_dataframe
    store['Y'] = Y_dataframe

def write_binary_data(dataframe):
    store = HDFStore('resources/pump_system.h5')
    store['raw'] = dataframe

def load_binary_data():
    store = HDFStore('resources/pump_system.h5')
    return store['raw']

if __name__ == "__main__":
    path = 'resources/pump_system.xlsx'
    raw_data = load_data(path, 1)
    pre_process(raw_data)
    write_binary_data(raw_data)

    # print(raw_data.dtypes)
    # print(raw_data)
