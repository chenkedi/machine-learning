#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2018/11/15 下午4:40
# @Author  : chenkedi
# @Email   : chenkedi@baidu.com

"""
Copyright 2018 Baidu, Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""

import setuptools as st

st.setup(
    name="mini_spider",
    version="0.1.0",
    description="A simple mini spider",
    author="chenkedi",
    author_email="chenkedi@baidu.com",
    license="MIT",
    packages=st.find_packages(),
    install_requires=[
        "requests>=2.14.2",
        "beautifulsoup4>=4.6"
    ]
)