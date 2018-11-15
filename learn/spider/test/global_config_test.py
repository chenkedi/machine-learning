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

import unittest

from global_config import GlobalConfig


class TestGlobalConfig(unittest.TestCase):

    def test_ini(self):
        conf = GlobalConfig('./spider_test.conf')
        # conf = GlobalConfig('./nowhere')
        self.assertEqual(conf.parser_path, './spider_test.conf')
        self.assertEqual(conf.url_list_file, './test')
        self.assertEqual(conf.output_directory, './test_output')
        self.assertEqual(conf.max_depth, 2)
        self.assertEqual(conf.crawl_interval, 1)
        self.assertEqual(conf.crawl_timeout, 2)
        self.assertEqual(conf.target_url, '.*.(gif|png|jpg|bmp)$')
        self.assertEqual(conf.thread_count, 9)
        self.assertEqual(conf.process_count, 10)

    def test_parse(self):
        with self.assertRaises(FileNotFoundError):
            # 给定不存在的配置文件
            conf = GlobalConfig('./nowhere.conf')

    def test_read(self):
        with self.assertRaises(ValueError):
            # 在配置文件应该给予整数的字段给出小数，期待得到读取整数异常
            conf = GlobalConfig('./spider_test_read_int.conf')