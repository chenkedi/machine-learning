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

import configparser
import logging


class GlobalConfig:
    """
    爬虫配置文件读取与校验, 配置文件包含以下几个项目：
    url_list_file: 默认值为./urls，指定初始种子文件的路径，支持读取多个种子文件
    output_directory: 默认值为./output, 指定抓取结果的存储目录，若目录不存在，则爬虫会新建该目录
    max_depth: 默认值为1，必须为整数, 指定爬虫的最大爬取深度，初始种子深度为0
    crawl_interval: 默认值为1, 必须为整数，指定爬虫在多线程抓取符合target_url的页面时的间隔，单位为秒
    crawl_timeout: 默认值为1, 必须为整数，指定爬虫抓取任意页面时，请求响应的超时时间，单位为秒
    target_url: 默认值为.*.(gif|png|jpg|bmp)$, 指定爬虫存储的目标网页URL的正则表达式筛选规则
    thread_count: 默认值为8，指定爬虫并发下载target_url时的最大线程数
    """

    def __init__(self, config_path):
        self.parser_path = config_path
        # setting default config
        self.url_list_file = './urls'
        self.output_directory = './output'
        self.max_depth = 10
        self.crawl_interval = 1
        self.crawl_timeout = 1
        self.target_url = '.*.(gif|png|jpg|bmp)'
        self.thread_count = 8
        self.process_count = 4

        self.parser = configparser.ConfigParser()
        self._parse()

    def _parse(self):
        """
        读取并解析配置文件，若发现解析后的配置文件list小于1，则判断为无法找到配置文件
        """
        data = self.parser.read(self.parser_path)
        if len(data) < 1:
            raise FileNotFoundError("Can not find spider config file, exit ...")
        else:
            self._read()

    def _read(self):
        """
        检测配置文件中的字段，并对未提供的字段使用默认值
        """
        section = 'spider'
        try:
            self.max_depth = self.parser.getint(section, 'max_depth')
            self.crawl_interval = self.parser.getint(section, 'crawl_interval')
            self.crawl_timeout = self.parser.getint(section, 'crawl_timeout')
            self.thread_count = self.parser.getint(section, 'thread_count')
            self.process_count = self.parser.getint(section, 'process_count')
        except Exception as er:
            logging.error(f'parsing numeric config items to integer error, exit....{er}')
            raise ValueError(f'parsing numeric config items to integer error, exit....{er}')
        else:
            self.url_list_file = self.parser.get(section, 'url_list_file') \
                if self.parser.get(section, 'url_list_file') else self.url_list_file
            self.output_directory = self.parser.get(section, 'output_directory') \
                if self.parser.get(section, 'output_directory') else self.output_directory
            self.target_url = self.parser.get(section, 'target_url') \
                if self.parser.get(section, 'target_url') else self.target_url
