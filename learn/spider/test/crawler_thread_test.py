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
import mock

import crawler_thread

url = 'https://news.sina.com.cn'
tid = 1
url_queue = []
visited_queue = set()
target_url_pattern = '.*.(gif|png|jpg|bmp)$'
filepath = './'
mode = 'save'
logger_name = 'root'


class TestCrawlerThread(unittest.TestCase):

    def test_ini(self):
        crawler = crawler_thread.CrawlerThread(url, tid, url_queue, visited_queue, target_url_pattern, filepath, mode)
        self.assertEqual(crawler.url, url)
        self.assertEqual(crawler.tid, tid)
        self.assertEqual(crawler.url_queue, url_queue)
        self.assertEqual(crawler.pattern, target_url_pattern)
        self.assertEqual(crawler.filepath, filepath)
        self.assertEqual(crawler.mode, mode)

    @mock.patch('crawler_thread.CrawlerThread._save_url')
    def test_run_with_save_mode(self, mock_save_url):
        # 给定mode为save，期望执行网页下载的线程，通过mock方法断言调用过程
        crawler = crawler_thread.CrawlerThread(url, tid, url_queue, visited_queue, target_url_pattern, filepath, mode)
        crawler.run()
        self.assertTrue(mock_save_url.called)

    @mock.patch('crawler_thread.CrawlerThread._extract_new_url')
    def test_run_with_extract_mode(self, mock_extract_mode):
        # 给定mode为extract，期望执行网页下载的线程，通过mock方法断言调用过程
        crawler = crawler_thread.CrawlerThread(url, tid, url_queue, visited_queue, target_url_pattern, filepath,
                                                   'extract')
        crawler.run()
        self.assertTrue(mock_extract_mode.called_once)

    def test_run_with_not_support_mode(self):
        # 给定线程的mode为不支持的mode，期望抛出异常
        with self.assertRaises(ValueError):
            crawler = crawler_thread.CrawlerThread(url, tid, url_queue, visited_queue, target_url_pattern, filepath,
                                                   'some other mode')
            crawler.run()

    def test_check_target_url(self):
        self.assertEqual(
            crawler_thread.CrawlerThread.check_target_url(target_url_pattern, "//news.a-cdn.com/3094u5.jpg")[0],
            "//news.a-cdn.com/3094u5.jpg")
        self.assertEqual(
            crawler_thread.CrawlerThread.check_target_url(target_url_pattern, "//news.a-cdn.com/3094u5.html"), None)

    def test_file_name_fix(self):
        crawler = crawler_thread.CrawlerThread(url, tid, url_queue, target_url_pattern, filepath, 'extract')
        self.assertEqual(crawler.file_name_fix('//news.a-cdn.com/3094u5.jpg'), '--news.a-cdn.com-3094u5.jpg')

    def test_url_filter(self):
        crawler = crawler_thread.CrawlerThread(url, tid, url_queue, visited_queue, target_url_pattern, filepath,
                                               'extract')
        self.assertEqual(crawler.url_filter("//news.a-cdn.com/3094u5.jpg"), True)
        self.assertEqual(crawler.url_filter("//news.sina.com.cn/346slkdfj.html"), True)
        self.assertEqual(crawler.url_filter("auto/camery.html"), True)
        self.assertEqual(crawler.url_filter("https://news.baidu.com/slkljg.html"), False)
