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

import os
import sys
import argparse
import logging
import time

import crawler_thread
import global_config
import logger


class MiniSpider:
    """
    爬虫主类
    使用多进程并行爬取种子文件指定的顶级域名
    使用多线程并发爬取一个种子域名内部指定depth的页面
    """

    def __init__(self, config_path):
        self.config_path = config_path
        self.conf = global_config.GlobalConfig(config_path)
        self.seeds = self.read_seeds()
        self.thread_pool = []

    def read_seeds(self):
        """
        将配置文件中给定的种子文件目内所有的种子url读入爬虫待爬取队列
        :return: 返回读取种子文件后生成的种子url list
        """
        seeds = []
        for filename in os.listdir(self.conf.url_list_file):
            with open(f'{self.conf.url_list_file}/{filename}') as file:
                for line in file:
                    seeds.append(line.strip())
        return seeds

    def traverse_seed(self, seed_url):
        """
        广度遍历一个seed指定depth的所有面
        :param seed_url: 初始种子url链接
        """
        url_queue = list(seed_url)
        visited_url = set()
        depth = 0
        # 爬取指定的深度
        while url_queue and depth < self.conf.max_depth:
            depth += 1
            logging.info(f'current depth is {depth}')
            # 获取当前url_queue中第一个url内的所有链接
            logging.info(f'=======url queue length before extend: {len(url_queue)}')
            self.get_all_links(url_queue, visited_url)
        logging.info(f'Sub process for craw top level seed {seed_url} complete')
        exit()

    def get_all_links(self, url_queue, visited_url):
        """
        抓取url队列中同一层级的url源码，并将新增下一层级的url扩增到到url_queue
        在扩展下一层url时，为了防止url_queue集合过大导致内存不足，每次只对队列中的一个url进行扩增，而后执行爬取
        :return:
        """
        initial_level_length = len(url_queue)
        for i in range(initial_level_length):
            self.multi_thread_retriver(url_queue, visited_url, self.get_link, 1, 1)
            logging.info(f'=======url queue length after extend: {len(url_queue)}')
            self.download_all(url_queue, visited_url)

    def download_all(self, url_queue, visited_url):
        """
        以指定的thread数量并发遍历url_queue中的所有链接
        并发遍历完成后，清空url_queue
        :param url_queue: 带爬取的url队列
        :param visited_url: 已爬取的url队列，用于避免重复爬取
        :return:
        """
        download_queue = []
        not_download_url = []
        # 从当前url_queue中筛选符合下载pattern的连接开启多线程下载；更新url_queue为不需要下载的链接
        for url in url_queue:
            if crawler_thread.CrawlerThread.check_target_url(self.conf.target_url, url):
                download_queue.append(url)
            else:
                not_download_url.append(url)
        # 对要下载的连接去重
        downloaded_set = set(download_queue)
        self.multi_thread_retriver(list(downloaded_set), visited_url, self.download, self.conf.thread_count,
                                   sys.maxsize)
        url_queue.clear()
        url_queue.extend(not_download_url)

    def multi_thread_retriver(self, url_queue, visited_url, act_func, thread_limit, queue_pop_limit):
        """
        限制给定数量的线程来并发处理下载页面或扩增url的操作
        :param url_queue: 待爬取的url队列
        :param act_func: 指定要进行操作的函数
        :param visited_url: 已爬取的url队列，用于避免重复爬取
        :param thread_limit: 并发下载的线程限制
        :param queue_pop_limit: 当前批次并发从url_queue中下载url数量的最大数量
        :return:
        """
        # downloade_count + thread_count = 当前遍历到的url_queue的位置
        downloaded_count = 0
        # initial length 记录url_queue在当前层级的url数量(在对url进行下一层级的连接抓取时，也会加入此队列），以实现广度优先遍历
        initial_length = len(url_queue)
        while downloaded_count < initial_length and downloaded_count < queue_pop_limit:
            thread_count = 0
            while thread_count < thread_limit and downloaded_count + thread_count < initial_length:
                act_func(url_queue[downloaded_count + thread_count],
                         self.conf.output_directory, thread_count, url_queue, visited_url)
                thread_count += 1
            downloaded_count += thread_count

            # 线程数量达到上限则等待该批url爬取完成
            for thread in self.thread_pool:
                # 设置每个线程的超时时间
                thread.join(self.conf.crawl_timeout)
            # 每一批线程处理完毕休眠指定的interval
            time.sleep(self.conf.crawl_interval)

        # 删除广度优先遍历完的本层级的所有url
        del url_queue[0:downloaded_count]
        self.thread_pool.clear()

    def get_link(self, url, download_path, tid, url_queue, visited_url):
        """
        启动一个crawler_thread来抓取当前url中的所有链接, 加入url_queue
        :param url: 待爬取的url
        :param download_path: url下载到本地的路径
        :param tid: 线程ID
        :param url_queue: 待爬取的url队列
        :param visited_url: 已爬取的url队列，用于避免重复爬取
        """
        thread = crawler_thread.CrawlerThread(url, tid, url_queue, visited_url, self.conf.target_url, download_path,
                                              mode='extract')
        self.thread_pool.append(thread)
        thread.start()

    def download(self, url, download_path, tid, url_queue, visited_url):
        """
        启动一个crawler_thread来下载符合正则表达式要求url对应的内容
        :param url: 待爬取的url
        :param download_path: url下载到本地的路径
        :param tid: 线程ID
        :param url_queue: 待爬取的url队列
        :param visited_url: 已爬取的url队列，用于避免重复爬取
        """
        thread = crawler_thread.CrawlerThread(url, tid, url_queue, visited_url, self.conf.target_url, download_path)
        self.thread_pool.append(thread)
        thread.start()

    def start(self):
        """
        启动爬虫
        """
        # # 进程池中的进程的个数最大为配置文件中指定的process_count, 在不超过该数值的前提下，为每一个种子开启一个进程
        # process_pool_size = self.conf.process_count if self.conf.process_count < len(self.seeds) else len(self.seeds)
        # process_pool = multiprocessing.Pool(processes=process_pool_size)
        # # 多进程并行爬取每个种子
        # # 每个种子内部使用多线程并发处理抓取和存储等IO密集型任务
        # process_pool.map(self.traverse_seed, self.seeds)
        # process_pool.terminate()
        # process_pool.join()
        # logging.info(f'All subprocess of top level seeds complete')
        self.traverse_seed(self.seeds)


if __name__ == '__main__':
    # 初始化符合百度log规范的logger,将log输出到同级目录log文件夹下的mini_spider.log
    logger.init_log("./log/mini_spider")

    arger_parser = argparse.ArgumentParser()
    arger_parser.add_argument('-c', help='set spider_test.conf file path', dest='config_path', required=True)
    arger_parser.add_argument('-v', help='set spider version', dest='version')

    args = arger_parser.parse_args()
    spider = MiniSpider(args.config_path)
    spider.start()
