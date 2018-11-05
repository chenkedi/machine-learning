from configparser import ConfigParser
from sys import exit
import logging as log


class GlobalConfig:
    """
    Config reader of spider

    url_list_file: ./urls ; 种子文件路径
    output_directory: ./output ; 抓取结果存储目录
    max_depth: 1 ; 最大抓取深度(种子为0级)
    crawl_interval: 1 ; 抓取间隔. 单位: 秒
    crawl_timeout: 1 ; 抓取超时. 单位: 秒
    target_url: .*.(gif|png|jpg|bmp)$ ; 需要存储的目标网页URL pattern(正则表达式)
    thread_count: 8 ; 抓取线程数
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

        self.parser = ConfigParser()
        self._parse()

    def _parse(self):
        """
        load config file
        :return:
        """
        data = self.parser.read(self.parser_path)
        if len(data) < 1:
            raise FileNotFoundError("Can not find spider config file, exit ...")
        else:
            self._read()

    def _read(self):
        """
        read and check some config value
        :return:
        """
        section = 'spider'
        try:
            self.max_depth = self.parser.getint(section, 'max_depth')
            self.crawl_interval = self.parser.getint(section, 'crawl_interval')
            self.crawl_timeout = self.parser.getint(section, 'crawl_timeout')
            self.thread_count = self.parser.getint(section, 'thread_count')
            self.process_count = self.parser.getint(section, 'process_count')
        except Exception as er:
            log.error(f'parsing numeric config items to integer error, exit....{er}')
            raise ValueError(f'parsing numeric config items to integer error, exit....{er}')
        else:
            self.url_list_file = self.parser.get(section, 'url_list_file') \
                if self.parser.get(section, 'url_list_file') else self.url_list_file
            self.output_directory = self.parser.get(section, 'output_directory') \
                if self.parser.get(section, 'output_directory') else self.output_directory
            self.target_url = self.parser.get(section, 'target_url') \
                if self.parser.get(section, 'target_url') else self.target_url
