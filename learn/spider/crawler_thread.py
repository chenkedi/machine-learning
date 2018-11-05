from threading import Thread, Lock
from urllib import request, parse
from bs4 import BeautifulSoup
from re import compile
import logging


class CrawlerThread(Thread):
    """
    定义"抓取"和"存储"指定URL过程的线程
    以便使用多线程对IO密集型任务加速
    """
    threadLock = Lock()
    logging.getLogger('crawler_test')

    def __init__(self, url, tid, url_queue, visited_url, target_url_pattern, filepath=None, mode='save'):
        """
        :param url: 目标网页的URl
        :param filepath: 目标网页的存储地址
        :param tid: 线程id
        :param mode: 决定该线程执行的操作：1. 直接存储目标url；2. 读取目标url，并返回html源码中所有未爬取的url
        """
        super().__init__()
        self.url = url
        self.pattern = target_url_pattern
        if mode == 'save' and (not filepath):
            raise ValueError("param filepath must be set in mode 'save' !")
        self.filepath = filepath
        self.tid = tid
        self.mode = mode
        self.url_queue = url_queue
        self.visited_url = visited_url
        self.domain = parse.urlparse(self.url).netloc
        self.scheme = parse.urlparse(self.url).scheme

    def run(self):
        if self.mode == 'extract':
            self._extract_new_url()
        elif self.mode == 'save':
            self._save_url()
        else:
            raise ValueError('Getting mode param which is not supported ')

    def _save_url(self):
        try:
            logging.info(f'Thread {self.tid} is retrieving {self.url} .....')
            filname = self.file_name_fix(self.url)
            request.urlretrieve(self.url, f'{self.filepath}/{filname}')
        except Exception as e:
            logging.error(f'Thread {self.tid} retrieving {self.url} failed for {e}')
            exit()
        else:
            # 记录已访问的set为多线程共享，需进行同步
            CrawlerThread.threadLock.acquire()
            self.visited_url.add(self.url)
            CrawlerThread.threadLock.release()

    def _extract_new_url(self):
        try:
            logging.info(f'Thread {self.tid} is parsing {self.url} .....')
            html = request.urlopen(self.url)
            bsobj = BeautifulSoup(html.read(), "lxml")
        except Exception as e:
            logging.error(f'Thread {self.tid} parsing {self.url} failed for {e}')
            exit()
        else:
            # 记录已访问的set为多线程共享，需进行同步
            CrawlerThread.threadLock.acquire()
            self.visited_url.add(self.url)
            CrawlerThread.threadLock.release()

            # 获取当前网页源码中所有具有src属性和href属性的标签,并将其中的链接加入到new_url_list集合
            new_url_list = set()
            for src_tag in bsobj.findAll(src=True):
                # 扩增url时，去除已访问的url
                if 'src' in src_tag.attrs and src_tag['src'] not in self.visited_url:
                    if self.url_filter(src_tag['src']):
                        new_url_list.add(self.url_fix(src_tag['src']))

            for href_tag in bsobj.findAll(href=True):
                if 'href' in href_tag.attrs and href_tag not in self.visited_url:
                    if self.url_filter(href_tag['href']):
                        new_url_list.add(self.url_fix(href_tag['href']))

            logging.info(f'Thread {self.tid} is adding all links of {self.url} to url queue')

            # 对多个线程需要并发写入的url队列加锁同步
            CrawlerThread.threadLock.acquire()
            self.url_queue.extend(new_url_list)
            CrawlerThread.threadLock.release()
            logging.info(
                f'After Thread {self.tid} added all links of {self.url}, url queue length is {len(self.url_queue)}')

    def url_filter(self, url):
        """
        在进行url扩增时，决定哪些url会加入url_queue(站点内链，包含绝对链接与相对链接，或者符合url_pattern的才能加入url_queue)
        :param url:
        :return:
        """

        # 对于符合下载pattern的页面，无论是否为外链，均需加入url队列
        # 筛除站点中的外链
        # 保留站内绝对链接和相对链接
        url = self.url_fix(url)
        if CrawlerThread.check_target_url(self.pattern, url):
            return True
        # 筛除外链,这一句必须在上一句之后，因为符合url_pattern的链接有可能不包含domian(如cdn缓存等）
        if self.domain not in url and (url.startswith('http://')
                                       or url.startswith('https://') or url.startswith('www.')):
            return False
        # 相对链接也要加入队列
        return True

    def url_fix(self, url):

        if url.startswith('http://') or url.startswith('https://'):
            url_fixed = url
        elif url.startswith('www.'):
            url_fixed = f'{self.scheme}://{url}'
        elif url.startswith('//'):
            url_fixed = f'{self.scheme}:{url}'
        elif url.startswith('/'):
            # 绝对地址
            url_fixed = f'{self.scheme}://{self.domain}{url}'
        else:
            # 相对地址
            url_fixed = f'{self.url}/{url}'
        return url_fixed

    def file_name_fix(self, filename):
        """
        处理以url为文件名存储文件时的路径问题
        :param filename:
        :return:
        """
        return filename.replace('/', '-')

    @staticmethod
    def check_target_url(pattern, url):
        """
        检测url是否为配置文件指定pattern的 url
        :param url:
        :return: boolean
        """
        reg = compile(pattern)
        return reg.match(url)
