#一、目录结构

```
--root
    |--config 爬虫配置文件目录
    |--output 爬虫下载文件的存储目录
    |--urls   爬虫种子URL文件目录
    |--doc    说明文档的图片等素材目录
    |--test   单元测试目录
```

#二、架构与流程设计

> 爬虫在满足题目要求的基础上，加入了多进程的支持，每个原始种子（顶级域名）可进行并行爬取（不超过配置文件设置的
    进程数量限制的情况下），同时每个原始种子内部的爬取使用多线程处理IO密集的下载任务，整体架构如下图所示

<img src="http://agroup.baidu.com:8964/static/9e/8f4481388289eeb77d7f275d1dd35e53eefa0b.png?filename=architecture.png" align="center"></img>

> 为了对Python的多线程更为熟练，本次未使用futures.concurrent包中的ThreadExecutorPool线程池，而是自己通过
    继承Thread类，实现了简单的线程上限管理调配。


> 其中每个进程内部爬取，解析，扩展，下载的流程如下图所示


<img src="http://agroup.baidu.com:8964/static/e7/36c91a48ba8b43544696ea78b7f4827d7c8bae.png?filename=process.png" align="center"></img>

#三、运行环境与相关核心依赖
- python3
- requests>=2.14.2
- beautifulsoup4>=4.6

安装对应的依赖提供两种方式：
1. 使用setuptools和本项目中提供的setup.py
```
python setup.py bdist_egg

python setup.py install
```

2. 使用requirements.txt
```
pip install -r requirements.txt
```


#三、爬虫运行方法

在安装相关的依赖后，运行爬虫使用以下命令：
```
python mini_spider.py -c config/spider.conf
```
