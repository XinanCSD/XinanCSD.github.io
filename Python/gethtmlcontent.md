# python 爬虫获取网页 html 内容以及下载附件的方法

python 爬虫获取网页 html 内容的两种方法: 获取静态网页和使用浏览器获取动态内容。

```python
from urllib.request import urlopen
from urllib import request
from bs4 import BeautifulSoup
from urllib.request import urlretrieve
from selenium import webdriver
import socket
import time


def get_static_url_content(url, encoding='utf-8', timeout=socket._GLOBAL_DEFAULT_TIMEOUT):
    '''
    获取静态网页内容
    :param url:         网页url
    :param encoding:    网页编码
    :param timeout:     设置超时
    :return:
    '''
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:23.0) Gecko/20100101 Firefox/23.0'}
    req = request.Request(url, headers=headers)
    html = urlopen(req, timeout=timeout)
    bsObj = BeautifulSoup(html.read(), "html.parser", from_encoding=encoding)
    return bsObj


def get_driver_url_content(url, encoding='utf-8', timeout=3):
    '''
    使用浏览器获取动态内容
    :param url:         网页url
    :param encoding:    网页编码
    :param timeout:     设置超时
    :return:
    '''
    chromedriver_path = '/path/to/chromedriver'
    driver = webdriver.Chrome(executable_path=chromedriver_path)
    # 也可以使用phantomJS
    # driver =webdriver.Phantomjs(executable_path="/path/to/phantomjs")
    driver.get(url)
    time.sleep(timeout)
    bsObj = BeautifulSoup(driver.page_source, 'html.parser', from_encoding=encoding)
    driver.close()
    return bsObj

def load_appendix(url, filename):
    '''
    下载附件
    :param url:         附件 url(附件文档和图片均可)
    :param filename:    保存的文件名
    :return:
    '''
    urlretrieve(url, filename)
```