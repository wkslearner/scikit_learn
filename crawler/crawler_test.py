'''
网页数据爬取
'''

# !/usr/bin/python
# -*- coding: UTF-8 -*-

import urllib.request
from urllib import request
import chardet
from bs4 import BeautifulSoup
import requests
from urllib.parse import urlparse


'''基于urllib的网页信息获取'''
# response = request.urlopen("http://www.baidu.com/")
# html = response.read()
# charset = chardet.detect(html)  # 查询网页格式
# html = html.decode("utf-8")
# print(charset)
# print(html)


'''基于beautifulsoup的网页解析'''
response = requests.get("http://jecvay.com")
# response.text  网页为中文时可能显示乱码
soup = BeautifulSoup(response.content)  # fromEncoding="gb18030"

# print(soup.prettify()) # 打印soup对象内容

# 主体打印
# print(soup.body.text)


"""
Beautiful Soup 将复杂HTML文档转换成一个复杂的树形结构,每个节点都是 Python 对象,
所有对象可以归纳为4种:Tag，NavigableString，BeautifulSoup，Comment
"""

# 标签tag 处于两个<> 符号中间
print(soup.title)
print(soup.title.text)  #提取标题中的文本
# print(soup.head)















# result = urlparse('http://www.baidu.com/index.html;user?id=5#comment')
# print(type(result), result)










