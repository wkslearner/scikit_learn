
import pandas as pd
import numpy as np

data = pd.read_excel('test_baidu.xlsx')

import json
from urllib.request import urlopen, quote
import requests


def getlnglat(address):
    url = 'http://api.map.baidu.com/geocoder/v2/'
    output = 'json'
    ak = '你的百度地图ak' # 百度地图ak，具体申请自行百度，提醒需要在“控制台”-“设置”-“启动服务”-“正逆地理编码”，启动
    address = quote(address) # 由于本文地址变量为中文，为防止乱码，先用quote进行编码
    uri = url + '?' + 'address=' + address  + '&output=' + output + '&ak=' + ak
    req = urlopen(uri)
    res = req.read().decode()
    temp = json.loads(res)
    lat = temp['result']['location']['lat']
    lng = temp['result']['location']['lng']
    return lat,lng   # 纬度 latitude   ，   经度 longitude  ，

    for indexs in data.index:
        get_location = getlnglat(data.loc[indexs,'圈定区域'])
        get_lat = get_location[0]
        get_lng = get_location[1]
        data.loc[indexs,'纬度'] = get_lat
        data.loc[indexs,'经度'] = get_lng

data_html = pd.DataFrame(columns=['content'])

for indexs in data.index:
    data_html.loc[indexs,'content'] = '{' + \
 '"lat":' + str(data.loc[indexs,'纬度']) + ',' +  \
 '"lng":' + str(data.loc[indexs,'经度']) + ',' +  \
 '"quyu":' + '"' + str(data.loc[indexs,'圈定区域']) +'"' +   \
 '}' + ','


data_html.to_csv ("data_html.csv",encoding="gbk")





