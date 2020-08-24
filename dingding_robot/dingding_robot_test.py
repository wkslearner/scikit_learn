
# root@pts/1 $ cat dingding.py
#!/usr/bin/env python
#-*- coding: utf-8 -*-
#Author: Colin
#Date:
#Desc:
#

import os
import sys
import json
import datetime
import requests

## 钉钉组中创建机器人的时候给出的webhook
## 告警测试
webhook = 'https://oapi.dingtalk.com/robot/send?access_token=dcb6811794604c625e6d4d70ef752da55be4662e32f0333b1b5f2996b7991a06'

## 定义接受信息的人和信息内容
# user = sys.argv[1]
# content = sys.argv[2]

# print(user)
# print(content)

## 组装内容
## refer to: https://open-doc.dingtalk.com/docs/doc.htm?spm=a219a.7629140.0.0.karFPe&treeId=257&articleId=105735&docType=1
# data = {
#      "msgtype": "markdown",
#      "markdown": {
#          "title":"杭州天气",
#          "text": "#### 杭州天气 @156xxxx8827\n" +
#                  "> 9度，西北风1级，空气良89，相对温度73%\n\n" +
#                  "> ![screenshot](https://gw.alicdn.com/tfs/TB1ut3xxbsrBKNjSZFpXXcXhFXa-846-786.png)\n"  +
#                  "> ###### 10点20分发布 [天气](http://www.thinkpage.cn/) \n"
#      },
#     "at": {
#         "atMobiles": [
#             "15280013634"
#         ],
#         "isAtAll": False
#     }
#  }

## 调用request.post发送json格式的参数
# headers = {'Content-Type': 'application/json'}
# result = requests.post(url=webhook, data=json.dumps(data), headers=headers)
#
# print('--'*30)
# print(result)
# print(result.json())
# print('--'*30)
#
# if result.json()["errcode"] == 0:
#      print("send ok")
# else:
#     print("send failed!")


body={
         "msgtype": "markdown",
         "markdown": {
             "title":"项目单测情况",
             "text": "#### 杭州天气 @156xxxx8827\n" +
              "> 9度，西北风1级，空气良89，相对温度73%\n\n" +
              "> ![screenshot](https://gw.alicdn.com/tfs/TB1ut3xxbsrBKNjSZFpXXcXhFXa-846-786.png)\n"  +
              "> ###### 10点20分发布 [天气](http://www.thinkpage.cn/) \n"
         },
       "at": {
           "atMobiles": ["15280013634"]
       }
    }

name='xxx'
tests=['a','b','c']
covg={'lineCoverage':1,'classCoverage':2}

# 字符串连接 用  反斜杠。
sendInfo="### **项目警告** \n \n" \
        " **项目构建:**%s  \n \n" \
        " **单测成功率**：%s%%   ----->> %s \n \n" \
        " **行覆盖率:** %s%% \n \n"  \
        " **类覆盖率:** %s%% \n \n"  \
        " ###  [查看详情](http://host/job/%s/) \n" %(str(name),tests[0],tests[1]+"/"+tests[2],covg["lineCoverage"],covg["classCoverage"],str(name))
 # 把这个拼接的内容，添加到 markdown text 中。
body["markdown"]["text"]=sendInfo


# print(str(sendInfo))
header001={'Content-Type': "application/json;charset=utf-8"}
dingdingToken='https://oapi.dingtalk.com/robot/send?access_token=dcb6811794604c625e6d4d70ef752da55be4662e32f0333b1b5f2996b7991a06'
resp=requests.post(url=dingdingToken,data=json.dumps(body),headers=header001)

print('--'*30)
print(resp)
print(resp.json())
print('--'*30)


if resp.json()["errcode"] == 0:
     print("send ok")
else:
    print("send failed!")





