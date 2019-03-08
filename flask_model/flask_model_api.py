'''
使用flask生成api接口，用于模型预测
'''

import numpy as np
from flask import Flask
from flask import request
from flask import jsonify
from sklearn.externals import joblib
import pickle
import requests


# 导入模型
model = joblib.load('model_v1.0.pk')
# model=pickle.load('model_v1.0.pk')

# temp =  [5.1,3.5,1.4,0.2]
# temp = np.array(temp).reshape((1, -1))
# ouputdata = model.predict(temp)
##获取预测分类结果
# print('分类结果是：',ouputdata[0])


app = Flask(__name__)
print(app)

#route装饰器，实现在网页上输出结果的功能
#装饰器是一种接受函数,当你装饰一个函数，意味着你告诉Python调用的是那个由你的装饰器返回的新函数，而不仅仅是直接返回原函数体的执行结果。
@app.route('/', methods=['POST', 'GET'])
def output_data():
    text = request.args.get('inputdata')
    if text:
        temp = [float(x) for x in text.split(',')]
        temp = np.array(temp).reshape((1, -1))
        ouputdata = model.predict(temp)
        return jsonify(str(ouputdata[0]))



'''外部调用预测模型时，需要在服务器终端 运行命令行 python flask_model_api'''
if __name__ == '__main__':
    app.config['JSON_AS_ASCII'] = False
    app.run(host='127.0.0.1', port=5003)  # 127.0.0.1 #指的是本地ip,在文件内直接运行APP服务


    # from sklearn.datasets import load_iris
    # iris = load_iris()
    #
    # for row in iris.data:
    #     row_s = ''
    #     for i in range(len(row)):
    #         if i == 0 or i == len(row):
    #             row_s = row_s + str(row[i])
    #         else:
    #             row_s = row_s + ',' + str(row[i])
    #
    #     url = 'http://127.0.0.1:5003/?inputdata='
    #     base = url + row_s
    #
    #     response = requests.get(base)
    #     answer = response.json()
    #     print('预测结果', answer)

print('运行结束')
