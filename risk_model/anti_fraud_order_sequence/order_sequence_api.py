

from flask import Flask, jsonify,request
from flask import make_response
from flask import abort
import pandas as pd
from anti_fraud_order_sequence.order_sequence_kmeans import data_standar,models
import json
import numpy as np


app = Flask(__name__)


# 数据标准化格式
standar = data_standar(status='standar')
# 获取训练数据集
train_data = data_standar(status='data')
# 模型训练
clf = models(train_data)


@app.route('/predict', methods=['POST'])
def apicall():


    # 接口传入的json数据
    try:
        test = json.loads(request.data)
    except Exception as e:   #捕获错误并挂起
        raise e


    if len(test)==0:
        return (bad_request())  # 错误信息返回
    else:
        #传入数据标准化
        test=standar.transform(np.array(test['data']).reshape(1,-1))
        #输出预测结果
        result=clf.predict(test)

        responses = jsonify({'result':result.tolist()})
        responses.status_code = 200

        return responses


'''返回错误消息'''
@app.errorhandler(400)
def bad_request(error=None):
    message = {
        'status': 400,
        'message': 'Bad Request: ' + request.url + '--> Please check your data payload...',
    }
    resp = jsonify(message)
    resp.status_code = 400

    return resp



if __name__ == '__main__':
    app.run(host='127.0.0.1',port=8088,debug=True)


















