
from flask import Flask, jsonify ,request
from anti_fraud_order_sequence.order_sequence_kmeans import data_standar ,models,dict_analysis
import json
import numpy as np


app = Flask(__name__)


class  model_route():

    def __init__(self):
        pass


    def order_sequence_model(self):

        # 获取训练数据集
        self.train_data = data_standar(status='data')

        # 数据标准化格式
        self.standar=data_standar(status='standar')

        # 模型训练
        self.clf = models(self.train_data)

        return self



@app.route('/predict', methods=['POST'])
def apicall():

    # 接口传入的json数据
    try:
        test = json.loads(request.data)
    except Exception as e:  # 捕获错误并挂起（同时传给调用方，exception为Python框架报错机制）
        raise e


    if len(test)==0:
        return (bad_request())  # 错误信息返回
    else:

        # 传入数据标准化
        test=dict_analysis(test['data'])

        test =route.standar.transform(np.array(test).reshape(1 ,-1))
        # 输出预测结果
        result =route.clf.predict(test)

        responses = jsonify({'result' :result.tolist()[0]})
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
    #模型运行
    route=model_route()
    route.order_sequence_model()
    app.run(host='127.0.0.1',port=8088 ,debug=True)


















