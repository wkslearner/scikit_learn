
'''
flask服务模型部署
'''

from flask import Flask
from sklearn.externals import joblib   #生成或载入模型
import os
import json
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
import pickle
from flask import Flask, jsonify, request
from sklearn.datasets import load_iris

warnings.filterwarnings("ignore")

# df=pd.read_excel('/Users/admin/Documents/data_analysis/fraud_model/data_source/fraud_data.xlsx')
#
# #测试数据
# item=[1.2,5.5,0.8,0.8]
# print(np.array(item).reshape(1,4))

#载入已生成的pikle模型进行预测
# pkl_file=open('model_v1.0.pk','rb')
# load_pickle=pickle.load(pkl_file)
# print(load_pickle.predict(np.array(item).reshape(1,4)))
# pkl_file.close()


'''生成flask上的api接口'''
app = Flask(__name__)

@app.route('/web', methods=['POST'])
def apicall():
    """API Call
    Pandas dataframe (sent as a payload) from API Call
    """
    #接口传入的json数据

    try:
        print(request)
        print(request.headers)
        print(request.form)
        print(request.json)

        test_json = request.get_json(force=True)
        test = pd.read_json(test_json, orient='records')  #json转换成dataframe

        # To resolve the issue of TypeError: Cannot compare types 'ndarray(dtype=int64)' and 'str'
        test['Dependents'] = [str(x) for x in list(test['Dependents'])]  #把数据转换成文本格式

        # Getting the Loan_IDs separated out
        loan_ids = test['Loan_ID']

    except Exception as e:
        raise e

    #pickle模型文档所在路径
    clf = 'model_v1.pk'

    if test.empty:
        return (bad_request())   #错误信息返回
    else:
        # Load the saved model
        print("Loading the model...")
        loaded_model = None
        with open('./models/' + clf, 'rb') as f:
            loaded_model = pickle.load(f)

        print("The model has been loaded...doing predictions now...")
        predictions = loaded_model.predict(test)  #使用载入模型进行预测

        """Add the predictions as Series to a new pandas dataframe
                                OR
           Depending on the use-case, the entire test data appended with the new files
        """
        prediction_series = list(pd.Series(predictions))

        final_predictions = pd.DataFrame(list(zip(loan_ids, prediction_series)))

        """We can be as creative in sending the responses.
           But we need to send the response codes as well.
        """
        responses = jsonify(predictions=final_predictions.to_json(orient="records"))
        responses.status_code = 200

        return (responses)


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
    # 运行到指定端口
    app.run(host='127.0.0.1',port=3000,debug=True)












