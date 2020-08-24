

import requests
import pandas as pd
import numpy as np
from anti_fraud_order_sequence.order_sequence_kmeans import data_standar,order_sequence_predit,models
import time
import json
import time

dataset=pd.read_excel('/Users/admin/Documents/data_analysis/fraud_model/analysis_behave/user_visit_behave_transform_1903.xlsx')
dataset=dataset[['all_num','zero_num','one_num','diff_ratio','last_diff','mean_diff']]

start_time=time.time()

urls='http://127.0.0.1:8088/predict'
respone=requests.post(url=urls,data=json.dumps({'data':{'all_num':5,'zero_num':2,'one_num':3,
                                                        'diff_ratio':4,'last_diff':5,'mean_diff':6}}))
print(respone.json())
print(time.time()-start_time)

# for line in np.array(dataset):
#     print(line)
#     start_time=time.time()
#     print({'data':list(line)})
#
#     respone=requests.post(url=urls,data=json.dumps({'data':list(line)}))
#     # print(respone.status_code)
#     # print(respone.text)
#     print(respone.json())
#     print(time.time()-start_time)



# base = 'http://127.0.0.1:8088/tasks/1'
# response = requests.get(base)
# answer = response.json()
# print(answer)


