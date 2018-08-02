

import pandas as pd
import statsmodels.api as sm
import copy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.metrics import classification_report
import time
from matplotlib import pylab
from matplotlib import pyplot

start_time = time.time()

iris = pd.read_csv('/Users/andpay/Downloads/iris.csv')
new_df=iris[(iris['Species']=='setosa') | (iris['Species']=='versicolor')]
new_df=new_df.replace(['setosa','versicolor'],[0,1])

col_name=new_df.columns[1:5]
locgit=LogisticRegression()
locgit.fit(new_df[col_name],new_df['Species'])


predict_data=copy.deepcopy(new_df)
predict_col=predict_data.columns[1:5]
y_test=predict_data['Species']
print(y_test)

predict_result=locgit.predict(predict_data[predict_col])

#predict_data['predict']=locgit.predict(predict_data[predict_col])
answer = locgit.predict_proba(predict_data[predict_col])[:,1]
print(answer)
precision, recall, thresholds = precision_recall_curve(y_test, answer)
report = answer > 0.5
print(classification_report(y_test, report, target_names = ['neg', 'pos']))
print("time spent:", time.time() - start_time)


