

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import  load_iris
from sklearn.model_selection import train_test_split
from tensorflow.contrib.learn.python.learn.datasets import base
import xgboost
import shap
import os
import pandas as pd
import datetime


#解决xgboost训练报错问题
os.environ['KMP_DUPLICATE_LIB_OK'] ='True'

# a=pd.DataFrame({'a':[1,2],'b':[3,4]})
# b=pd.DataFrame({'a':[5,6],'b':[7,8]})
# print(a.append(b).sample(frac=1).reset_index(drop=True))

print(datetime.datetime(2019,6,1).isocalendar())


# dataset=pd.read_excel('/Users/admin/Documents/data_analysis/model/model_verification/verification_passrate.xlsx')
# max_id=dataset['report_id'].groupby(dataset['user_id']).max().reset_index()
# dataset=dataset[dataset['report_id'].isin(max_id['report_id'])]
# print(max_id)
# dataset.to_excel('/Users/admin/Documents/data_analysis/model/model_verification/verification_passrate.xlsx',index=False)



# # load JS visualization code to notebook
# shap.initjs()
# # train XGBoost model
# X,y = shap.datasets.boston()
# print(X)
# model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y), 100)
# # explain the model's predictions using SHAP values
# # (same syntax works for LightGBM, CatBoost, and scikit-learn models)
# explainer = shap.TreeExplainer(model)
# shap_values = explainer.shap_values(X)
#
# # # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
# # shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:],matplotlib=True)
#
# # # visualize the training set predictions
# # shap.force_plot(explainer.expected_value, shap_values,X)
#
# # create a SHAP dependence plot to show the effect of a single feature across the whole dataset
# shap.dependence_plot("LSTAT", shap_values, X)
# # summarize the effects of all the features
# shap.summary_plot(shap_values, X)
# shap.summary_plot(shap_values, X, plot_type="bar")







