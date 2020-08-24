import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn2pmml import sklearn2pmml
from sklearn.datasets import load_iris
import os
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

iris=load_iris()
data=iris.data
target=iris.target

# os.environ["PATH"] += os.pathsep + 'C:/Program Files/Java/jdk1.8.0_171/bin'
# X=[[1,2,3,1],[2,4,1,5],[7,8,3,6],[4,8,4,7],[2,5,6,9]]
# y=[0,1,0,2,1]

# pipeline = PMMLPipeline([("classifier", tree.DecisionTreeClassifier(random_state=9))]);
# pipeline.fit(data,target)
# sklearn2pmml(pipeline, "tree_result.pmml")


# url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
# names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
# dataframe = pd.read_csv(url, names=names)
# print(dataframe)
# array = dataframe.values
# X = array[:,0:8]
# Y = array[:,8]
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(data, target, test_size=test_size, random_state=seed)
# Fit the model on training set
# model = LogisticRegression()
# model.fit(X_train, Y_train)

pipeline = PMMLPipeline([("classifier",  LogisticRegression())]);
pipeline.fit(X_train, Y_train)
sklearn2pmml(pipeline, "logit_result.pmml")







