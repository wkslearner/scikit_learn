
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn2pmml import sklearn2pmml
from sklearn.datasets import load_iris
import os

iris=load_iris()
data=iris.data
target=iris.target


# os.environ["PATH"] += os.pathsep + 'C:/Program Files/Java/jdk1.8.0_171/bin'
# X=[[1,2,3,1],[2,4,1,5],[7,8,3,6],[4,8,4,7],[2,5,6,9]]
# y=[0,1,0,2,1]


pipeline = PMMLPipeline([("classifier", tree.DecisionTreeClassifier(random_state=9))]);
pipeline.fit(data,target)
sklearn2pmml(pipeline, "tree_result.pmml")












