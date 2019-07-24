
import  pandas  as pd
import  numpy as np
# from sklearn2pmml.pipeline import PMMLPipeline
# from sklearn2pmml import sklearn2pmml
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


class MockBinaryClassifier(BaseEstimator):

    def __init__(self):
        self.n_classes_ = 2
        self.classes_ = np.array([0, 1])

    def fit(self, features: np.ndarray, target: np.ndarray, sample_weight: np.ndarray = None):
        return self

    def predict(self, features: np.ndarray):
        return np.where(features[:, 0] > 0, 1, 0)


test_feature = np.array([[0], [0.5], [3], [-1]])

train_feature = np.zeros_like(test_feature)
train_target  = np.zeros_like(test_feature)


pipe = Pipeline([("scale",MinMaxScaler()),
                 ("mock",MockBinaryClassifier())])

pred = pipe.fit(train_feature,train_target).predict(test_feature)
print(pred)



# class PreProcessing(BaseEstimator, TransformerMixin):
#     """Custom Pre-Processing estimator for our use-case
#     """
#
#     def __init__(self):
#         pass





