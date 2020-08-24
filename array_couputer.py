#!/usr/bin/python
# encoding=utf-8

import pandas as pd
import numpy as np
from data_process.list_process import remove_list
from data_process.feature_handle import disper_split
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV, KFold ,RandomizedSearchCV
from xgboost.sklearn import XGBClassifier
from sklearn.metrics import make_scorer, fbeta_score, accuracy_score,recall_score
from sklearn import metrics
import time
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from scipy import sparse
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_array, check_consistent_length
from sklearn.utils.testing import ignore_warnings
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster._dbscan_inner import dbscan_inner
import random
from sklearn.datasets.samples_generator import make_blobs
import Levenshtein
import networkx as nx
from sklearn.model_selection import GridSearchCV
from xgboost.sklearn import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

x=[(i+1)*0.05 for i in range(5)]
print(x)

xgb=XGBClassifier()
# xgb.set_params(**{'n_estimators':50})

# print(xgb.get_params())
# print('')
# print(xgb.get_xgb_params())










