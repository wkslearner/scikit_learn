# Make sure that you have all these libaries available to run the code successfully

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from vecstack import stacking

link = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'

names = ['Class', 'Alcohol', 'Malic acid', 'Ash',
         'Alcalinity of ash', 'Magnesium', 'Total phenols',
         'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
         'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
         'Proline']

df = pd.read_csv(link, header=None, names=names)
print(df)





