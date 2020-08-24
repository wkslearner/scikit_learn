
'''
模型生成pkl文件
https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/
'''

import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle
import numpy as np

# url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
# names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
# dataframe = pandas.read_csv(url, names=names)
# print(dataframe)
# array = dataframe.values
# X = array[:,0:8]
# Y = array[:,8]
# test_size = 0.33
# seed = 7
# X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)
# # Fit the model on training set
# model = LogisticRegression()
# model.fit(X_train, Y_train)
# # save the model to disk
filename = 'finalized_model.pkl'
# pickle.dump(model, open(filename, 'wb'))

'''
data
preg  plas  pres  skin  test  mass   pedi  age  class
6   148    72    35     0  33.6  0.627   50      1
'''

test_data=pandas.DataFrame({
    'preg':[6,5,7,8,9],
    'plas':[145,137,124,165,178],
    'pres':[87,35,67,102,56],
    'skin':[23,14,45,33,51],
    'test':[0,1,1,1,0],
    'mass':[23.5,35.6,22.6,42.3,25.8],
    'pedi':[0.536,0.356,0.876,0.987,0.435],
    'age':[35,46,31,25,65]
})

# print(test_data)


# some time later...
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))
print(loaded_model)

for index,line in test_data.iterrows():
    # print(line)
    print(np.array(line).reshape(1,-1))
    res=loaded_model.predict(np.array(line).reshape(1,-1))
    print(res.__class__)

# result = loaded_model.score(X_test, Y_test)
# print(result)


'''
生成joblib文件'''
'''
# Save Model Using joblib
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import joblib

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:, 0:8]
Y = array[:, 8]
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)
# Fit the model on training set
model = LogisticRegression()
model.fit(X_train, Y_train)
# save the model to disk
filename = 'finalized_model.sav'
joblib.dump(model, filename)

# some time later...

# load the model from disk
loaded_model = joblib.load(filename)
result = loaded_model.score(X_test, Y_test)
print(result)
'''
