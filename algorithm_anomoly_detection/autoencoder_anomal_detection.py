
'''
异常检测
https://towardsdatascience.com/anomaly-detection-with-autoencoder-b4cdce4866a6
'''

import pandas as pd
import numpy as np
import pandas as pdcode_transfer
# from pyod.models.knn import KNN
from pyod.models.auto_encoder import AutoEncoder
from pyod.utils.data import generate_data

contamination = 0.1  # percentage of outliers
n_train = 500  # number of training points
n_test = 500  # number of testing points
n_features = 25 # Number of features

X_train, y_train, X_test, y_test = generate_data(
    n_train=n_train, n_test=n_test,
    n_features= n_features,
    contamination=contamination,random_state=1234)

X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

# 数据标准化
from sklearn.preprocessing import StandardScaler
X_train = StandardScaler().fit_transform(X_train)
X_train = pd.DataFrame(X_train)
X_test = StandardScaler().fit_transform(X_test)
X_test = pd.DataFrame(X_test)

# 数据降维
from sklearn.decomposition import PCA
pca = PCA(2)
x_pca = pca.fit_transform(X_train)
x_pca = pd.DataFrame(x_pca)
x_pca.columns=['PC1','PC2']

# Plot
import matplotlib.pyplot as plt
# plt.scatter(X_train[0], X_train[1], c=y_train, alpha=0.8)
# plt.title('Scatter plot')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.show()

# 自编码函数
clf1 = AutoEncoder(hidden_neurons =[25, 2, 2, 25])
clf1.fit(X_train)

# Get the outlier scores for the train data
y_train_scores = clf1.decision_scores_

# Predict the anomaly scores  预测函数
y_test_scores = clf1.decision_function(X_test)  # outlier scores
y_test_scores = pd.Series(y_test_scores)

# Plot it!
import matplotlib.pyplot as plt
# plt.hist(y_test_scores, bins='auto')
# plt.title("Histogram for Model Clf1 Anomaly Scores")
# plt.show()

#
df_test = X_test.copy()
df_test['score'] = y_test_scores
df_test['cluster'] = np.where(df_test['score']<4, 0, 1)
df_test['cluster'].value_counts()
df_test.groupby('cluster').mean()

clf2 = AutoEncoder(hidden_neurons =[25, 10,2, 10, 25])
clf2.fit(X_train)

# Predict the anomaly scores
y_test_scores = clf2.decision_function(X_test)
y_test_scores = pd.Series(y_test_scores)

# Plot the histogram
import matplotlib.pyplot as plt
# plt.hist(y_test_scores, bins='auto')
# plt.title("Histogram for Model Clf2 Anomaly Scores")
# plt.show()

df_test = X_test.copy()
df_test['score'] = y_test_scores
df_test['cluster'] = np.where(df_test['score']<4, 0, 1)
df_test['cluster'].value_counts()
df_test.groupby('cluster').mean()


# Step 1: Build the model
clf3 = AutoEncoder(hidden_neurons =[25, 15, 10, 2, 10,15, 25])
clf3.fit(X_train)

# Predict the anomaly scores
y_test_scores = clf3.decision_function(X_test)
y_test_scores = pd.Series(y_test_scores)

# Step 2: Determine the cut point
import matplotlib.pyplot as plt
# plt.hist(y_test_scores, bins='auto')
# plt.title("Histogram with Model Clf3 Anomaly Scores")
# plt.show()

df_test = X_test.copy()
df_test['score'] = y_test_scores
df_test['cluster'] = np.where(df_test['score']<4, 0, 1)
df_test['cluster'].value_counts()

# Step 3: Get the summary statistics by cluster
df_test.groupby('cluster').mean()


# Put all the predictions in a data frame
from pyod.models.combination import aom, moa, average, maximization

# Put all the predictions in a data frame
train_scores = pd.DataFrame({'clf1': clf1.decision_scores_,
                             'clf2': clf2.decision_scores_,
                             'clf3': clf3.decision_scores_
                            })

test_scores  = pd.DataFrame({'clf1': clf1.decision_function(X_test),
                             'clf2': clf2.decision_function(X_test),
                             'clf3': clf3.decision_function(X_test)
                            })

# Although we did standardization before, it was for the variables.
# Now we do the standardization for the decision scores

train_scores_norm, test_scores_norm = StandardScaler.standardizer(train_scores,test_scores)


# Combination by average
y_by_average = average(test_scores_norm)

import matplotlib.pyplot as plt

plt.hist(y_by_average, bins='auto')  # arguments are passed to np.histogram
plt.title("Combination by average")
plt.show()


df_test = pd.DataFrame(X_test)
df_test['y_by_average_score'] = y_by_average
df_test['y_by_average_cluster'] = np.where(df_test['y_by_average_score']<0, 0, 1)
df_test['y_by_average_cluster'].value_counts()

# Combination by max
y_by_maximization = maximization(test_scores_norm)
print(y_by_maximization.shape)

