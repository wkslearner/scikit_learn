

'''
基于sklearn的受限玻尔兹曼机
这里RBM的作用是进行特征处理然后使用logit进行分类
'''

from __future__ import print_function

print(__doc__)

# Authors: Yann N. Dauphin, Vlad Niculae, Gabriel Synnaeve
# License: BSD

import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import convolve
from sklearn import linear_model, datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from plot_function.cluster_plot import plot_cluster

# #############################################################################
# Setting up

'''对数据进行膨胀操作'''
def nudge_dataset(X, Y):

    """
    这会产生比原始数据集大5倍的数据集，
    将Xx中的8x8图像向左，向右，向下移动1px
    """
    direction_vectors = [
        [[0, 1, 0],
         [0, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [1, 0, 0],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 1],
         [0, 0, 0]],

        [[0, 0, 0],
         [0, 0, 0],
         [0, 1, 0]]]

    #对数据进行移动操作
    def shift(x, w):
        return convolve(x.reshape((8, 8)), mode='constant', weights=w).ravel()

    print(X)
    print([np.apply_along_axis(shift, 1, X, vector)
                        for vector in direction_vectors])
    X = np.concatenate([X] +
                       [np.apply_along_axis(shift, 1, X, vector)
                        for vector in direction_vectors])

    #对Y变量进行五次重复，然后进行合并操作
    Y = np.concatenate([Y for _ in range(5)], axis=0)

    return X, Y


# Load Data
digits = datasets.load_digits()
iris=datasets.load_iris()

X=iris.data
Y=iris.target

# X = np.asarray(digits.data, 'float32')
# X, Y = nudge_dataset(X, digits.target)

#数据标准化
X = (X - np.min(X, 0)) / (np.max(X, 0) + 0.0001)  # 0-1 scaling

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=0)

# Models we will use
logistic = linear_model.LogisticRegression(solver='lbfgs', max_iter=10000,
                                           multi_class='multinomial')
rbm = BernoulliRBM(random_state=0, verbose=True)

rbm_features_classifier = Pipeline(
    steps=[('rbm', rbm), ('logistic', logistic)])


# Training

# Hyper-parameters. These were set by cross-validation,
# using a GridSearchCV. Here we are not performing cross-validation to
# save time.
rbm.learning_rate = 0.06
rbm.n_iter = 20
# More components tend to give better prediction performance, but larger
# fitting time
rbm.n_components = 100
logistic.C = 6000

# Training RBM-Logistic Pipeline
rbm_features_classifier.fit(X_train, Y_train)

# Training the Logistic regression classifier directly on the pixel
raw_pixel_classifier = clone(logistic)
raw_pixel_classifier.C = 100.
raw_pixel_classifier.fit(X_train, Y_train)



# Evaluation
Y_pred = rbm_features_classifier.predict(X_test)
print("Logistic regression using RBM features:\n%s\n" % (
    metrics.classification_report(Y_test, Y_pred)))

Y_pred = raw_pixel_classifier.predict(X_test)
print("Logistic regression using raw pixel features:\n%s\n" % (
    metrics.classification_report(Y_test, Y_pred)))


#label=rbm_features_classifier.predict(X)
label=raw_pixel_classifier.predict(X)
plot_cluster(iris.data,iris.target)


# # Plotting
#
# plt.figure(figsize=(4.2, 4))
# for i, comp in enumerate(rbm.components_):
#     plt.subplot(10, 10, i + 1)
#     plt.imshow(comp.reshape((8, 8)), cmap=plt.cm.gray_r,
#                interpolation='nearest')
#     plt.xticks(())
#     plt.yticks(())
# plt.suptitle('100 components extracted by RBM', fontsize=16)
# plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)
# plt.show()