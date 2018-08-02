""" Random Forest.
Implement Random Forest algorithm with TensorFlow, and apply it to classify 
handwritten digit images. This example is using the MNIST database of 
handwritten digits as training samples (http://yann.lecun.com/exdb/mnist/).
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.tensor_forest.python import tensor_forest
from tensorflow.python.ops import resources

# Ignore all GPUs, tf random forest does not benefit from it.
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=False)


# Parameters
num_steps = 500    # Total steps to train 训练次数
batch_size = 1024  # The number of samples per batch
num_classes = 10   # The 10 digits
num_features = 784 # Each image is 28x28 pixels
num_trees = 10    #树的数量
max_nodes = 1000  #最大节点数


# Input and Target data  输入数据和目标数据
X = tf.placeholder(tf.float32, shape=[None, num_features])
# For random forest, labels must be integers (the class id)  建立分类标签
Y = tf.placeholder(tf.int32, shape=[None])


# Random Forest Parameters  模型参数
hparams = tensor_forest.ForestHParams(num_classes=num_classes,
                                      num_features=num_features,
                                      num_trees=num_trees,
                                      max_nodes=max_nodes).fill()


# Build the Random Forest  建立随机森林计算图
forest_graph = tensor_forest.RandomForestGraphs(hparams)
# Get training graph and loss 添加训练和损失函数计算图
train_op = forest_graph.training_graph(X, Y)
loss_op = forest_graph.training_loss(X, Y)


# Measure the accuracy 测量精度
infer_op, _, _ = forest_graph.inference_graph(X)
correct_prediction = tf.equal(tf.argmax(infer_op, 1), tf.cast(Y, tf.int64))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# Initialize the variables (i.e. assign their default value) and forest resources 初始化参数
init_vars = tf.group(tf.global_variables_initializer(),
    resources.initialize_resources(resources.shared_resources()))


# Start TensorFlow session
sess = tf.Session()


# Run the initializer  参数初始化
sess.run(init_vars)


# Training
for i in range(1, num_steps + 1):
    # Prepare Data
    # Get the next batch of MNIST data (only images are needed, not labels) 导入训练数据
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    _, l = sess.run([train_op, loss_op], feed_dict={X: batch_x, Y: batch_y})
    if i % 50 == 0 or i == 1:
        acc = sess.run(accuracy_op, feed_dict={X: batch_x, Y: batch_y})
        print('Step %i, Loss: %f, Acc: %f' % (i, l, acc))


# Test Model 模型测试
test_x, test_y = mnist.test.images, mnist.test.labels
print("Test Accuracy:", sess.run(accuracy_op, feed_dict={X: test_x, Y: test_y}))