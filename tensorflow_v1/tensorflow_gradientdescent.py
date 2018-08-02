import tensorflow as tf
import matplotlib.pyplot as plt

X = [1., 2., 3.]
Y = [1., 2., 3.]
m = n_samples = len(X)
W = tf.placeholder(tf.float32)

hypothesis = tf.multiply(X, W)
print(hypothesis)

#累计均方误差计算（损失函数）
cost = tf.reduce_mean(tf.pow(hypothesis - Y, 2)) / m

W_val = []
cost_val = []

#变量初始化
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)


for i in range(-10, 50):
    print (i * 0.1, sess.run(cost, feed_dict={W: i * 0.1}))
    W_val.append(i * 0.1)
    cost_val.append(sess.run(cost, feed_dict={W: i * 0.1}))


plt.plot(W_val, cost_val, 'ro')
plt.ylabel('cost')
plt.xlabel('W')
plt.show()
