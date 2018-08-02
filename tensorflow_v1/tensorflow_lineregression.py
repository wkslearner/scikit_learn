import tensorflow as tf

x_data = [1, 2, 3]
y_data = [1, 2, 3]

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

#线性函数
hypothesis = W * X + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a) #采用梯度下降法训练函数，变量a为学习步长
train = optimizer.minimize(cost)

#变量的初始化
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

for step in range(100):
    sess.run(train, feed_dict={X: x_data, Y: y_data})  #往模型传入参数
    if step % 20 == 0:
        print (step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W), sess.run(b))  #运行张量或变量

print (sess.run(hypothesis, feed_dict={X: 5}))  #预测结果数据
print (sess.run(hypothesis, feed_dict={X: 2.5}))