import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
#导入MNIST数据集

import tensorflow as tf
#导入tensorflow

x = tf.placeholder("float", [None, 784])
#x负责输入任意数量的MNIST图像

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
#模型参数初始化

y = tf.nn.softmax(tf.matmul(x,W) + b)
#实现模型

y_  = tf.placeholder("float", [None,10])
#用于输入正确值的占位符

cross_entropy = -tf.reduce_sum(y_*tf.log(y))
#计算交叉熵

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
#梯度下降算法以0.01的学习速率最小化交叉熵

init = tf.initialize_all_variables()
#初始化创建的变量

sess =tf.Session()
sess.run(init)
#启动模型并初始化变量

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x:batch_xs, y_:batch_ys})
#训练模型，让模型循环1000次，随机抓取训练数据中的100个批处理数据点

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#检测预测是否真实标签匹配

accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
#把判断的布尔值转为浮点数

print(sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels}))

