# -*- coding:utf-8 -*-
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
sess = tf.InteractiveSession()

import matplotlib.pyplot as plt
import matplotlib as mpl

def weight_variable(shape):
    initial = tf.truncated_normal(shape=shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(value=0.1, shape=shape)
    return tf.Variable(initial)

x = tf.placeholder(tf.float32,[None,784])
y_target = tf.placeholder(tf.float32,[None,10])
x_image = tf.reshape(x,[-1,28,28,1])    #由末到前的reshape


w_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])#和[1,32]应该是一致的？
h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image,w_conv1,strides=[1,1,1,1],padding='SAME') + b_conv1)    #The stride of the sliding window for each dimension of input
h_pool1 = tf.nn.max_pool(h_conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')      #也都是表示滑动窗口的各维度步长，以及对应的窗口大小

w_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1,w_conv2,strides=[1,1,1,1],padding='SAME') + b_conv2)
h_pool2 = tf.nn.max_pool(h_conv2,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

w_fc1 = weight_variable([7*7*64,1024])      #28*28*1,14*14*32，7*7*64
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob=keep_prob)

w_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,w_fc2) + b_fc2)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_target * tf.log(y_conv),reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss=cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv,axis=1),tf.argmax(y_target,axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))      #The dimensions to reduce. If `None` (the default), reduces all dimensions.

tf.global_variables_initializer().run()
cross_entropy_plot = []

for i in range(2000):
    batch = mnist.train.next_batch(50)
    cross_entropy_t = cross_entropy.eval(feed_dict={x:batch[0],y_target:batch[1],keep_prob:1.0})
    cross_entropy_plot.append(cross_entropy_t)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_target: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i,train_accuracy))

    train_step.run(feed_dict={x:batch[0],y_target:batch[1],keep_prob:0.5})

print("test accuracy %g"%accuracy.eval({x:mnist.test.images,y_target:mnist.test.labels,keep_prob:1.0}))
#
# 绘制cost曲线
mpl.rcParams['font.sans-serif'] = [u'SimHei']
mpl.rcParams['axes.unicode_minus'] = False
plt.figure(1)
plt.plot(cross_entropy_plot,color='r')
plt.xlabel(u'迭代次数', fontsize=12)
plt.ylabel(u'训练精度', fontsize=12)
plt.title('training accuracy', fontsize=14)
plt.show()






















