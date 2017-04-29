from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/",one_hot=True,validation_size=5000)
print(mnist.train.images.shape,mnist.train.labels.shape)
print(mnist.test.images.shape,mnist.test.labels.shape)
print(mnist.validation.images.shape,mnist.validation.labels.shape)


sess = tf.InteractiveSession()
x = tf.placeholder(tf.float32,[None,784])
w = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([1,10]))

y_output = tf.nn.softmax(tf.matmul(x,w)+b)

y_label = tf.placeholder(tf.float32,[None,10])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_label*tf.log(y_output),reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(loss=cross_entropy)

tf.global_variables_initializer().run()

for i in range(1000):
    batch_x, batch_y = mnist.train.next_batch(100)
    train_step.run({x:batch_x,y_label:batch_y})

# estimate
correct_prediction = tf.equal(tf.argmax(y_output,dimension=1),tf.argmax(y_label,dimension=1))
accracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print(accracy.eval({x:mnist.test.images,y_label:mnist.test.labels}))




