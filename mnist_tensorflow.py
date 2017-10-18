import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# import mnist data
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


# set placeholder, set values after TensofFlow runs
# It is flattened to 784 dimensional vector for optimization
# None means it can be any length
x = tf.placeholder(tf.float32, [None, 784])


# variable is modifiable tensor that lives in TensorFlow's graph
# initial W and b to full of zeros

# weight
W = tf.Variable(tf.zeros([784, 10]))

# bias
b = tf.Variable(tf.zeros([10]))


# implement model, vector multiplication
# after this, model can be run in multiple platforms
# softmax is the activation function
y = tf.nn.softmax(tf.matmul(x, W) + b)


# determin the loss of the model
# input correct answers
y_ = tf.placeholder(tf.float32, [None, 10])
# implment cross entropy
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))


# train, gradient descent, backpropagation 
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


# launch the model in an InteractiveSession
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()


# train 1000 times
# stochastic gradient descent, use subsets
for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


# evaluating the model
# list of booleans [True, False, True, True] [1,0,1,1] = 0.75
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))


# accuracy on test data
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# This should be around 92% accuracy.  We want around 97% accuracy
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
