from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets('MNIST_data',one_hot=True)
#mnist 轻量级 训练 验证 测试储存为Numpy数组
#tensorflow interactiveSession 与c++后端相连接为会话
import tensorflow as tf
sess=tf.InteractiveSession()
x=tf.placeholder(tf.float32,shape=[None,784])#长不确定n,宽784
y_=tf.placeholder(tf.float32,shape=[None,10])
w = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
sess.run(tf.global_variables_initializer())
y=tf.matmul(x,w)+b
cross_entropy=tf.reduce_mean(
    tf.nn.softmax_cross_entroy_with_logits(labels=y_,logits=y))
train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
for _ in range(1000):
    batch=mnist.train.next_batch(100)
    train_step.run(feed_dict={x:batch[0],y_:batch[1]})
