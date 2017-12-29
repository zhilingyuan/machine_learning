#maximum margin  used in linear poly
#not svm
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets

from tensorflow.python.framework import ops

sess=tf.Session()
ops.reset_default_graph()

iris=datasets.load_iris()

x_vals=np.array([x[3] for x in iris.data])
y_vals=np.array([y[0] for y in iris.data])
train_indices=np.random.choice(len(x_vals),
                            round(len(x_vals)*0.8),
                            replace=False)
test_indices=np.array(list(set(range(len(x_vals)))-set(train_indices)))

x_vals_train=x_vals[train_indices]
x_vals_test=x_vals[test_indices]
y_vals_train=y_vals[train_indices]
y_vals_test=y_vals[test_indices]

batch_size=50
x_data=tf.placeholder(shape=[None,1],dtype=tf.float32)
y_data=tf.placeholder(shape=[None,1],dtype=tf.float32)

A=tf.Variable(tf.random_normal(shape=[1,1]))
b=tf.Variable(tf.random_normal(shape=[1,1]))

model_output=tf.add(tf.matmul(x_data,A),b)
epsilon=tf.constant([0.5])
loss=tf.reduce_mean(tf.maximum(0.,
            tf.subtract(tf.abs(tf.subtract(model_output,y_vals)),epsilon)))
my_opt=tf.train.GradientDescentOptimizer(0.075)
train_step=my_opt.minimize(loss)
init=tf.global_variables_initializer()
sess.run(init)

train_loss=[]
test_loss=[]
for i in range(200):
    rand_index=np.random.choice(len(x_vals_train),size=batch_size)
    rand_x=np.transpose([x_vals_train[rand_index]])
    rand_y=np.transpose([x_vals_train[rand_index]])
    sess.run(train_step,feed_dict={x_data:rand_x,y_target:rand_y})
    temp_train_loss=sess.run(loss,feed_dict={x_data:np.transpose([x_vals_train]),
                                             y_data:np.transpose([y_vals_train])})
    temp_test_loss=sess.run(loss,feed_dict={x_data:np.transpose([x_vals_test]),
                                            y_data:np.transpose([y_vals_test])})
    #一维的transpose 有意义吗？？？
    test_loss.append(temp_test_loss)
    train_loss.append(temp_train_loss)
    if(i+1)%50==0:
        print('---------')
        print('generation'+str(i+1))
        print('A='+str(sess.run(A))+'b='+str(sess.run(b)))
        print('train_loss'+str(temp_train_loss))
        print('test_loss'+str(temp_test_loss))
        
