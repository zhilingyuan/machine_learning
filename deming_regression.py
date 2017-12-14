#当初始的A值为负数的时候，因为垂直距离的问题？，导致失衡无法收敛正确
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
from tensorflow.python.framework import ops
ops.reset_default_graph()
sess=tf.Session()
iris=datasets.load_iris()
x_vals=np.array([x[3] for x in iris.data])
y_vals=np.array([y[0] for y in iris.data])
batch_size=50
x_data=tf.placeholder(shape=[None,1],dtype=tf.float32)
y_target=tf.placeholder(shape=[None,1],dtype=tf.float32)
A=tf.Variable(tf.random_normal(shape=[1,1]))
b=tf.Variable(tf.random_normal(shape=[1,1]))
model_output=tf.add(tf.matmul(x_data,A),b)

deming_numerator=tf.abs(tf.subtract(y_target,tf.add(tf.matmul(x_data,A),b)))#subtract
deming_denominator=tf.sqrt(tf.add(tf.square(A),1))
loss=tf.reduce_mean(tf.truediv(deming_numerator,deming_denominator))

init=init=tf.global_variables_initializer()
sess.run(init)
my_opt=tf.train.GradientDescentOptimizer(0.1)
step_train=my_opt.minimize(loss)
loss_vec=[]
for i in range(200000000):
    rand_index=np.random.choice(len(x_vals),size=batch_size)
    rand_x=np.transpose([x_vals[rand_index]])
    rand_y=np.transpose([y_vals[rand_index]])
    sess.run(step_train,feed_dict={x_data:rand_x,y_target:rand_y})
    temp_loss=sess.run(loss,feed_dict={x_data:rand_x,y_target:rand_y})
    loss_vec.append(temp_loss)
    if (i+1)%500==0:
        print('step#'+str(i+1)+'  A:'+str(sess.run(A))+'  b:'+str(sess.run(b)))
        print('loss='+str(temp_loss))


[slope]=sess.run(A)
[y_intercept]=sess.run(b)
best_fit=[]
for i in x_vals:
    best_fit.append(slope*i+y_intercept)
plt.plot(x_vals,y_vals,'o',label='data point')
plt.plot(x_vals,best_fit,'r-',label='best fit line')
plt.legend(loc='upper left')
plt.xlabel('pedal width')
plt.ylabel('sepal length')
plt.show()
