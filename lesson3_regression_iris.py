import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets
from tensorflow.python.framework import ops
ops.reset_default_graph()
sess=tf.Session()
iris=datasets.load_iris()

x_vals=np.array([x[3] for x in iris.data])
y_vals=np.array([y[0] for y in iris.data])#回忆iris data 的格式

learning_rate=0.05
batch_size=25
x_data=tf.placeholder(shape=[None,1],dtype=tf.float32)
y_target=tf.placeholder(shape=[None,1],dtype=tf.float32)
A=tf.Variable(tf.random_normal(shape=[1,1]))
b=tf.Variable(tf.random_normal(shape=[1,1]))

model_output=tf.add(tf.matmul(x_data,A),b)

loss=tf.reduce_mean(tf.square(y_target-model_output))
init=tf.global_variables_initializer()
sess.run(init)
my_opt=tf.train.GradientDescentOptimizer(learning_rate)
train_step=my_opt.minimize(loss)

loss_vec_l2=[]
for i in range(100):
    rand_index=np.random.choice(len(x_vals),size=batch_size)
    rand_x=np.transpose([x_vals[rand_index]])
    rand_y=np.transpose([y_vals[rand_index]])
    sess.run(train_step,feed_dict={x_data:rand_x,y_target:rand_y})
    temp_loss=sess.run(loss,feed_dict={x_data:rand_x,y_target:rand_y})
    loss_vec_l2.append(temp_loss)
    if(i+1)%25==0:
        print('step#'+str(i+1)+'A='+str(sess.run(A))+' A'+
              str(sess.run(b)))
        print('loss= '+str(temp_loss))

[slope]=sess.run(A)
[y_intercept]=sess.run(b)
best_fit=[]
for i in x_vals:
    best_fit.append(slope*i+y_intercept)

plt.plot(x_vals,y_vals,'o',label='Data Points')
plt.plot(x_vals,best_fit,'r-',label='best fit')
plt.legend(loc='upper left')
plt.title('sepal length vs pedal width')
plt.xlabel('pedal width')
plt.ylabel('sepal length')
plt.show()
plt.plot(loss_vec_l2,'k-')
plt.title('L2 loss per generation')
plt.ylabel('L2 loss')
plt.xlabel('generation')
plt.show()

              
