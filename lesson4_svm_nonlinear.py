import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn import datasets
from tensorflow.python.framework import ops
import pdb
ops.reset_default_graph()
sess=tf.Session()

iris=datasets.load_iris()
x_vals=np.array([[x[0],x[3]] for x in iris.data])
y_vals=np.array([1 if y==0 else -1 for y in iris.target])
#pdb.set_trace()
class1_x=[x[0] for i,x in enumerate(x_vals) if y_vals[i]==1]
class1_y=[x[1] for i,x in enumerate(x_vals) if y_vals[i]==1]
class2_x=[x[0] for i,x in enumerate(x_vals) if y_vals[i]==-1]
class2_y=[x[1] for i,x in enumerate(x_vals) if y_vals[i]==-1]

batch_size=100
x_data=tf.placeholder(shape=[None,2],dtype=tf.float32)
y_target=tf.placeholder(shape=[None,1],dtype=tf.float32)
prediction_grid=tf.placeholder(shape=[None,2],dtype=tf.float32)
b=tf.Variable(tf.random_normal(shape=[1,batch_size]))#分割矩阵

gamma=tf.constant(-10.0)
dist=tf.reduce_sum(tf.square(x_data),1)
dist=tf.reshape(dist,[-1,1])
sq_dists=tf.add(tf.subtract(dist,tf.multiply(2.,tf.matmul(x_data,
                tf.transpose(x_data)))),tf.transpose(dist))
my_kernel=tf.exp(tf.multiply(gamma,tf.abs(sq_dists)))

model_output=tf.matmul(b,my_kernel)
first_term=tf.reduce_sum(b)
b_vec_cross=tf.matmul(tf.transpose(b),b)
y_target_cross=tf.matmul(y_target,tf.transpose(y_target))
second_term=tf.reduce_sum(tf.multiply(my_kernel,
                                      tf.multiply(b_vec_cross,y_target_cross)))
loss=tf.negative(tf.subtract(first_term,second_term))
#事实上计算的loss这里就可以进行分割
#之后的是为prediction和画图决定的
#(g-x)^2
rA=tf.reshape(tf.reduce_sum(tf.square(x_data),1),[-1,1])
rB=tf.reshape(tf.reduce_sum(tf.square(prediction_grid),1),[-1,1])
pred_sq_dist=tf.add(tf.subtract(rA,tf.multiply(2.,tf.matmul(x_data,
                                        tf.transpose(prediction_grid)))),
                    tf.transpose(rB))
pred_kernel=tf.exp(tf.multiply(gamma,tf.abs(pred_sq_dist)))

prediction_output=tf.matmul(tf.multiply(tf.transpose(y_target),b),
                           pred_kernel)
prediction=tf.sign(prediction_output-tf.reduce_mean(prediction_output))
accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.squeeze(prediction),tf.squeeze(y_target)),
                                tf.float32))

my_opt=tf.train.GradientDescentOptimizer(0.01)
train_step=my_opt.minimize(loss)
init=tf.global_variables_initializer()
sess.run(init)

loss_vec=[]
batch_accuracy=[]

for i in range(300):
    rand_index=np.random.choice(len(x_vals),size=batch_size)
    rand_x=x_vals[rand_index]
    rand_y=np.transpose([y_vals[rand_index]])
    sess.run(train_step,feed_dict={x_data:rand_x,
                                   y_target:rand_y})
    temp_loss=sess.run(loss,feed_dict={x_data:rand_x,
                                       y_target:rand_y})
    loss_vec.append(temp_loss)
    acc_temp=sess.run(accuracy,feed_dict={x_data:rand_x,
                                          y_target:rand_y,
                                          prediction_grid:rand_x})
    batch_accuracy.append(acc_temp)

x_min,x_max=x_vals[:,0].min()-1,x_vals[:,0].max()+1
y_min,y_max=x_vals[:,1].min()-1,x_vals[:,1].max()+1
xx,yy=np.meshgrid(np.arange(x_min,x_max,0.02),
                  np.arange(y_min,y_max,0.02))
grid_points=np.c_[xx.ravel(),yy.ravel()]
[grid_predictions]=sess.run(prediction,feed_dict={x_data:rand_x,
                                                  y_target:rand_y,
                                                  prediction_grid:grid_points})
grid_predictions=grid_predictions.reshape(xx.shape)
#
#pdb.set_trace()

plt.contourf(xx,yy,grid_predictions,cmap=plt.cm.Paired,alpha=0.8)
plt.plot(class1_x,class1_y,'ro',label='I. setosa')
plt.plot(class2_x,class2_y,'kx',label='Non setosa')
plt.title('gaussian svm results on iris data')
plt.xlabel('pedal length')
plt.ylabel('sepal with')
plt.legend(loc='lower right')
plt.ylim([-0.5,3.0])
plt.xlim([3.5,8.5])
plt.show()
