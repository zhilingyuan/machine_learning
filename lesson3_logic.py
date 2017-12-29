import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
import requests
import os.path
from sklearn import datasets
from sklearn.preprocessing import normalize
print(os.path.abspath('.'))
ops.reset_default_graph()
sess=tf.Session()
birthdata_url='https://www.umass.edu/statdata/statdata/data/lowbwt.dat'
birth_file=requests.get(birthdata_url)
birth_data=birth_file.text.split('\r\n')[5:]#error birth file 为<response 403>
                                            #因此error
# import pdb;
# pdb.set_trace();
birth_header=[x for x in birth_data[0].split(' ') if len(x)>=1]
birth_data=[[float(x) for x in y.split(' ') if len(x)>=1] for y in
            birth_data[1:] if len(y)>=1]
y_vals=np.array([x[1] for x in birth_data])
x_vals=np.array([x[2:9] for x in birth_data])

train_indices=np.random.choice(len(x_vals),round(len(x_vals)*0.8),
                               repalce=False)
test_indices=np.array(list(set(range(len(x_vals)))-set(train_indices)))
x_vals_train=x_vals(train_indices)
x_vals_test=x_vals(test_indices)
y_vals_train=y_vals(train_indices)
y_vals_test=y_vals(test_indices)

def normalize_cols(m):
    col_max=m.max(axis=0)
    col_min=m.min(axis=0)
    return(m-col_min)/(col_max-col_min)#归一化，使得logic 回归的效果更好

x_vals_train=np.nan_to_num(normalize_cols(x_vals_train))
x_vals_test=np.nan_to_num(normalize_cols(x_vals_test))#nan=0 并且 inf=无穷大
                        #而inf-inf是nan */0 0*inf inf/inf都是nan
batch_size=25
x_data=tf.placeholder(shape=[None,7],dtype=tf.float32)
y_target=tf.placeholder(shape=[None,1],dtype=tf.float32)
A=tf.Variable(tf.random_normal(shape=[7,1]))
b=tf.Variable(tf.random_normal(shape=[1,1]))
model_output=tf.add(tf.matmul(x_data,A),b)

loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    model_output,y_target))#熵 逻辑交叉熵损失函数
init=tf.gloabal_variables_initializer()
sess.run(init)
my_opt=tf.train.GradientDescentOpetimizer(0.01)
train_step=my_opt.minimize(loss)

prediction=tf.round(tf.sigmoid(model_output))
predictions_correct=tf.cast(tf.equal(prediction,y_target),tf.float32)#类似c++ 转换cast
accuracy=tf.reduce_mean(predictions_correct)

loss_vec=[]
train_acc=[]
for i in range(1500):
    rand_index=np.random.choice(len(x_vals_train),
                                size=batch_size)
    rand_x=x_vals_train[rand_index]
    rand_y=np.transpose(y_vals_train[rand_index])
    sess.run(train_step,feed_dict={x_data:rand_x,y_target:rand_y})
    loss_vec.append(temp_loss)
    temp_acc_train=sess.run(accuracy,feed_dict={x_data:x_vals_train,
                                               y_target:np.transpose([y_vals_train])}
                           )
    train_acc.append(temp_acc_train)
    temp_acc_test=sess.run(accuracy,feed_dict={x_data:x_vals_test,
                                               y_target:np.transpose([y_vals_test])})#transpose 注意np.transpose([数组])
    test_acc.append(temp_acc_test)

plt.plot(loss_vec,'k-')
plt.title('cross entropy loss per generation')
plt.xlabel('generation')
plt.ylabel('cross entropy loss')
plt.show()
plt.plot(train_acc,'k-',label='train_set_accuracy')
plt.plot(test_acc,'r--',label='test_set_accuracy')
plt.title('train and test set accuracy')
plt.xlabel('generation')
plt.ylabel('accuracy')
plt.legent(loc='lower right')
plt.show()
