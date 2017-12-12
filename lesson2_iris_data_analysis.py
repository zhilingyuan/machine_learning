# iris 数据 花粉形状分类
#
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()

#load the iris data
#iris.target={0,1,2}对应花粉的三个类别
#iris.data1-[sepal_width ,sepad.length,pedal.width,pedal.length]

#get iris data
iris=datasets.load_iris()
binary_target=np.array([1.if x==0 else 0. for x in iris.target])#regular grammer
iris_2d=np.array([[x[2],x[3]]for x in iris.data])#data pedal

batch_size=20
sess=tf.Session()
x1_data=tf.placeholder(shape=[None,1],dtype=tf.float32)
x2_data=tf.placeholder(shape=[None,1],dtype=tf.float32)
y_target=tf.placeholder(shape=[None,1],dtype=tf.float32)

A=tf.Variable(tf.random_normal(shape=[1,1]))
b=tf.Variable(tf.random_normal(shape=[1,1]))#

my_mult=tf.matmul(x2_data,A)
my_add=tf.add(my_mult,b)
my_output=tf.subtract(x1_data,my_add)

xentropy=tf.nn.sigmoid_cross_entropy_with_logits(logits=my_output,
                                                 labels=y_target)

my_opt=tf.train.GradientDescentOptimizer(0.05)
train_step=my_opt.minimize(xentropy)

init=tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
    rand_index = np.random.choice(len(iris_2d), size=batch_size)
    
    rand_x = iris_2d[rand_index]

    rand_x1 = np.array([[x[0]] for x in rand_x])
    rand_x2 = np.array([[x[1]] for x in rand_x])
    
    #if [] missing or dislocation.
    #error: float() argument must be a string or a number, not 'generator'
    
    rand_y = np.array([[y] for y in binary_target[rand_index]])
    sess.run(train_step, feed_dict={x1_data: rand_x1,
                                    x2_data: rand_x2,
                                    y_target: rand_y})

    if(i+1)%200==0:
        print('step:'+str(i+1)+'A='+str(sess.run(A))+',b='+str(sess.run(b)))

[[slope]]=sess.run(A)
[[intercept]]=sess.run(b)

x=np.linspace(0,3,num=50)
ablineValues=[]
for i in x:
    ablineValues.append(slope*i+intercept)

setosa_x = [a[1] for i,a in enumerate(iris_2d) if binary_target[i]==1]#enumerate 索引遍历
setosa_y = [a[0] for i,a in enumerate(iris_2d) if binary_target[i]==1]
non_setosa_x = [a[1] for i,a in enumerate(iris_2d) if binary_target[i]==0]
non_setosa_y = [a[0] for i,a in enumerate(iris_2d) if binary_target[i]==0]
plt.plot(setosa_x, setosa_y, 'rx', ms=10, mew=2, label='setosa')
plt.plot(non_setosa_x, non_setosa_y, 'ro', label='Non-setosa')
plt.plot(x, ablineValues, 'b-')
plt.xlim([0.0, 2.7])
plt.ylim([0.0, 7.1])
plt.suptitle('Linear Separator For I.setosa', fontsize=20)
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.legend(loc='lower right')
plt.show()
        

        
              
