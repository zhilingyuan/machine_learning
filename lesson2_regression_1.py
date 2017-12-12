#反向传播
#first
#对均值1 方差0.1的100个数据x y_=Ax拟合y=10 梯度法 区别最小二乘法（LMS）
import numpy as np
import tensorflow as tf
sess=tf.Session()
x_vals=np.random.normal(1,0.1,100)#uniform and normal
y_vals=np.repeat(10.,100)
x_data=tf.placeholder(shape=[1],dtype=tf.float32)
y_target=tf.placeholder(shape=[1],dtype=tf.float32)
A=tf.Variable(tf.random_normal(shape=[1]))
my_output=tf.multiply(x_data,A)
loss=tf.square(my_output-y_target)
#init=tf.initialize_all_variables()
init=tf.global_variables_initializer()
sess.run(init)
my_opt=tf.train.GradientDescentOptimizer(learning_rate=0.32)
train_step=my_opt.minimize(loss)
for i in range(100):
    rand_index=np.random.choice(100)
   # print('index:'+str(rand_index))
    rand_x=[x_vals[rand_index]]
    print(rand_x)
    rand_y=[y_vals[rand_index]]
    sess.run(train_step,feed_dict={x_data:rand_x,y_target:rand_y})
    if (i+1)%1==0:
        print('step #'+str(i+1)+' A='+str(sess.run(A)))
        print('loss ='+str(sess.run(loss,feed_dict={x_data:rand_x,
                                                    y_target:rand_y})))
