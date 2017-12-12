#sigmoid 分割不同的正太分布 
import tensorflow as tf
import numpy as np
sess=tf.Session()
x_vals=np.concatenate((np.random.normal(-1,1,50),
                       np.random.normal(3,1,50)))
y_vals=np.concatenate((np.repeat(0.,50),np.repeat(1.,50)))#数组相接类似append
x_data=tf.placeholder(shape=[1],dtype=tf.float32)
y_target=tf.placeholder(shape=[1],dtype=tf.float32)
A=tf.Variable(tf.random_normal(mean=10,shape=[1]))
my_output=tf.add(x_data,A)
my_output_expanded=tf.expand_dims(my_output,0)# shape=[2]
                                              # expand(0) 则 shape=[1,2]
                                              # 0 represents the index of shape
                                              # reverse fuction is squeeze()
y_target_expanded=tf.expand_dims(y_target,0)
init=tf.global_variables_initializer()
sess.run(init)
xentropy=tf.nn.sigmoid_cross_entropy_with_logits(logits=my_output_expanded,
                                                 labels=y_target_expanded)
my_opt=tf.train.GradientDescentOptimizer(0.05)
train_step=my_opt.minimize(xentropy)
for i in range(1400):
    rand_index=np.random.choice(100)
    rand_x=[x_vals[rand_index]]
    rand_y=[y_vals[rand_index]]
    sess.run(train_step,feed_dict={x_data:rand_x,
                                   y_target:rand_y})
    if (i+1)%200==0:
        print('step:'+str(i+1)+' A='+str(sess.run(A)))
        print('loss:'+str(sess.run(xentropy,feed_dict={
            x_data:rand_x,
            y_target:rand_y})))
