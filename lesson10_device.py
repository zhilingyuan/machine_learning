#using multiple device
import tensorflow as tf
#sess=tf.Session(config=tf.ConfigProto(log_device_placement=True))
a=tf.constant([1.,2.,3.,4.,5.,6.],shape=[2,3],name='a')
b=tf.constant([1.,2.,3.,4.,5.,6.],shape=[3,2],name='b')
c=tf.matmul(a,b)
#print(sess.run(c))

#允许使用不同的设备
config=tf.ConfigProto()

config.allow_soft_placement=True
config.log_device_placement=True
sess_soft=tf.Session(config=config)
#在默认情况下，即使机器有多个CPU，TensorFlow也不会区分它们，所有的CPU都使用/cpu:0作为名称。


print(sess_soft.run(c))
