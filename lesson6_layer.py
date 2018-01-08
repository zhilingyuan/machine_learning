import tensorflow as tf
import numpy as np
import pdb
from tensorflow.python.framework import ops
ops.reset_default_graph()
sess=tf.Session()
data_size=25
data_1d=np.random.normal(size=data_size)
x_input_1d=tf.placeholder(dtype=tf.float32,shape=[data_size])
#most layer were designed to 4d[batch size , width ,height ,channels]
#using expand_dims and squeeze to expand or reduce dims
def conv_layer_1d(input_1d,my_filter):
    input_2d=tf.expand_dims(input_1d,0)
    input_3d=tf.expand_dims(input_2d,0)
    input_4d=tf.expand_dims(input_3d,3)
    convolution_output=tf.nn.conv2d(input_4d,filter=my_filter,strides=[1,1,1,1],
                                    padding='VALID')
    conv_output_1d=tf.squeeze(convolution_output)
    return(conv_output_1d)

my_filter=tf.Variable(tf.random_normal(shape=[1,5,1,1]))
my_convolution_output=conv_layer_1d(x_input_1d,my_filter)

def activation(input_1d):
    return(tf.nn.relu(input_1d))#max(0,x)

my_activation_output=activation(my_convolution_output)
#reduce dim of charactor vector
#reduce over ployfit phenomenon
#max_pooling 2*2 non-repeating max value
#mean-pooling 2*2 non-repeating mean value 
def max_pool(input_1d,width):
    input_2d=tf.expand_dims(input_1d,0)
    input_3d=tf.expand_dims(input_2d,0)
    input_4d=tf.expand_dims(input_3d,3)
    pool_output=tf.nn.max_pool(input_4d,ksize=[1,1,width,1],strides=[1,1,1,1],
                               padding='VALID')
    pool_output_1d=tf.squeeze(pool_output)
    return(pool_output_1d)

my_maxpool_output=max_pool(my_activation_output,width=5)

def fully_connected(input_layer,num_outputs):
    weight_shape=tf.squeeze(tf.stack([tf.shape(input_layer),[num_outputs]]))
    weight=tf.random_normal(weight_shape,stddev=0.1)
    bias=tf.random_normal(shape=[num_outputs])
    input_layer_2d=tf.expand_dims(input_layer,0)
    full_output=tf.add(tf.matmul(input_layer_2d,weight),bias)
    full_output_1d=tf.squeeze(full_output)
    return(full_output_1d)
my_full_output=fully_connected(my_maxpool_output,5)
init=tf.global_variables_initializer()
sess.run(init)
print(sess.run(my_filter))
feeddict={x_input_1d:data_1d}
print('input= array of length 25')
print('convolution result(len is 21):')
print(sess.run(my_convolution_output,feed_dict=feeddict))
print('relu:')
print(sess.run(my_activation_output,feed_dict=feeddict))
print('maxpool:')
print(sess.run(my_maxpool_output,feed_dict=feeddict))
print('fully connected:')
print(sess.run(my_full_output,feed_dict=feeddict))
