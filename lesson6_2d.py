import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
ops.reset_default_graph()
sess=tf.Session()
data_size=[10,10]
data_2d=np.random.normal(size=data_size)
x_input_2d=tf.placeholder(dtype=tf.float32,shape=data_size)
#dtype and shape position can switch
def conv_layer_2d(input_2d,filter):
	input_3d=tf.expand_dims(input_2d,0)
	input_4d=tf.expand_dims(input_3d,3)
	convolution_output=tf.nn.conv2d(input_4d,filter,strides=[1,2,2,1],padding='VALID')
	conv_output_2d=tf.squeeze(convolution_output)
	return(conv_output_2d)

my_filter=tf.Variable(tf.random_normal(shape=[2,2,1,1]))#why not [1,2,2,1]
my_convolution_output=conv_layer_2d(x_input_2d,my_filter)

def activation(input_2d):
	return(tf.nn.relu(input_2d))

my_activation_output=activation(my_convolution_output)

def max_pool(input_2d,width,height):
	input_3d=tf.expand_dims(input_2d,0)
	input_4d=tf.expand_dims(input_3d,3)
	pool_output=tf.nn.max_pool(input_4d,ksize=[1,height,width,1],strides=[1,1,1,1],
			padding='VALID')
	pool_output_2d=tf.squeeze(pool_output)
	return(pool_output_2d)

my_maxpool_output=max_pool(my_activation_output,width=2,height=2)

def fully_connected(input_layer,num_outputs):
	flat_input=tf.reshape(input_layer,[-1])# -1 means without define and max
					       # try differences in 1 
	weight_shape=tf.squeeze(tf.stack([tf.shape(flat_input),[num_outputs]]))
	weight=tf.random_normal(weight_shape,stddev=0.1)
	bias=tf.random_normal(shape=[num_outputs])
	input_2d=tf.expand_dims(flat_input,0)
	full_output=tf.add(tf.matmul(input_2d,weight),bias)
	full_output_2d=tf.squeeze(full_output)
	return(full_output_2d)

my_full_output=fully_connected(my_maxpool_output,5)

init=tf.global_variables_initializer()
sess.run(init)
print(sess.run(my_filter))

feeddict={x_input_2d:data_2d}
print('input array size 10 X 10')
print('2*2 convolution strides=[2 2],result in 5*5')
print(sess.run(my_convolution_output,feed_dict=feeddict))
print('relu element wise of above')
print(sess.run(my_activation_output,feed_dict=feeddict))
print('maxpool,stride=[1 1],result in 4*4')
print(sess.run(my_maxpool_output,feed_dict=feeddict))
print('all connected ')
print(sess.run(my_full_output,feed_dict=feeddict)) 
