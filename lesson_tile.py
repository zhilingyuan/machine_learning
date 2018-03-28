import tensorflow as tf 
import numpy as np 

sess = tf.Session()
data = tf.constant([[1, 2, 3, 4], [9, 8, 7, 6]])
d = tf.tile(data, [2,3])
print(sess.run(d))

