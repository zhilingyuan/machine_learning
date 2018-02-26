import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

file='1.bmp'

file_contents=tf.read_file(file)
image=tf.image.decode_image(file_contents)
 
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    img=sess.run(image)
    plt.figure(1)
    plt.imshow(img)
    plt.show()
    img=img[:,:,1]
    img=np.squeeze(img)
    
