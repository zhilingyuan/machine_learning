import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

file='1.bmp'

file_contents=tf.read_file(file)
image=tf.image.decode_image(file_contents)
 
with tf.Session() as sess:
    #sess.run(tf.global_variables_initializer())
    img=sess.run(image)
    #plt.figure(1)
    #plt.imshow(img)
    #plt.show()
    img=img[:,:,1]
    img=np.squeeze(img)
    img_ty=img.astype(np.float32)
    img_ty=img_ty/256
    input_img=tf.placeholder(shape=[300,300],dtype=tf.float32)
    output_img=tf.Variable(tf.random_normal(shape=[300,300]))
    lamda=2.1
    tv1=tf.subtract(tf.slice(output_img,[0,0],[299,299]),tf.slice(output_img,[0,1],[299,299]))
    tv2=tf.subtract(tf.slice(output_img,[0,1],[299,299]),tf.slice(output_img,[1,0],[299,299]))
    tv=tf.multiply(lamda,tf.add(tf.nn.l2_loss(tv1),tf.nn.l2_loss(tv2)))
    
    loss1=tf.nn.l2_loss(tf.subtract(input_img,output_img))
    loss=tf.reduce_mean(tf.add(loss1,tv))
    init=tf.global_variables_initializer()
    sess.run(init)
    myopt=tf.train.GradientDescentOptimizer(0.1)
    train_step=myopt.minimize(loss)
    for i in range(5000):
        sess.run(train_step,feed_dict={input_img:img_ty})
        temp_loss=sess.run(loss1,feed_dict={input_img:img_ty})
        if(i%500==0):
            print(str(temp_loss))
    print((sess.run(output_img))*256)
    print((sess.run(output_img)).shape)
    plt.imshow((sess.run(output_img))*256)
    plt.show()
    
    
