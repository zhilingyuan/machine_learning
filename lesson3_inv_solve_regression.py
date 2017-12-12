import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
sess=tf.Session()
x_vals=np.linspace(0,10,100)
y_vals=x_vals+np.random.normal(0,1,100)

x_vals_column=np.transpose(np.matrix(x_vals))
ones_column=np.transpose(np.repeat(1,100))
A=np.column_stack((x_vals_column,ones_column))
b=np.transpose(np.matrix(y_vals))
print('A='+str(A))
print('b='+str(b))

A_tensor=tf.constant(A)
b_tensor=tf.constant(b)

tA_A=tf.matmul(tf.transpose(A_tensor),A_tensor)
tA_A_inv=tf.matrix_inverse(tA_A)
product=tf.matmul(tA_A_inv,tf.transpose(A_tensor))
solution=tf.matmul(product,b_tensor)
solution_val=sess.run(solution)
print('ans='+str(solution_val))
slope=solution_val[0][0]
y_intercept=solution_val[1][0]

best_fit=[]
for i in x_vals:
    best_fit.append(slope*i+y_intercept)
plt.plot(x_vals,y_vals,'o',label='Data')
plt.plot(x_vals,best_fit,'r-',label='best fit line')
plt.legend(loc='upper left')
plt.show()
