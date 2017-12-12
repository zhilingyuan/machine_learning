#分解矩阵高效稳定，公式Ax=b Lx=A'T A'A=L'L A'Ax1=L'Lx A'b=L'Lx
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()
sess=tf.Session()
x_vals=np.linspace(0,10,100)
y_vals=x_vals+np.random.normal(0,1,100)
x_vals_column=np.transpose(np.matrix(x_vals))
ones_column=np.transpose(np.matrix(np.repeat(1,100)))
A=np.column_stack((x_vals_column,ones_column))
print(A)
b=np.transpose(np.matrix(y_vals))
print(b)
A_tensor=tf.constant(A)
b_tensor=tf.constant(b)

tA_A=tf.matmul(tf.transpose(A_tensor),A_tensor)
L=tf.cholesky(tA_A)#返回下三角矩阵，A=L*L'(上三角与下三角互为转置)
                    # 返回下三角所以和第一行公式理论推导（数学上）有出路
print(sess.run(L))
tA_b=tf.matmul(tf.transpose(A_tensor),b)
sol1=tf.matrix_solve(L,tA_b)
print(sess.run(sol1))
sol2=tf.matrix_solve(tf.transpose(L),sol1)

solution_eval=sess.run(sol2)
slope=solution_eval[0][0]
y_intercept=solution_eval[1][0]
print('slope:'+str(slope))
print('y_intercept:'+str(y_intercept))

best_fit=[]
for i in x_vals:
    best_fit.append(slope*i+y_intercept)
plt.plot(x_vals,y_vals,'o',label='DATA')
plt.plot(x_vals,best_fit,'r-',label='fit line')
plt.legend(loc='upper left')
plt.show()#理论和lesson3 逆矩阵拟合是一样的
