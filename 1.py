import tensorflow as tf
node1=tf.constant(3.0,dtype=tf.float32)
node2=tf.constant(4.0)
print(node1,node2)
sess=tf.Session();
print(sess.run([node1,node2]))

node3=tf.add(node1,node2)
print(node3)
print(sess.run(node3))

a=tf.placeholder(tf.float32)
b=tf.placeholder(tf.float32)
adder_node=a+b

print(sess.run(adder_node,{a:3,b:4}))
print(sess.run(adder_node,{a:[1.0,3.0],b:[2.0,4.0]}))

add_and_triple=adder_node*3
print(sess.run(add_and_triple,{a:3,b:4.5}))  #类似lambda 函数
      
w=tf.Variable([.3],dtype=tf.float32)
b=tf.Variable([-.3],dtype=tf.float32)
x=tf.placeholder(tf.float32)
linear_model=w*x+b

print(w)
#print(sess.run(w)) 此时变量没有初始化
init=tf.global_variables_initializer()
sess.run(init)
print(sess.run(w))
print(sess.run(linear_model,{x:[1,2,3,4]}))

y=tf.placeholder(tf.float32)
squared_deltas=tf.square(linear_model-y)
loss=tf.reduce_sum(squared_deltas)
print(sess.run(loss,{x:[1,2,3,4],y:[0,-1,-2,-3]}))

fixw=tf.assign(w,[-1.])
fixb=tf.assign(b,[1.])
sess.run([fixw,fixb])
print(sess.run(loss,{x:[1,2,3,4],y:[0,-1,-2,-3]}))

optimizer=tf.train.GradientDescentOptimizer(0.01)
train=optimizer.minimize(loss)
sess.run(init)
for i in range(10000):
    sess.run(train,{x:[1,2,3,4],y:[0,-1,-2,-3]})    
print(sess.run([w,b]))
