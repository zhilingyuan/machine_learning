#rnn 是为了解决序列数据的问题 进行预测
#y(t)=fun[y(t-1)+x(t)]
#区别 卡尔曼滤波器  导航 和 跟踪目标  修正了速度v
#x（k）=Ax（k-1）+bu(k-1)+q(k-1）系统自身变化
#y（k）=Hx（k）+r（k）测量系统
#卡尔曼 和 粒子 都需要概率 预测
'''
alman Filter中的高斯和线性假设有时并不能满足实际情况，因为高斯模型是单峰的（uni-modal）。比如当一个运动目标被遮挡后，我们不知道它将在遮挡物的后面做什么运动，是继续前进，或是停下，还是转向或后退。这种情况下，Kalman Filter只能给出一个位置的预测，至多加大这个预测的不确定性（即增大协方差矩阵，或，若state是单维度时的方差）。这种情况下，预测位置的不确定噪音事实上已不是高斯模型，它将具有多个峰值（multi-modal）。而且，这种噪音常常无法解析表达。这就引入了Particle Filter。

Particle Filter是基于蒙特卡洛方法（Monte Carlo Method），简言之就是用点来表达概率密度函数。点密集的地方，概率密度就大。在时间序列中，就是时间序列蒙特卡洛（Sequential Monte Carlo）。所以Particle Filter在机器视觉中的应用，称为CONDENSATION（Conditional Density Propagation），它和Bootstrap Filter一样，同属于Sequential Monte Carlo范畴。

具体实施上，PF对state的更新不再采用KF中的高斯模型更新，而是采用factored sampling方法。简单说就是对t-1时刻的所有Particle，根据每个Particle的概率，重新对他们采样。高概率的Particle将得到高的采样几率，而低概率的Particle对应低的采样几率。这样，高概率Particle将可能被多次采样，而低概率Particle可能就被放弃。这样得到t时刻的Particle。然后将t时刻每一个Particle所对应的测量值结合起来，为t时刻的Particle重新赋以新的概率，以用于t+1时刻新Particle的生成。

所以可以总结， KF和PF相同的是，都分为三个步骤：Prediction，Measurement和Assimilation（或称为correction）。只是每步的实现上不同。

在Prediction阶段，KF利用高斯模型进行预测，而PF采用factored sampling对原Particle重采样。

'''
#RNN 更新更好得方法
#受限玻尔兹曼 有方向 玻尔兹曼是随机过程 目前没用
#波尔兹曼分布 指数e^-a/e^-b
#粒子数转移的概率


#所谓的规则学习 就是语义理解的一部分

import os
import re
import io
import requests
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from zipfile import ZipFile

sess=tf.Session()
epochs=20
batch_size=250
max_sequence_length=25#短信长度 不够的填充0 长的截断
rnn_size=10
embedding_size=50
min_word_frequency=10
learning_rate=0.0005
dropout_keep_prob=tf.placeholder(tf.float32)

data_dir='temp'
data_file='text_data.txt'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
if not os.path.isfile(os.path.join(data_dir,data_file)):
    zip_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
    r=requests.get(zip_url)
    z=ZipFile(io.BytesIO(r.content))
    file=z.read('SMSSpamCollection')
    text_data=file.decode()
    text_data=text_data.encode('ascii',errors='ignore')
    text_data=text_data.decode().split('\n')
    with open(os.path.join(data_dir,data_file),'w') as file_conn:
        for text in text_data:
            file_conn.write("{}\n".format(text))
else:
    text_data=[]
    with open(os.path.join(data_dir,data_file),'r') as file_conn:
        for row in file_conn:
            text_data.append(row)
    text_data=text_data[:-1]
text_data=[x.split('\t') for x in text_data if len(x)>=1]
[text_data_target,text_data_train]=[list(x) for x in zip(*text_data)]#？？？制表符 分割 成为list

def clean_text(text_string):
    text_string=re.sub(r'([^\s\w]|_|[0-9])+','',text_string)#r'' 禁用转义符 sub 替换
    text_string=" ".join(text_string.split())
    text_string=text_string.lower()
    return(text_string)

text_data_train=[clean_text(x) for x in text_data_train]

vocab_processor=tf.contrib.learn.preprocessing.VocabularyProcessor(max_sequence_length,
                                                                   min_frequency=min_word_frequency)
text_processed=np.array(list(vocab_processor.fit_transform(text_data_train)))

text_processed=np.array(text_processed)
text_data_target=np.array([1 if x=='ham' else 0 for x in text_data_target])
shuffled_ix=np.random.permutation(np.arange(len(text_data_target)))#arange = array[(range)]
x_shuffled=text_processed[shuffled_ix]
y_shuffled=text_data_target[shuffled_ix]

ix_cutoff=int(len(y_shuffled)*0.80)
x_train,x_test=x_shuffled[:ix_cutoff],x_shuffled[ix_cutoff:]
y_train,y_test=y_shuffled[:ix_cutoff],y_shuffled[ix_cutoff:]
vocab_size=len(vocab_processor.vocabulary_)
print("vocabulary size:{:d}".format(vocab_size))
print("80-20 Train Test split:{:d} -- {:d}".format(len(y_train),len(y_test)))

x_data=tf.placeholder(tf.int32,[None,max_sequence_length])
y_output=tf.placeholder(tf.int32,[None])

embedding_mat=tf.Variable(tf.random_uniform([vocab_size,embedding_size],
                                            -1.0,1.0))
embedding_output=tf.nn.embedding_lookup(embedding_mat,x_data)#embedding_lookup(params, ids)其实就是按照ids顺序返回params中的第ids行。

if tf.__version__[0]>='1':
    cell=tf.contrib.rnn.BasicRNNCell(num_units=rnn_size)
else:
    cell=tf.nn.rnn_cell.BasicDRNNCell(num_units=rnn_size)

output,state=tf.nn.dynamic_rnn(cell,embedding_output,dtype=tf.float32)#RNN
output=tf.nn.dropout(output,dropout_keep_prob)

output=tf.transpose(output,[1,0,2])
last=tf.gather(output,int(output.get_shape()[0])-1)

weight=tf.Variable(tf.truncated_normal([rnn_size,2],stddev=0.1))
bias=tf.Variable(tf.constant(0.1,shape=[2]))
logits_out=tf.matmul(last,weight)+bias

losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_out, labels=y_output) # logits=float32, labels=int32
loss = tf.reduce_mean(losses)

accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits_out,1),tf.cast(y_output,tf.int64)),tf.float32))
optimizer=tf.train.RMSPropOptimizer(learning_rate)
train_step=optimizer.minimize(loss)

init=tf.global_variables_initializer()
sess.run(init)

train_loss = []
test_loss = []
train_accuracy = []
test_accuracy = []
# Start training
for epoch in range(epochs):

    # Shuffle training data
    shuffled_ix = np.random.permutation(np.arange(len(x_train)))
    x_train = x_train[shuffled_ix]
    y_train = y_train[shuffled_ix]
    num_batches = int(len(x_train)/batch_size) + 1
    # TO DO CALCULATE GENERATIONS ExACTLY
    for i in range(num_batches):
        # Select train data
        min_ix = i * batch_size
        max_ix = np.min([len(x_train), ((i+1) * batch_size)])
        x_train_batch = x_train[min_ix:max_ix]
        y_train_batch = y_train[min_ix:max_ix]
        
        # Run train step
        train_dict = {x_data: x_train_batch, y_output: y_train_batch, dropout_keep_prob:0.5}
        sess.run(train_step, feed_dict=train_dict)
        
    # Run loss and accuracy for training
    temp_train_loss, temp_train_acc = sess.run([loss, accuracy], feed_dict=train_dict)
    train_loss.append(temp_train_loss)
    train_accuracy.append(temp_train_acc)
    
    # Run Eval Step
    test_dict = {x_data: x_test, y_output: y_test, dropout_keep_prob:1.0}
    temp_test_loss, temp_test_acc = sess.run([loss, accuracy], feed_dict=test_dict)
    test_loss.append(temp_test_loss)
    test_accuracy.append(temp_test_acc)
    print('Epoch: {}, Test Loss: {:.2}, Test Acc: {:.2}'.format(epoch+1, temp_test_loss, temp_test_acc))
    
# Plot loss over time
epoch_seq = np.arange(1, epochs+1)
plt.plot(epoch_seq, train_loss, 'k--', label='Train Set')
plt.plot(epoch_seq, test_loss, 'r-', label='Test Set')
plt.title('Softmax Loss')
plt.xlabel('Epochs')
plt.ylabel('Softmax Loss')
plt.legend(loc='upper left')
plt.show()

# Plot accuracy over time
plt.plot(epoch_seq, train_accuracy, 'k--', label='Train Set')
plt.plot(epoch_seq, test_accuracy, 'r-', label='Test Set')
plt.title('Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')
plt.show()

    

