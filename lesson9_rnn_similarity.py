#构建双向RNNN
#fully connected layer
#cosine similarity
import os
import random
import string
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
sess=tf.Session()

batch_size=200
n_batches=300
max_address_len=20
margin=0.25
num_features=50
dropout_keep_prob=0.8#保留的概率 但总和元素值不变，为保留元素 */0.8

#创建孪生RNN 相似度模型 
def snn(address1,address2,dropout_keep_prob,
        vocab_size,num_features,input_length):
    def siamese_nn(input_vector,num_hidden):
        cell_unit=tf.nn.rnn_cell.BasicLSTMCell

        lstm_forword_cell=cell_unit(num_hidden,forget_bias=1.0)
        lstm_forward_cell=tf.nn.rnn_cell.DropoutWrapper(lstm_forward_cell,output_keep_prob=dropout_keep_prob)

        lstm_backward_cell=cell_unit(num_hidden,forget_bias=1.0)
        lstm_backward_cell=tf.nn.rnn_cell.DropoutWrapper(lstm_bacward_cell,output_keep_prob=dropout_keep_prob)

        input_embed_split=tf.split(1,input_length,input_vector)#分裂
        input_embed_split=[tf.squeeze(x,squeeze_dims=[1]) for x in input_embed_split]

        outputs,_,_=tf.contrib.rnn.static_bidirectional_rnn(lstm_forward_cell,
                                                            lstm_backward_cell,
                                                            input_embed_split,
                                                            dtype=tf.float32)
        temporal_mean=tf.ad_n(outputs)/input_length
        output_size=10
        A=tf.get_variable(name='A',shape=[2*num_hidden,output_size],
                          dtype=tf.float32,
                          initializer=tf.random_normal_initializer(stddev=0.1))
        b=tf.get_variable(name='b',shape=[output_size],dtype=tf.float32,
                          initializer=tf.random_normal_initializer(stddev=0.1))

        final_output=tf.matmul(temporal_mean,A)+b
        final_output=tf.nn.dropout(final_output,dropout_keep_prob)
        return(final_output)

    with tf.variable_scope("siamese") as scope:#复用
        output1=siamese_nn(address1,num_features)
        scope.reuse_variables()
        output2=siamese_nn(address2,num_features)

    output1=tf.nn.l2_normalize(output1,1)
    output2=tf.nn.l2_normalize(output2,1)

    dot_prod=tf.reduce_sum(tf.multiply(output1,output2),1)

    return(dot_prod)

def get_predictions(scores):
    predictions=tf.sign(socres,name="predictions")
    return(predictions)

def loss(scores,y_target,margin):
    pos_loss_term=0.25*tf.square(tf.sub(1.,scores))
    pos_mult=tf.cast(y_target,tf.float32)

    neg_mult=tf.sub(1.,tf.cast(y_target,tf.float32))
    positive_loss=tf.multiply(pos_mult,pos_loss_term)
    loss=tf.add(positive_loss,neg_mult)
    target_zero=tf.equal(tf.cast(y_target,tf.float32),0.)
    less_than_margin=tf.less(scores,margin)

    both_logical=tf.logical_and(target_zero,less_than_margin)
    both_logical=tf.cast(both_logical,tf.float32)

    multiplicative_factor=tf.cast(1.-both_logical,tf.float32)
    total_loss=tf.multiply(loss,multiplicative_factor)

    avg_loss=tf.reduce_mean(total_loss)
    return(avg_loss)

def create_typo(s):
    rand_ind=random.choice(range(len(s)))
    s_list=list(s)
    s_list[rand_ind]=random.choice(string.ascii_lowercase+'0123456789')
    s=''.join(s_list)
    return(s)

street_names=['abbey','baker','canal','donner'


    
