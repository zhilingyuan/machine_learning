import random
import string
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
ops.reset_default_graph()
n=10
street_names=['abbey','baker','canal','donner','elm']
street_types=['rd','st','ln','pass','ave']
rand_zips=[random.randint(65000,65999) for i in range(5)]
numbers=[random.randint(1,9999) for i in range(n)]
streets=[random.choice(street_names) for i in range(n)]
street_snuffs=[random.choice(street_types) for i in range(n)]
zips=[random.choice(rand_zips) for i in range(n)]
full_streets=[str(x)+' '+y+' '+z for x,y,z in zip(numbers,streets,street_snuffs)]
reference_data=[list(x) for x in zip(full_streets,zips)]

#make errors in address
def create_typo(s,prob=0.75):
    if random.uniform(0,1)<prob:
        rand_ind=random.choice(range(len(s)))
        s_list=list(s)
        s_list[rand_ind]=random.choice(string.ascii_lowercase)
        s=''.join(s_list)
    return(s)
typo_streets=[create_typo(x) for x in streets]
typo_full_streets=[str(x)+' '+y+' '+z for x,y,z in
                   zip(numbers,typo_streets,street_snuffs)]
test_data=[list(x) for x in zip(typo_full_streets,zips)]

sess=tf.Session()
test_address=tf.sparse_placeholder(dtype=tf.string)
test_zip=tf.placeholder(shape=[None,1],dtype=tf.float32)
ref_address=tf.sparse_placeholder(dtype=tf.string)
ref_zip=tf.placeholder(shape=[None,n],dtype=tf.float32)

#tf.gather 取对应indice的param的值
zip_dist=tf.square(tf)

