# skip-gram 模型
# 方法名字：word2vec
# 如何理解相互关联的单词
# king-man+woman=queen
# india pale ale-hops+malt=stout

#skip-gram 根据目标单词预测上下文
#cbow continuous bag of words 根据上下文预测目标单词
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import string
import requests
import collections
import io
import tarfile
import urllib.request
from nltk.corpus import stopwords
from tensorflow.python.framework import ops
ops.reset_default_graph()
sess=tf.Session()
bathc_size=50
embedding_size=200
vocabulary_size=10000
generations=50000
print_loss_every=500
num_sampled=int(batch_size/2)
window_size=2
stops=stopword.word('english')
print_valid_every=2000
valid_words=['cliche','love','hate','silly','sad']

def load_movie_data():
    save_folder_name='temp'
    pos_file=os.path.join(save_folder_name,'rt-polarity.pos')
    neg_file=os.path.join(save_folder_name,'rt-polarity.neg')
    if os.path.exists(save_folder_name):
        pos_data=[]
        with open(pos_file,'r') as temp_pos_file:
            for row in temp_pos_file:
                pos_data.append(row)
        neg_data=[]
        with open(neg_file,'r') as temp_neg_file:
            for row in temp_neg_file:
                neg_data.append(row)
    else:
        movie_data_url='http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz'
        stream_data=urlib.request.urlopen(movie_data_url)
        tmp=io.BytesIO()
        while True:
            s=steam_data.read(16384)
            if not s:
                break
            tmp.write(s)
            stream_data.close()
            tmp.seek(0)
        
            
