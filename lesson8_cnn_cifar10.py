import os
import sys
import tarfile
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from six.moves import urllib#这里用sixmove 是为了兼容性
from tensorflow.python.framework import ops
ops.reset_default_graph()

abspath=os.path.abspath(__file__)
dname=os.path.dirname(abspath) #上一级文件的位置

learning_rate=0.1
lr_decay=0.1
num_gens_to_wait=250.

image_vec_length=image_height*image_width*num_channels
record_length=1+image_vec_length#+1 label

data_dir = 'temp'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
cifar10_url = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'

data_file = os.path.join(data_dir, 'cifar-10-binary.tar.gz')
if os.path.isfile(data_file):
    pass
else:
    # Download file
    def progress(block_num, block_size, total_size):
        progress_info = [cifar10_url, float(block_num * block_size) / float(total_size) * 100.0]
        print('\r Downloading {} - {:.2f}%'.format(*progress_info), end="")
    filepath, _ = urllib.request.urlretrieve(cifar10_url, data_file, progress)
    # Extract file
    tarfile.open(filepath, 'r:gz').extractall(data_dir)
