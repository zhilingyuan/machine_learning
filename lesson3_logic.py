import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops
import requests
import os.path
print(os.path.abspath('.'))
ops.reset_default_graph()
sess=tf.Session()
birth_weight_file='birth_weight.csv'
if not os.path.exists(birth_weight_file):
    birthdata_url='https://github.com/nfmcclure/tensorflow_cookbook/raw/master/\
01_Introduction/07_Working_with_Data_Sources/birthweight_data/birthweight.dat'
    birth_file=requests.get(birthdata_url)
    
