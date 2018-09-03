##############################################################################
######################BASICS IN TENSORFLOW
############################################################
import tensorflow as tf
import numpy as np
p=tf.constant([2,3,4,5,7], dtype= tf.int32)
o = tf.ones([5],dtype= tf.int32)
ad=tf.add(p,o)
m= ad*p
print(m)
