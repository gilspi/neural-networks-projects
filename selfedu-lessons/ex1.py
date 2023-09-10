import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


import tensorflow as tf
import numpy as np


a = tf.constant(1, shape=(1, 1))  # переменная в tensorflow не изменяемая
b = tf.constant([1, 2, 3, 4])
c = tf.constant([[1, 2],
                 [3, 4],
                 [5, 6]], dtype=tf.float16)
print(c)


a2 = tf.cast(a, dtype=tf.float32)
print(a2)


b2 = np.array(b)
print(b2)


v1 = tf.Variable(-1.2)
v2 = tf.Variable([4, 5, 6, 7], dtype=tf.float32)
v3 = tf.Variable(b)

v1.assign(0)
v2.assign([0, 1, 6, 7])

print(v1, v2, v3, sep="\n\n")