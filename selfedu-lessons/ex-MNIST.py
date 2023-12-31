import os
from typing import Any
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

import numpy as np


with np.load('selfedu-lessons/mnist.npz') as f:
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']


x_train = x_train/255
x_test = x_test/255

x_train = tf.reshape(tf.cast(x_train, tf.float32), [-1, 28*28])
x_test = tf.reshape(tf.cast(x_test, tf.float32), [-1, 28*28])

y_train = to_categorical(y_train, 10)


class DenseNN(tf.Module):
    def __init__(self, outputs, activate='relu'):
        super().__init__()
        self.outputs = outputs
        self.activate = activate
        self.fl_init = False
    
    def __call__(self, x):
        if not self.fl_init:
            self.w = tf.random.truncated_normal([x.shape[-1], self.outputs], stddev=0.1, name="w")
            self.b = tf.zeros([self.outputs], dtype=tf.float32, name="b")

            self.w = tf.Variable(self.w)
            self.b = tf.Variable(self.b)

            self.fl_init = True
        
        y = tf.matmul(x, self.w) + self.b

        if self.activate == 'relu':
            return tf.nn.relu(y)
        elif self.activate == 'softmax':
            return tf.nn.softmax(y)
        
        return y
    

class SequentialModule(tf.Module):
    def __init__(self):
        super().__init__()
        self.layers = []
    
    def add(self, layer):
        return self.layers.append(layer)

    def __call__(self, x):
        y = x

        for layer in self.layers:
            y = layer(y)
        
        return y


model = SequentialModule()
model.add(DenseNN(128))
model.add(DenseNN(10, activate='softmax'))

cross_entropy = lambda y_true, y_pred: tf.reduce_mean(tf.losses.categorical_crossentropy(y_true, y_pred))
opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.01)

BATCH_SIZE = 32
EPOCHS = 10
TOTAL = x_train.shape[0]

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE)

for n in range(EPOCHS):
    loss = 0
    for x_batch, y_batch in train_dataset:
        with tf.GradientTape() as tape:
            f_loss = cross_entropy(y_batch, model(x_batch))
        
        loss += f_loss
        grads = tape.gradient(f_loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
    
    print(loss.numpy())


y = model(x_test)
y2 = tf.argmax(y, axis=1).numpy()
acc = len(y_test[y_test == y2]) / y_test.shape[0] * 100
print(acc)
