import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as v1

import matplotlib.pyplot as plt


SAMPLES = 50  # количество точек
PACKET_SIZE = 5  # размер пакета


def f(x):  # искомая функция
    return (2*x  - 3)


X_0 = -2  # начало интервала
X_LAST = 2  # конец интервала
SIGMA = 0.5  # среднеквадратичное отклонение шума


v1.disable_eager_execution()  # отключаем активное выполнение

np.random.seed(0)  # делаем случайность предсказуемой (чтобы можно было повторить вычисления на этом же наборе данных)
data_x = np.arange(X_0, X_LAST, (X_LAST - X_0) / SAMPLES)  # массив [-2, -1.92, -1.84, ..., 2]
np.random.shuffle(data_x)  # перемешать
data_y = list(map(f, data_x)) + np.random.normal(0, SIGMA, SAMPLES)  # массив значений функции с шумом
print(",".join(list(map(str, data_x[:PACKET_SIZE]))))  # первый пакет иксов
print(",".join(list(map(str, data_y[:PACKET_SIZE]))))  # первый пакет игреков

print("DATA_X=", data_x)

tf_data_x = v1.placeholder(tf.float32, shape=(PACKET_SIZE, ))  # узел на который будет подаваться аргумент функции
tf_data_y = v1.placeholder(tf.float32, shape=(PACKET_SIZE, ))  # узел на который будет подаваться значение функции

weight = tf.Variable(initial_value=0.1, dtype=tf.float32, name="a")
bias = tf.Variable(initial_value=0.0, dtype=tf.float32, name="b")
model = tf.add(tf.multiply(tf_data_x, weight), bias)

loss = tf.reduce_mean(tf.square(model - tf_data_y))  # функция потерь
optimizer = v1.train.GradientDescentOptimizer(0.5).minimize(loss)  # метод оптимизации

with v1.Session() as session:
    v1.global_variables_initializer().run()

    for i in range(SAMPLES//PACKET_SIZE):
        feed_dict = {tf_data_x: data_x[i * PACKET_SIZE: (i + 1) * PACKET_SIZE], tf_data_y: data_y[i * PACKET_SIZE: (i + 1) * PACKET_SIZE]}
        _, l = session.run([optimizer, loss], feed_dict=feed_dict)  # запускаем оптимизацию и вычисляем "потери"
        print("Ошибка: %f" % (l,))
        print("a = %f, b = %f" % (weight.eval(), bias.eval()))
    
    plt.plot(data_x, list(map(lambda x: weight.eval() * x + bias.eval(), data_x)), data_x, data_y, 'ro')

