import numpy as np


def act(x):
    return 1 if x >= 0.5 else 0


def start(house, rock, pretty):
    x = np.array([house, rock, pretty])
    w11 = [0.3, 0.3, 0]
    w12 = [0.4, -0.5, 1]
    weight1 = np.array([w11, w12])
    weight2 = np.array([-1, 1])

    sum_hidden = np.dot(weight1, x)
    print('Sum on a hidden layer: ', sum_hidden)

    sum_out = np.array([act(x) for x in sum_hidden])
    print('Sum on a hidden out layer: ', sum_out)

    sum = np.dot(weight2, sum_out)
    y = act(sum)

    print('Output value: ', y)

    return y


house = 1
rock = 0
pretty = 1

result = start(house, rock, pretty)
if result == 1:
    print('Go married!')
else:
    print('Go away...')

